// End-to-end model build: table -> (CRF scorer) -> (lexicon) -> binary -> decode -> evaluate.

import { decodeModel as decodeCore } from "../src/codec";
import { CLS_NOT_COUNTER, emptyScorer, type Model } from "../src/model";
import { applyVariant, variantsOf } from "../src/phonology";
import { readerOf } from "./read";
import { trainCrf, type CrfConfig, type Trained } from "./crf";
import { rng, type Example, type KanjiInfo } from "./data";
import { encodeModel, type ModelSpec } from "./encode";
import { buildTable, collectStats, variantPenalties, type TableConfig } from "./table";
import { selectCompoundRobust, selectLexicon, selectRepairs, type LexConfig } from "./lexicon";
import { align } from "./align";
import { okuriganaVariants } from "./okurigana";
import { twoLoop, type ChunkConfig } from "./hooks";
import { loadHiraWords } from "./data";

/** Bump when src/features.ts changes so cached CRF weights are not reused. */
export const FEATURE_VERSION = 6;

export type BuildConfig = {
  name: string;
  table: TableConfig;
  priorBits: number;
  quantBits: number;
  crf?: Omit<CrfConfig, "log">;
  lex?: LexConfig;
  trainFile?: string; // "train_all" = release model (nothing held out; public test sets become invalid)
  foodLex?: number; // multiply lexicon-selection weight of food-domain words
  okuriAug?: number; // add okurigana-omitted variants (焼そば, 盛合せ) to CRF training with this relative weight
  counters?: { minN: number; minShare: number }; // counter readings after a digit (3種 -> 3しゅ, 3種盛り -> 3しゅもり), see counterReadings
  digitAug?: number; // add Arabic-digit variants of numeral + counter words (三種 -> 3種) to CRF training with this weight
  okuriLex?: number; // add okurigana-omitted variants only as exception-lexicon candidates (CRF weights untouched)
  okuriExtra?: number; // extra bytes for okurigana-omitted variants, added on top of the lexicon (nothing is evicted)
  compoundExtra?: { budget: number; minFail: number; prefixes: number }; // compound-robust entries on top of the lexicon
  repairExtra?: number; // bytes for words that the lexicon itself breaks (盛り=さかり inside 大盛り)
  chunk?: ChunkConfig; // pass-2 chunk-boundary scorer (experimental two-loop decoder)
  lexScore?: number; // score per character of a lexicon node (default 4 = practically an override)
  placeFameLex?: boolean; // apply the fame weighting only when choosing lexicon entries
  placeFame?: boolean; // weight place names by how many other place names contain them (fame proxy)
  foodBoost?: number; // multiply the weight of food-domain words (JMdict field "food")
  target?: number; // total byte budget: the lexicon gets whatever the table + scorer leave
  keep?: number; // keep only this many largest-magnitude hashed weights (emit + trans), 0/undefined = all
};

/** Magnitude pruning: zero all but the `keep` largest |w| across the given arrays. */
export const prune = (arrs: Float32Array[], keep: number): Float32Array[] => {
  const mags = arrs.flatMap((a) => Array.from(a, Math.abs)).filter((x) => x > 0).sort((a, b) => b - a);
  if (mags.length <= keep) return arrs;
  const th = mags[keep];
  return arrs.map((a) => a.map((x) => (Math.abs(x) > th ? x : 0)));
};

/** Place-name "fame": number of distinct other place surfaces that contain this one (e.g. 十勝 in 十勝川, 十勝岳). */
export const withPlaceFame = (ex: readonly Example[]): Example[] => {
  const contain = new Map<string, number>();
  const surfaces = new Set(ex.filter((e) => e.source === "place").map((e) => e.surface));
  for (const s of surfaces) {
    const cs = Array.from(s), seen = new Set<string>();
    for (let i = 0; i < cs.length; i++) for (let j = i + 2; j <= Math.min(cs.length, i + 4); j++) {
      const sub = cs.slice(i, j).join("");
      if (sub !== s && !seen.has(sub)) { seen.add(sub); contain.set(sub, (contain.get(sub) ?? 0) + 1); }
    }
  }
  return ex.map((e) => (e.source === "place" ? { ...e, weight: e.weight * (1 + Math.log2(1 + (contain.get(e.surface) ?? 0))) } : e));
};

/**
 * Arabic-digit variants of words that start with a kanji numeral followed by kanji (三種 -> 3種, 一杯 -> 1杯).
 * Menus write counts with digits, but the dictionaries almost only have kanji numerals. The digit is read
 * verbatim, the counter keeps its (possibly sound-changed) reading: 一本 いっぽん -> 1本 1ぽん.
 */
const NUMERALS = "一二三四五六七八九";
export const digitVariants = (ex: readonly Example[], weight: number): Example[] => {
  const known = new Set(ex.map((e) => e.surface));
  const out: Example[] = [];
  for (const e of ex) {
    if (e.source !== "word" || !e.segs || e.cps.length < 2 || e.cps.length > 4) continue;
    const d = NUMERALS.indexOf(e.surface[0]);
    const s0 = e.segs[0];
    if (d < 0 || s0.e !== 1 || !/^[\u4e00-\u9fff]/.test(e.surface.slice(1))) continue;
    const digit = String(d + 1);
    const surface = digit + e.surface.slice(1);
    if (known.has(surface)) continue;
    known.add(surface);
    out.push({
      ...e, surface, cps: [digit.codePointAt(0)!, ...e.cps.slice(1)],
      reading: digit + e.reading.slice(s0.r.length),
      segs: [{ s: 0, e: 1, r: digit }, ...e.segs.slice(1)],
      weight: e.weight * weight, flags: e.flags + ",digit",
    });
  }
  return out;
};

/**
 * Counter readings: for each kanji that follows a kanji numeral read as a number (三種 さん|しゅ, 二杯 に|はい),
 * the candidate whose reading (up to rendaku / sokuon) is used there most often. Returns kanji -> candidate reading.
 */
const NUM_READ = /^(いち|いっ|に|さん|し|よん|よ|ご|ろく|ろっ|なな|しち|はち|はっ|く|きゅう)$/;
export const counterReadings = (ex: readonly Example[], kanji: ModelSpec["kanji"], minN: number, minShare: number): Map<number, string> => {
  const byCp = new Map(kanji.map((k) => [k.cp, k.cands]));
  const cnt = new Map<number, Map<string, number>>();
  for (const e of ex) {
    if (e.source !== "word" || !e.segs || e.segs.length < 2 || !"一二三四五六七八九".includes(e.surface[0])) continue;
    const [a, b] = e.segs;
    if (a.e !== 1 || b.e !== 2 || !NUM_READ.test(a.r)) continue;
    const cands = byCp.get(e.cps[1]);
    const hit = cands?.find((c) => variantsOf(c.reading, true, true).some((v) => applyVariant(c.reading, v) === b.r));
    if (!hit) continue;
    const m = cnt.get(e.cps[1]) ?? new Map<string, number>();
    m.set(hit.reading, (m.get(hit.reading) ?? 0) + 1);
    cnt.set(e.cps[1], m);
  }
  const out = new Map<number, string>();
  for (const [cp, m] of cnt) {
    const tot = [...m.values()].reduce((x, y) => x + y, 0);
    const [r, n] = [...m].sort((x, y) => y[1] - x[1] || (x[0] < y[0] ? -1 : 1))[0];
    if (n >= minN && n / tot >= minShare) out.set(cp, r);
  }
  return out;
};

/** KANJIDIC kun readings with okurigana markers kept (や.く), hiragana. */
export const kunOf = (dic: Map<string, KanjiInfo>): Map<string, string[]> =>
  new Map([...dic].map(([c, i]) => [c, i.kunRaw]));

export type BuildResult = { name: string; bytes: Uint8Array; model: Model; breakdown: Record<string, number> };

export const splitDev = (ex: readonly Example[], n: number, seed: number) => {
  const r = rng(seed);
  const devIdx = new Set<number>();
  const pool = ex.map((e, i) => i).filter((i) => ex[i].source === "word" && ex[i].weight >= 1);
  while (devIdx.size < Math.min(n, pool.length)) devIdx.add(pool[Math.floor(r() * pool.length)]);
  const devSurf = new Set([...devIdx].map((i) => ex[i].surface));
  return {
    train: ex.filter((e) => !devSurf.has(e.surface)),
    dev: [...devIdx].map((i) => ex[i]).filter((e, i, a) => a.findIndex((x) => x.surface === e.surface) === i),
  };
};

export const devAccuracy = (m: Model, dev: readonly Example[]) => {
  const read = readerOf(m);
  let ok = 0;
  for (const e of dev) if (read(e.surface) === e.reading) ok++;
  return ok / dev.length;
};

const sizeOf = (spec: ModelSpec, patch: Partial<ModelSpec>) => encodeModel({ ...spec, ...patch }).length;

export const build = async (cfg: BuildConfig, ex0: readonly Example[], dic: Map<string, KanjiInfo>, log = console.log): Promise<BuildResult> => {
  // models with a pass-2 chunker keep it when decoded (experimental, see train/hooks.ts)
  const decodeModel = cfg.chunk ? twoLoop().decode : decodeCore;
  const fb = cfg.foodBoost ?? 1;
  let ex = fb === 1 ? ex0 : ex0.map((e) => (e.flags.includes("food") ? { ...e, weight: e.weight * fb } : e));
  if (cfg.placeFame) ex = withPlaceFame(ex);
  const split = splitDev(ex, 2000, 7);
  const dev = split.dev;
  if (cfg.okuriAug) {
    const extra = okuriganaVariants(split.train, cfg.okuriAug, kunOf(dic));
    log(`okurigana-omitted variants: ${extra.length}`);
    split.train = [...split.train, ...extra];
  }
  if (cfg.digitAug) {
    const extra = digitVariants(split.train, cfg.digitAug);
    log(`digit variants: ${extra.length}`);
    split.train = [...split.train, ...extra];
  }
  // pass 1: table from dictionary-aligned data; pass 2: add forced-aligned KEN_ALL names
  const aligned0 = split.train.filter((e) => e.segs);
  let kanji = buildTable(collectStats(aligned0, dic, cfg.table.placeWeight), dic, cfg.table);
  const pending = split.train.filter((e) => !e.segs);
  let train: Example[] = split.train;
  if (pending.length) {
    const m1 = decodeModel(encodeModel({ kanji, lexicon: [], priorBits: 0, scorer: { ...emptyScorer(), quantBits: 4 } }));
    const got = pending.map((e) => ({ ...e, segs: align(m1, e) }));
    log(`aligned KEN_ALL ${got.filter((e) => e.segs).length}/${pending.length}`);
    train = [...aligned0, ...got];
    kanji = buildTable(collectStats(train.filter((e) => e.segs), dic, cfg.table.placeWeight), dic, cfg.table);
  }
  const base = emptyScorer();
  const m0 = decodeModel(encodeModel({ kanji, lexicon: [], priorBits: 8, scorer: { ...base, quantBits: 8 } }));
  const varPen = variantPenalties(m0, train);
  let spec: ModelSpec = { kanji, lexicon: [], priorBits: cfg.priorBits, scorer: { ...base, varPen, quantBits: cfg.quantBits } };
  let model: Model = decodeModel(encodeModel({ ...spec, priorBits: 8 }));

  if (cfg.crf) {
    const crf = cfg.crf;
    const dataKey = `${ex.length}:${ex.reduce((a, e) => a + e.weight, 0).toFixed(1)}:${cfg.okuriAug ?? 0}${cfg.digitAug ? `:d${cfg.digitAug}` : ""}`;
    const key = Bun.hash(JSON.stringify({ fv: FEATURE_VERSION, data: dataKey, t: cfg.table, crf, fb, pf: cfg.placeFame ?? false })).toString(16);
    const cachePath = new URL(`../.cache/crf-${key}.json`, import.meta.url).pathname;
    const cached = Bun.file(cachePath);
    let trained: Trained;
    if (await cached.exists()) {
      const j = await cached.json();
      const f = (x: number[]) => Float32Array.from(x);
      trained = { prior: f(j.prior), scorer: { ...j.scorer, emit: f(j.scorer.emit), out: f(j.scorer.out), bias: f(j.scorer.bias), trans: f(j.scorer.trans), varPen: f(j.scorer.varPen) } };
      log(`loaded cached CRF ${key}`);
    } else {
      trained = trainCrf(model, train, { ...crf, log }, (t) => devAccuracy({ ...model, candPrior: t.prior, scorer: t.scorer }, dev));
      const a = (x: Float32Array) => Array.from(x);
      await Bun.write(cachePath, JSON.stringify({ prior: a(trained.prior), scorer: { ...trained.scorer, emit: a(trained.scorer.emit), out: a(trained.scorer.out), bias: a(trained.scorer.bias), trans: a(trained.scorer.trans), varPen: a(trained.scorer.varPen) } }));
    }
    let ci = 0;
    const sorted = [...kanji].sort((a, b) => a.cp - b.cp);
    const learned = sorted.map((k) => ({ cp: k.cp, cands: k.cands.map((c) => ({ ...c, prior: trained.prior[ci++] })) }));
    let sc = trained.scorer;
    if (cfg.keep) { const [emit, trans] = prune([sc.emit, sc.trans], cfg.keep); sc = { ...sc, emit, trans }; }
    spec = { ...spec, kanji: learned, scorer: { ...sc, quantBits: cfg.quantBits } };
    model = decodeModel(encodeModel(spec));
  }
  if (cfg.chunk) {
    const ck = twoLoop().trainChunker(train, await loadHiraWords(), cfg.chunk, log);
    spec = { ...spec, chunker: { ...ck, quantBits: cfg.quantBits } };
    model = decodeModel(encodeModel(spec));
  }
  if (cfg.lexScore !== undefined) {
    spec = { ...spec, scorer: { ...spec.scorer, lexScore: cfg.lexScore } };
    model = decodeModel(encodeModel(spec));
  }
  if (cfg.lex) {
    // (+1: the format-v3 chunker field is not charged to the lexicon, so v2-era lexicons are reproduced exactly)
    const budget = cfg.target ? cfg.target - encodeModel(spec).length + (spec.chunker?.bits ? 0 : 1) : cfg.lex.budget;
    let lexTrain = cfg.placeFameLex ? withPlaceFame(train) : train;
    if (cfg.foodLex) lexTrain = lexTrain.map((e) => (e.flags.includes("food") ? { ...e, weight: e.weight * cfg.foodLex! } : e));
    if (cfg.okuriLex) lexTrain = [...lexTrain, ...okuriganaVariants(lexTrain, cfg.okuriLex, kunOf(dic))];
    const lexicon = budget > 0 ? selectLexicon(model, lexTrain, { ...cfg.lex, budget }, (lx) => encodeModel({ ...spec, lexicon: lx }).length) : [];
    spec = { ...spec, lexicon };
    if (cfg.okuriExtra) {
      // variants the reader still gets wrong, on top of the existing lexicon (food words first via foodLex)
      const withLex = decodeModel(encodeModel(spec));
      let vars = okuriganaVariants(lexTrain, 1, kunOf(dic));
      if (cfg.foodLex) vars = vars.map((e) => (e.flags.includes("food") ? { ...e, weight: e.weight * cfg.foodLex! } : e));
      const have = new Set(lexicon.map(([s]) => s));
      const extra = selectLexicon(withLex, vars.filter((e) => !have.has(e.surface)), { ...cfg.lex, budget: cfg.okuriExtra, minWeight: 0 },
        (lx) => encodeModel({ ...spec, lexicon: [...lexicon, ...lx] }).length);
      log(`okurigana-omitted lexicon entries: ${extra.length}`);
      spec = { ...spec, lexicon: [...lexicon, ...extra] };
    }
    if (cfg.compoundExtra) {
      // municipality stems (宇治, 十勝, 京都 ...) as left contexts: sorted by weight, spread deterministically
      const stems = [...new Set(train.filter((e) => e.flags === "kenall-city" && /^[\u4e00-\u9fff]{2,3}$/.test(e.surface)).map((e) => e.surface))].sort();
      const r0 = rng(11);
      const prefixes: string[] = [];
      while (prefixes.length < Math.min(cfg.compoundExtra.prefixes, stems.length)) {
        const s = stems[Math.floor(r0() * stems.length)];
        if (!prefixes.includes(s)) prefixes.push(s);
      }
      // product-name heads ending in okurigana (生姜焼き, 唐揚げ, 生姜入り ...) as a second kind of left context
      const heads = [...new Set(train.filter((e) => e.flags.includes("food") && /^[\u4e00-\u9fff]{1,3}[\u3041-\u3093]{1,2}$/.test(e.surface)).map((e) => e.surface))].sort();
      for (let t = 0; t < cfg.compoundExtra.prefixes / 2 && heads.length; t++) {
        const h = heads[Math.floor(r0() * heads.length)];
        if (!prefixes.includes(h)) prefixes.push(h);
      }
      log(`compound contexts: ${prefixes.join(" ")}`);
      const cur = decodeModel(encodeModel(spec));
      const curLex = spec.lexicon;
      let cand = lexTrain;
      if (cfg.foodLex) cand = cand.map((e) => (e.flags.includes("food") ? { ...e, weight: e.weight * cfg.foodLex! } : e));
      const extra2 = selectCompoundRobust(cur, cand, prefixes, { budget: cfg.compoundExtra.budget, minFail: cfg.compoundExtra.minFail, maxLen: 6 },
        (lx) => encodeModel({ ...spec, lexicon: [...curLex, ...lx] }).length, log);
      log(`compound-robust lexicon entries: ${extra2.length} (${extra2.slice(0, 12).map(([s]) => s).join(" ")} ...)`);
      spec = { ...spec, lexicon: [...curLex, ...extra2] };
    }
    if (cfg.repairExtra) {
      const full = decodeModel(encodeModel(spec));
      const curLex = spec.lexicon;
      const rep = selectRepairs(model, full, lexTrain, { budget: cfg.repairExtra, maxLen: cfg.lex.maxLen },
        (lx) => encodeModel({ ...spec, lexicon: [...curLex, ...lx] }).length, log);
      log(`lexicon repair entries: ${rep.length} (${rep.slice(0, 12).map(([s]) => s).join(" ")} ...)`);
      spec = { ...spec, lexicon: [...curLex, ...rep] };
    }
  }
  if (cfg.counters) {
    // Counters after a digit (src/lattice.ts), added on top of the budget.
    //   single kanji: keep only the candidate read after kanji numerals in the dictionaries (三種 さん|しゅ -> 3種 3しゅ);
    //                 the other candidates of that kanji are flagged
    //   words: JMdict counters (pos "ctr") the reader still gets wrong, stored behind a \x01 marker; they may also
    //          follow the first counter (3種|盛り -> もり, 2|名様 -> めいさま)
    const ctr = counterReadings(train, spec.kanji, cfg.counters.minN, cfg.counters.minShare);
    const before = encodeModel(spec).length;
    spec = { ...spec, kanji: spec.kanji.map((k) => (ctr.has(k.cp) ? { ...k, cands: k.cands.map((c) => (ctr.get(k.cp) === c.reading ? c : { ...c, cls: c.cls | CLS_NOT_COUNTER })) } : k)) };
    const cur = readerOf(decodeModel(encodeModel(spec)));
    const cand = new Map<string, string>();
    const best = new Map<string, Example>(); // JMdict's own single-kanji counter readings are unreliable (袋=たい): words only
    for (const e of train) {
      if (!e.flags.split(",").includes("ctr") || !/^[\u4e00-\u9fff]/.test(e.surface) || e.cps.length < 2 || e.cps.length > 4) continue;
      const b = best.get(e.surface);
      if (!b || e.weight > b.weight) best.set(e.surface, e);
    }
    for (const [sf, e] of best) cand.set(sf, e.reading);
    const add: Array<[string, string]> = [];
    for (const [sf, r] of [...cand].sort()) if (cur("3" + sf) !== "3" + r) add.push(["\x01" + sf, r]);
    spec = { ...spec, lexicon: [...spec.lexicon, ...add] };
    log(`counters: ${ctr.size} flagged kanji, ${add.length} of ${cand.size} words (${encodeModel(spec).length - before} B): ${add.map(([k, r]) => `${k.slice(1)}=${r}`).join(" ")}`);
  }

  const bytes = encodeModel(spec);
  model = decodeModel(bytes);
  const empty = { emit: new Float32Array(0), trans: new Float32Array(0), bits: 0, transBits: 0 };
  const noScorer = sizeOf(spec, { scorer: { ...spec.scorer, ...empty } });
  const noLex = sizeOf(spec, { lexicon: [] });
  const tableOnly = sizeOf(spec, { lexicon: [], scorer: { ...spec.scorer, ...empty } });
  const breakdown = { total: bytes.length, table: tableOnly, lexicon: bytes.length - noLex, scorer: bytes.length - noScorer, devAcc: devAccuracy(model, dev) };
  return { name: cfg.name, bytes, model, breakdown };
};
