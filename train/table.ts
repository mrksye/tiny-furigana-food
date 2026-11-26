// Build the single-kanji reading candidate table from aligned training data + KANJIDIC2.

import { isKanji } from "../src/kana";
import { CLS_KUN, CLS_ON, CLS_OTHER } from "../src/model";
import { explain } from "../src/phonology";
import type { Example, KanjiInfo } from "./data";
import type { CandSpec } from "./encode";

export type TableConfig = {
  maxKanji: number; //    kanji kept (joyo first, then by observed weight)
  maxCands: number; //    candidates per kanji (<= 16)
  minShare: number; //    drop candidates whose share of the kanji's weight is below this
  placeWeight: number; // multiplier for place-name examples
  alpha: number; //       prior smoothing
  dicPseudo?: number; //  pseudo-weight for KANJIDIC readings never observed in training
};

export type KanjiStats = Map<string, Map<string, number>>; // kanji -> base reading -> weight

/** Attribute every observed single-kanji reading to (base, variant); unexplained readings become new bases. */
export const collectStats = (ex: readonly Example[], dic: Map<string, KanjiInfo>, placeWeight: number) => {
  const raw = new Map<string, Map<string, number>>();
  for (const e of ex) {
    const w = e.weight * (e.source === "place" ? placeWeight : 1);
    for (const sg of e.segs ?? []) {
      if (sg.e - sg.s !== 1) continue;
      if (!/^[\u3041-\u3096ー]+$/.test(sg.r)) continue;
      const ch = String.fromCodePoint(e.cps[sg.s]);
      if (!isKanji(e.cps[sg.s]) || ch === "々") continue;
      const m = raw.get(ch) ?? new Map<string, number>();
      m.set(sg.r, (m.get(sg.r) ?? 0) + w);
      raw.set(ch, m);
    }
  }
  const stats: KanjiStats = new Map();
  for (const [ch, obs] of raw) {
    const info = dic.get(ch);
    const known = new Set([...(info?.on ?? []), ...(info?.kun ?? [])]);
    const bases = new Map<string, number>();
    for (const b of known) bases.set(b, 0);
    for (const [r, w] of [...obs].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))) {
      if (bases.has(r)) { bases.set(r, bases.get(r)! + w); continue; }
      const keys = [...bases.keys()];
      const ex2 = explain(r, keys).filter(([i]) => keys[i] !== r);
      // prefer explanations through readings that are themselves attested
      ex2.sort((a, b) => (bases.get(keys[b[0]])! - bases.get(keys[a[0]])!) || a[1] - b[1]);
      if (ex2.length) { const k = keys[ex2[0][0]]; bases.set(k, bases.get(k)! + w); }
      else bases.set(r, w);
    }
    stats.set(ch, bases);
  }
  return stats;
};

const clsOf = (info: KanjiInfo | undefined, r: string) =>
  info?.on.includes(r) ? CLS_ON : info?.kun.includes(r) ? CLS_KUN : CLS_OTHER;

export const buildTable = (stats: KanjiStats, dic: Map<string, KanjiInfo>, cfg: TableConfig) => {
  const total = (ch: string) => [...(stats.get(ch)?.values() ?? [])].reduce((a, b) => a + b, 0);
  const joyo = [...dic].filter(([, i]) => i.grade >= 1 && i.grade <= 8).map(([c]) => c);
  const others = [...stats.keys()].filter((c) => !joyo.includes(c) && total(c) > 0)
    .sort((a, b) => total(b) - total(a) || (a < b ? -1 : 1));
  const joyoSorted = [...joyo].sort((a, b) => total(b) - total(a) || (a < b ? -1 : 1));
  const chosen = [...joyoSorted, ...others].slice(0, cfg.maxKanji);

  return chosen.map((ch) => {
    const info = dic.get(ch);
    const st = new Map(stats.get(ch) ?? []);
    const ps = cfg.dicPseudo ?? 0;
    if (ps > 0) for (const r of [...(info?.on ?? []), ...(info?.kun ?? [])]) st.set(r, (st.get(r) ?? 0) + ps);
    const tot = [...st.values()].reduce((a, b) => a + b, 0);
    let cands = [...st].filter(([, w]) => w > 0 && w >= cfg.minShare * tot)
      .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, cfg.maxCands);
    if (cands.length === 0) {
      const fb = [info?.on[0], info?.kun[0]].filter((x): x is string => !!x);
      cands = (fb.length ? fb : [""]).slice(0, cfg.maxCands).map((r) => [r, 0]);
    }
    const ctot = cands.reduce((a, [, w]) => a + w, 0);
    const k = cands.length;
    const out: CandSpec[] = cands.filter(([r]) => r !== "").map(([r, w]) => ({
      reading: r, cls: clsOf(info, r), prior: Math.log((w + cfg.alpha) / (ctot + cfg.alpha * k)),
    }));
    return { cp: ch.codePointAt(0)!, cands: out };
  }).filter((k) => k.cands.length > 0);
};

import { variantsOf } from "../src/phonology";
import { CT } from "../src/kana";
import { makeCtx } from "../src/lattice";
import type { Model } from "../src/model";

/** Count-based log-odds of each sound-change variant, relative to "no change" (system A weights). */
export const variantPenalties = (m: Model, ex: readonly Example[]): Float32Array => {
  const used = new Float64Array(8), possible = new Float64Array(8);
  for (const e of ex) {
    const ctx = makeCtx(e.cps);
    for (const sg of e.segs ?? []) {
      if (sg.e - sg.s !== 1 || ctx.types[sg.s] !== CT.KANJI) continue;
      const idx = m.kanji.get(e.cps[sg.s]);
      if (idx === undefined) continue;
      const i = sg.s, n = e.cps.length;
      const pt = i > 0 ? ctx.types[i - 1] : CT.BOS;
      const init = pt === CT.KANJI || pt === CT.HIRA || pt === CT.KATA, fin = i + 1 < n && ctx.types[i + 1] === CT.KANJI;
      for (let c = m.candStart[idx]; c < m.candStart[idx + 1]; c++) {
        const base = m.readings[m.candReading[c]];
        const vs = variantsOf(base, init, fin);
        const hit = vs.find((v) => applyVariantLocal(base, v) === sg.r);
        if (hit === undefined) continue;
        for (const v of vs) possible[v] += e.weight;
        used[hit] += e.weight;
        break;
      }
    }
  }
  const lp = (v: number) => Math.log((used[v] + 1) / (possible[v] + 2));
  return Float32Array.from({ length: 8 }, (_, v) => (possible[v] ? lp(v) - lp(0) : -10));
};

import { applyVariant as applyVariantLocal } from "../src/phonology";
