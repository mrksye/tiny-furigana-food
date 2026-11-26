// Training / evaluation data loading (build-time only).

import { toHiragana } from "../src/kana";

const ROOT = new URL("..", import.meta.url).pathname;
export const WORK = ROOT + "data/work/";
export const EVAL = ROOT + "research/data/eval/";
export const DIST = ROOT + "dist/";

/** One aligned segment: chars [s, e) of the surface read as `r` (kana). */
export type Seg = { s: number; e: number; r: string };
export type Example = {
  surface: string; reading: string; weight: number; source: string; flags: string;
  cps: number[]; segs: Seg[] | null; // complete cover of the surface; null = not aligned yet (KEN_ALL)
};

const readTsv = async (path: string): Promise<Record<string, string>[]> => {
  const text = await Bun.file(path).text();
  const [head, ...lines] = text.split("\n").filter((l) => l.length);
  const cols = head.split("\t");
  return lines.map((l) => {
    const f = l.split("\t");
    return Object.fromEntries(cols.map((c, i) => [c, f[i] ?? ""]));
  });
};

const parseFurigana = (surface: string, furi: string): Seg[] | null => {
  const cps = Array.from(surface);
  const segs: Seg[] = [];
  const covered = new Array(cps.length).fill(false);
  for (const part of furi.split(";")) {
    const m = /^(\d+)(?:-(\d+))?:(.*)$/.exec(part);
    if (!m) return null;
    const s = +m[1], e = (m[2] !== undefined ? +m[2] : s) + 1;
    if (e > cps.length) return null;
    segs.push({ s, e, r: toHiragana(m[3]) });
    for (let i = s; i < e; i++) covered[i] = true;
  }
  cps.forEach((ch, i) => { if (!covered[i]) segs.push({ s: i, e: i + 1, r: toHiragana(ch) }); });
  return segs.sort((a, b) => a.s - b.s);
};

export const loadTrain = async (file = "train"): Promise<Example[]> => {
  const rows = await readTsv(WORK + file + ".tsv");
  const out: Example[] = [];
  for (const r of rows) {
    const norm = r.surface.normalize("NFKC");
    if (Array.from(norm).length !== Array.from(r.surface).length) continue;
    const segs = r.furigana ? parseFurigana(norm, r.furigana) : null;
    if (r.furigana && !segs) continue;
    if (segs && segs.map((s) => s.r).join("") !== toHiragana(r.reading)) continue;
    out.push({
      surface: norm, reading: toHiragana(r.reading), weight: +r.weight, source: r.source, flags: r.flags,
      cps: Array.from(norm, (c) => c.codePointAt(0)!), segs,
    });
  }
  return out;
};

/** Hiragana-only food words (JMdict food field, kana entries): そば, うどん ... */
export const loadHiraWords = async (): Promise<string[]> =>
  (await readTsv(WORK + "food_kana.tsv")).map((r) => r.surface).filter((s) => /^[\u3041-\u3096ー]{2,}$/.test(s));

export type EvalItem = { input: string; expected: string; alts: string[] };
export const loadEval = async (name: string): Promise<EvalItem[]> =>
  (await readTsv(EVAL + name + ".tsv")).map((r) => ({
    input: r.input, expected: toHiragana(r.expected),
    alts: r.alternatives ? r.alternatives.split("|").map(toHiragana) : [],
  }));

export type KanjiInfo = { grade: number; freq: number; on: string[]; kun: string[]; nanori: string[]; kunRaw: string[] };
export const loadKanjidic = async (): Promise<Map<string, KanjiInfo>> => {
  const rows = await readTsv(WORK + "kanji.tsv");
  const clean = (xs: string) => xs.split(",").filter(Boolean).map((x) => toHiragana(x.replace(/-/g, "").split(".")[0])).filter(Boolean);
  return new Map(rows.map((r) => [r.char, {
    grade: +r.grade || 0, freq: +r.freq || 0,
    on: clean(r.on), kun: clean(r.kun), nanori: clean(r.nanori),
    kunRaw: r.kun.split(",").filter(Boolean).map((x) => toHiragana(x.replace(/-/g, ""))),
  }]));
};

/** Deterministic PRNG (mulberry32). */
export const rng = (seed: number) => () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
