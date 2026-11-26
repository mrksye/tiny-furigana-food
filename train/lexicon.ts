// Exception lexicon selection: store whole-word readings the scorer gets wrong,
// ranked by (training weight) / (encoded bytes), until a byte budget is used.

import { readerOf } from "./read";
import type { Model } from "../src/model";
import type { Example } from "./data";

export type LexConfig = {
  budget: number; //    max bytes added by the lexicon section
  maxLen: number; //    max surface length (chars)
  minWeight: number; // ignore entries below this aggregated weight
  onlyErrors: boolean; // D (dictionary-only) stores frequent words regardless of model errors
};

export const selectLexicon = (
  m: Model, ex: readonly Example[], cfg: LexConfig, measure: (lx: Array<[string, string]>) => number,
): Array<[string, string]> => {
  const read = readerOf(m);
  // aggregate: best reading per surface by weight
  const agg = new Map<string, { r: string; w: number; tot: number }>();
  for (const e of ex) {
    const n = Array.from(e.surface).length;
    if (n < 2 || n > cfg.maxLen) continue;
    const a = agg.get(e.surface);
    if (!a) agg.set(e.surface, { r: e.reading, w: e.weight, tot: e.weight });
    else { a.tot += e.weight; if (e.weight > a.w) { a.r = e.reading; a.w = e.weight; } }
  }
  // Okurigana-omitted spellings (唐揚 = からあげ, also written 唐揚げ) would duplicate the following kana
  // (唐揚げ -> からあげ + げ) when matched inside a longer input, so they are never stored.
  const reading = new Map<string, string>();
  for (const e of ex) if (!reading.has(e.surface)) reading.set(e.surface, e.reading);
  const omitted = new Set<string>();
  for (const [t, rt] of reading) {
    const m = /^(.*[^\u3041-\u309f])([\u3041-\u309f]{1,3})$/.exec(t);
    if (m && reading.get(m[1]) === rt) omitted.add(m[1]);
  }
  const cands: Array<{ s: string; r: string; gain: number }> = [];
  for (const [s, a] of agg) {
    if (a.tot < cfg.minWeight || omitted.has(s)) continue;
    if (cfg.onlyErrors && read(s) === a.r) continue;
    const cost = Array.from(s).length * 1.6 + Array.from(a.r).length + 1;
    cands.push({ s, r: a.r, gain: a.w / cost });
  }
  cands.sort((x, y) => y.gain - x.gain || (x.s < y.s ? -1 : 1));
  const base = measure([]);
  // binary search the number of entries that fits the byte budget
  let lo = 0, hi = cands.length;
  const take = (k: number) => cands.slice(0, k).map((c) => [c.s, c.r] as [string, string]);
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (measure(take(mid)) - base <= cfg.budget) lo = mid; else hi = mid - 1;
  }
  return take(lo);
};

/**
 * Compound-robust lexicon entries: words the reader gets right on their own but wrong inside a longer
 * product name (宇治 + 抹茶 -> うじまつちゃ, 生姜焼き + 生ビール -> ...いびーる). Each candidate is read after
 * every prefix; the expected reading is reader(prefix) + gold(word). Candidates are then guarded against
 * regressions: adding the entry must not break any training word that contains it (煎り豆腐, ひな菓子 ...).
 */
export const selectCompoundRobust = (
  m: Model, ex: readonly Example[], prefixes: readonly string[], cfg: { budget: number; minFail: number; maxLen: number },
  measure: (lx: Array<[string, string]>) => number, log: (s: string) => void = () => {},
): Array<[string, string]> => {
  const read = readerOf(m);
  const pre = prefixes.map((p) => [p, read(p)] as const);
  const reading = new Map<string, string>();
  for (const e of ex) if (!reading.has(e.surface)) reading.set(e.surface, e.reading);
  const best = new Map<string, { r: string; w: number }>();
  for (const e of ex) {
    const n = Array.from(e.surface).length;
    if (n < 2 || n > cfg.maxLen || !/^[\u4e00-\u9fff々][\u4e00-\u9fff々\u30a1-\u30fa\u30fc]+$/.test(e.surface) || m.lexicon.has(e.surface)) continue;
    const b = best.get(e.surface);
    if (!b || e.weight > b.w) best.set(e.surface, { r: e.reading, w: e.weight });
  }
  // okurigana-omitted spellings (太巻 = ふとまき, also 太巻き) would duplicate the following kana
  for (const [t, rt] of reading) {
    const mm = /^(.*[^\u3041-\u309f])([\u3041-\u309f]{1,3})$/.exec(t);
    if (mm && reading.get(mm[1]) === rt) best.delete(mm[1]);
  }
  const cands: Array<{ s: string; r: string; gain: number }> = [];
  for (const [s, { r, w }] of best) {
    if (read(s) !== r) continue; // standalone errors are the regular lexicon's job
    let fail = 0;
    for (const [p, pr] of pre) if (read(p + s) !== pr + r) fail++;
    const rate = fail / pre.length;
    if (rate < cfg.minFail) continue;
    cands.push({ s, r, gain: (w * rate) / (Array.from(s).length * 1.6 + Array.from(r).length + 1) });
  }
  // regression guard on the training vocabulary
  const candSet = new Set(cands.map((c) => c.s));
  const containing = new Map<string, number[]>();
  ex.forEach((e, i) => {
    const cs = Array.from(e.surface);
    const seen = new Set<string>();
    for (let a = 0; a < cs.length; a++) for (let b = a + 2; b <= Math.min(cs.length, a + cfg.maxLen); b++) {
      const sub = cs.slice(a, b).join("");
      if (sub !== e.surface && candSet.has(sub) && !seen.has(sub)) {
        seen.add(sub);
        const l = containing.get(sub) ?? [];
        if (l.length < 400) l.push(i);
        containing.set(sub, l);
      }
    }
  });
  const kept = cands.filter((c) => {
    const lex = new Map(m.lexicon);
    lex.set(c.s, c.r);
    const withEntry = readerOf({ ...m, lexicon: lex, lexMaxLen: Math.max(m.lexMaxLen, Array.from(c.s).length) });
    for (const i of containing.get(c.s) ?? []) {
      const e = ex[i];
      if (read(e.surface) === e.reading && withEntry(e.surface) !== e.reading) return false;
    }
    return true;
  });
  log(`compound-robust: ${cands.length} candidates, ${kept.length} pass the regression guard`);
  kept.sort((x, y) => y.gain - x.gain || (x.s < y.s ? -1 : 1));
  const base = measure([]);
  let lo = 0, hi = kept.length;
  const take = (k: number) => kept.slice(0, k).map((c) => [c.s, c.r] as [string, string]);
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (measure(take(mid)) - base <= cfg.budget) lo = mid; else hi = mid - 1;
  }
  return take(lo);
};

/**
 * Lexicon repair: training words that the reader got right *without* the exception lexicon but wrong
 * *with* it (a shorter entry such as 盛り=さかり hijacks 大盛り) are added as entries themselves, so that the
 * longer match wins. Additive only: existing entries are never removed.
 */
export const selectRepairs = (
  bare: Model, full: Model, ex: readonly Example[], cfg: { budget: number; maxLen: number },
  measure: (lx: Array<[string, string]>) => number, log: (s: string) => void = () => {},
): Array<[string, string]> => {
  const r0 = readerOf(bare), r1 = readerOf(full);
  const best = new Map<string, { r: string; w: number }>();
  for (const e of ex) {
    const n = Array.from(e.surface).length;
    if (n < 2 || n > cfg.maxLen || full.lexicon.has(e.surface)) continue;
    const b = best.get(e.surface);
    if (!b || e.weight > b.w) best.set(e.surface, { r: e.reading, w: e.weight });
  }
  const cands: Array<{ s: string; r: string; gain: number }> = [];
  for (const [s, { r, w }] of best) {
    if (r0(s) !== r || r1(s) === r) continue;
    cands.push({ s, r, gain: w / (Array.from(s).length * 1.6 + Array.from(r).length + 1) });
  }
  log(`lexicon repair: ${cands.length} words broken by the lexicon`);
  cands.sort((x, y) => y.gain - x.gain || (x.s < y.s ? -1 : 1));
  const base = measure([]);
  let lo = 0, hi = cands.length;
  const take = (k: number) => cands.slice(0, k).map((c) => [c.s, c.r] as [string, string]);
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (measure(take(mid)) - base <= cfg.budget) lo = mid; else hi = mid - 1;
  }
  return take(lo);
};
