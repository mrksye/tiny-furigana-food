// Pass-2 symbolic step: split long kanji runs into word-sized chunks (京都|抹茶, 千葉|落花生).
// Each chunk is then decoded as if it were a standalone word, which is the distribution the
// scorer was trained on. Boundary evidence: a tiny hashed linear model over character n-grams,
// plus the edges of exception-lexicon nodes chosen in pass 1.

import { CT } from "../../src/kana";
import type { Ctx, Node } from "../../src/lattice";
import { K_LEX } from "../../src/lattice";
import type { Model } from "../../src/model";
import { CH_KANA } from "../../src/model";

const mix = (h: number, x: number): number => {
  h = Math.imul(h ^ x, 0x9e3779b1);
  return (h ^ (h >>> 15)) >>> 0;
};
const hs = (...xs: number[]) => xs.reduce(mix, 0x811c9dc5);

export const MIN_CHUNK = 2; // never split off a single kanji (近江牛 stays whole: 牛 alone reads うし)

/** Hashed features of the gap before position i inside the kanji run [rs, re). */
export const gapFeatures = (ctx: Ctx, i: number, rs: number, re: number, out: Uint32Array): number => {
  const c = (j: number) => (j < rs || j >= re ? -1 : ctx.cps[j]);
  let k = 0;
  out[k++] = hs(40, c(i - 1), c(i));
  out[k++] = hs(41, c(i - 2), c(i - 1), c(i));
  out[k++] = hs(42, c(i - 1), c(i), c(i + 1));
  out[k++] = hs(43, c(i - 1));
  out[k++] = hs(44, c(i));
  out[k++] = hs(45, Math.min(i - rs, 4), Math.min(re - i, 4));
  return k;
};

/** Hashed features of a kanji|hiragana gap at i (kanji run [rs, i), hiragana run [i, he)): 焼|そば vs 焼き. */
export const kanaGapFeatures = (ctx: Ctx, i: number, rs: number, he: number, out: Uint32Array): number => {
  const h = (j: number) => (j < he ? ctx.cps[j] : -1);
  let k = 0;
  out[k++] = hs(50, ctx.cps[i - 1], h(i), h(i + 1));
  out[k++] = hs(51, ctx.cps[i - 1], h(i));
  out[k++] = hs(52, h(i), h(i + 1), h(i + 2));
  out[k++] = hs(53, Math.min(he - i, 5), Math.min(i - rs, 3));
  out[k++] = hs(54, h(he - 1), Math.min(he - i, 5));
  return k;
};

/** Kanji runs long enough to hold two chunks. */
export const runsOf = (ctx: Ctx): Array<[number, number]> => {
  const out: Array<[number, number]> = [];
  const n = ctx.cps.length;
  for (let i = 0; i < n; ) {
    if (ctx.types[i] !== CT.KANJI) { i++; continue; }
    let j = i;
    while (j < n && ctx.types[j] === CT.KANJI) j++;
    if (j - i >= 2 * MIN_CHUNK) out.push([i, j]);
    i = j;
  }
  return out;
};

const buf = new Uint32Array(8);

/** Boundary decisions for pass 2. Returns null when nothing is split. */
export const chunkBreaks = (m: Model, ctx: Ctx, pass1: readonly Node[]): Uint8Array | null => {
  const ch = m.chunker;
  if (!ch || ch.bits === 0) return null; // two-loop decoding is opt-in (the model carries a chunker)
  const runs = runsOf(ctx);
  const brk = new Uint8Array(ctx.cps.length);
  let any = false;
  for (const [rs, re] of runs) {
    // evidence 1: lexicon nodes from pass 1 (known words / place names)
    for (const nd of pass1) {
      if (nd.kind !== K_LEX || nd.s < rs || nd.e > re) continue;
      if (nd.s - rs >= MIN_CHUNK) { brk[nd.s] = 1; any = true; }
      if (re - nd.e >= MIN_CHUNK) { brk[nd.e] = 1; any = true; }
    }
    // evidence 2: learned boundary scores, greedily from the best gap
    if (ch.bits === 0) continue;
    const mask = (1 << ch.bits) - 1;
    const scores: Array<[number, number]> = [];
    for (let i = rs + MIN_CHUNK; i <= re - MIN_CHUNK; i++) {
      let s = ch.bias;
      const n = gapFeatures(ctx, i, rs, re, buf);
      for (let k = 0; k < n; k++) s += ch.w[buf[k] & mask];
      if (s > 0) scores.push([s, i]);
    }
    scores.sort((a, b) => b[0] - a[0] || a[1] - b[1]);
    for (const [, i] of scores) {
      let ok = true;
      for (let j = i - MIN_CHUNK + 1; j < i + MIN_CHUNK; j++) if (j !== i && j > rs && j < re && brk[j]) ok = false;
      if (ok) { brk[i] = 1; any = true; }
    }
  }
  // kanji run directly followed by a hiragana word (焼|そば, 焼|うどん); okurigana (焼き) should score low
  if (ch.bits && ch.flags & CH_KANA) {
    const mask = (1 << ch.bits) - 1;
    const n = ctx.cps.length;
    for (let i = 1; i < n; i++) {
      if (ctx.types[i - 1] !== CT.KANJI || ctx.types[i] !== CT.HIRA) continue;
      let he = i;
      while (he < n && ctx.types[he] === CT.HIRA) he++;
      if (he - i < 2) continue;
      let rs = i - 1;
      while (rs > 0 && ctx.types[rs - 1] === CT.KANJI) rs--;
      let s = ch.bias;
      const k = kanaGapFeatures(ctx, i, rs, he, buf);
      for (let j = 0; j < k; j++) s += ch.w[buf[j] & mask];
      if (s > 0) { brk[i] = 1; any = true; }
    }
  }
  return any ? brk : null;
};
