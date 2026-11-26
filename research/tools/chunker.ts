// Train the pass-2 chunk-boundary scorer (src/chunk.ts) on synthetic concatenations of training words.
//   kanji|kanji gaps : place+food, word+word ... (positives at the joins, negatives inside words)
//   kanji|kana gaps  : kanji word + hiragana word (焼|そば) vs kanji + okurigana (焼き, 食べる)

import { CT, charType } from "../../src/kana";
import { makeCtx } from "../../src/lattice";
import { gapFeatures, kanaGapFeatures, MIN_CHUNK } from "../runtime/chunk";
import type { Chunker } from "../../src/model";
import { rng, type Example } from "../../train/data";

import type { ChunkConfig } from "../../train/hooks";

const isKanjiStr = (s: string) => Array.from(s).every((c) => charType(c.codePointAt(0)!) === CT.KANJI);
const isHiraStr = (s: string) => Array.from(s).every((c) => charType(c.codePointAt(0)!) === CT.HIRA);

type Sample = { feats: Uint32Array[]; labels: number[] };

export const trainChunker = (ex: readonly Example[], hiraWords: readonly string[], cfg: ChunkConfig, log = console.log): Chunker => {
  const r = rng(cfg.seed);
  const pool = (pred: (e: Example) => boolean) => {
    const xs = ex.filter(pred);
    const cum: number[] = [];
    let t = 0;
    for (const e of xs) { t += Math.sqrt(e.weight); cum.push(t); }
    return () => {
      const x = r() * t;
      let lo = 0, hi = cum.length - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (cum[mid] < x) lo = mid + 1; else hi = mid; }
      return xs[lo].surface;
    };
  };
  const kanjiLen = (lo: number, hi: number) => (e: Example) => isKanjiStr(e.surface) && e.surface.length >= lo && e.surface.length <= hi;
  const places = pool((e) => e.source === "place" && kanjiLen(2, 4)(e));
  const foods = pool((e) => e.flags.includes("food") && kanjiLen(2, 4)(e));
  const words = pool((e) => e.source === "word" && kanjiLen(2, 4)(e));
  const singles = pool((e) => e.source === "word" && kanjiLen(1, 2)(e));
  const okuri = pool((e) => /^[一-鿿]+[ぁ-ゖ]{1,3}$/.test(e.surface));
  const longWords = pool((e) => e.source === "word" && kanjiLen(4, 8)(e));
  const hira = hiraWords.filter((w) => w.length >= 2 && isHiraStr(w));

  const mask = (1 << cfg.bits) - 1;
  const buf = new Uint32Array(8);
  const samples: Sample[] = [];
  const add = (surface: string, joins: Set<number>) => {
    const cps = Array.from(surface, (c) => c.codePointAt(0)!);
    const ctx = makeCtx(cps);
    const s: Sample = { feats: [], labels: [] };
    const n = cps.length;
    for (let i = 1; i < n; i++) {
      const a = ctx.types[i - 1], b = ctx.types[i];
      if (a === CT.KANJI && b === CT.KANJI) {
        let rs = i - 1; while (rs > 0 && ctx.types[rs - 1] === CT.KANJI) rs--;
        let re = i; while (re < n && ctx.types[re] === CT.KANJI) re++;
        if (i - rs < MIN_CHUNK || re - i < MIN_CHUNK) continue;
        const k = gapFeatures(ctx, i, rs, re, buf);
        s.feats.push(buf.slice(0, k).map((x) => x & mask)); s.labels.push(joins.has(i) ? 1 : 0);
      } else if (a === CT.KANJI && b === CT.HIRA) {
        let he = i; while (he < n && ctx.types[he] === CT.HIRA) he++;
        if (he - i < 2) continue;
        let rs = i - 1; while (rs > 0 && ctx.types[rs - 1] === CT.KANJI) rs--;
        const k = kanaGapFeatures(ctx, i, rs, he, buf);
        s.feats.push(buf.slice(0, k).map((x) => x & mask)); s.labels.push(joins.has(i) ? 1 : 0);
      }
    }
    if (s.labels.length) samples.push(s);
  };
  for (let t = 0; t < cfg.samples; t++) {
    const u = r();
    const parts: string[] =
      u < 0.25 ? [places(), foods()]
      : u < 0.45 ? [words(), words()]
      : u < 0.55 ? [places(), words()]
      : u < 0.62 ? [foods(), foods()]
      : u < 0.70 ? [places(), words(), foods()]
      : u < 0.80 ? [singles(), hira[Math.floor(r() * hira.length)]] // 焼|そば
      : u < 0.92 ? [okuri()] // 焼き, 食べる: no boundary
      : [longWords()]; // real long words: no boundary
    const joins = new Set<number>();
    let pos = 0;
    for (let j = 0; j < parts.length; j++) { pos += parts[j].length; if (j < parts.length - 1) joins.add(pos); }
    add(parts.join(""), joins);
  }
  const pos = samples.reduce((a, s) => a + s.labels.filter((x) => x).length, 0);
  const tot = samples.reduce((a, s) => a + s.labels.length, 0);
  log(`chunker: ${samples.length} samples, ${tot} gaps, ${pos} boundaries`);

  const w = new Float32Array(1 << cfg.bits), G = new Float32Array(w.length);
  let bias = 0, Gb = 0;
  for (let ep = 0; ep < cfg.epochs; ep++) {
    let loss = 0, correct = 0;
    for (const s of samples) for (let g = 0; g < s.labels.length; g++) {
      const f = s.feats[g];
      let z = bias;
      for (const x of f) z += w[x];
      const p = 1 / (1 + Math.exp(-z));
      const y = s.labels[g];
      loss -= y ? Math.log(p + 1e-9) : Math.log(1 - p + 1e-9);
      if ((p > 0.5 ? 1 : 0) === y) correct++;
      const grad = y - p;
      Gb += grad * grad; bias += (cfg.lr * grad) / Math.sqrt(Gb + 1e-6);
      for (const x of f) { G[x] += grad * grad; w[x] += (cfg.lr * grad) / Math.sqrt(G[x] + 1e-6); }
    }
    log(`chunker epoch ${ep + 1}: loss=${(loss / tot).toFixed(4)} acc=${((correct / tot) * 100).toFixed(2)}%`);
  }
  // `margin` raises the decision threshold (split only when the boundary logit exceeds it)
  return { bits: cfg.bits, bias: bias - (cfg.margin ?? 0), w, flags: cfg.flags ?? 0 };
};
