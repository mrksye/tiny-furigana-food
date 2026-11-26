// System B: neural-only reader (BiGRU, slot-wise kana output). No reading dictionary at runtime.
//
// Binary "TGNN" v1: u8 E, u8 H, u8 K, H kanji code-point deltas (255 = escape, varint follows),
// then weight tensors (levels) in this order:
//   emb[V,E], fwd W_ih[3H,E], W_hh[3H,H], b_ih[3H], b_hh[3H], bwd (same 4), out W[K*A,2H], out b[K*A]
// Token ids: 0 PAD, 1 unknown kanji, 2 katakana, 3 digit, 4 latin, 5 other, 6.. hiragana U+3041..U+3096,
// then kanji in code-point order.

import { huff, levels, makeReader, u8, varint } from "../../src/codec";
import { charType, CT, isKanji, kataToHiraCp } from "../../src/kana";
import { normalize, type Read } from "../../src/reader";

const A = 0x3096 - 0x3041 + 3; // 86 hiragana + ー + end symbol
const NSPEC = 6 + (0x3096 - 0x3041 + 1);

type Net = {
  E: number; H: number; K: number; vocab: Map<number, number>;
  emb: Float32Array; f: Float32Array[]; b: Float32Array[]; outW: Float32Array; outB: Float32Array;
};

export const decodeNeural = (bytes: Uint8Array): Net => {
  const r = makeReader(bytes);
  if (String.fromCharCode(u8(r), u8(r), u8(r), u8(r)) !== "TGNN") throw new Error("tiny-furigana-food: bad neural model");
  u8(r);
  const E = u8(r), H = u8(r), K = u8(r);
  const vocab = new Map<number, number>();
  let cp = 0;
  huff(r).forEach((d, i) => { cp += d === 255 ? varint(r) : d; vocab.set(cp, NSPEC + i); });
  const emb = levels(r);
  const f = [levels(r), levels(r), levels(r), levels(r)];
  const b = [levels(r), levels(r), levels(r), levels(r)];
  return { E, H, K, vocab, emb, f, b, outW: levels(r), outB: levels(r) };
};

const token = (net: Net, cp: number): number => {
  if (isKanji(cp)) return net.vocab.get(cp) ?? 1;
  if (cp >= 0x3041 && cp <= 0x3096) return 6 + cp - 0x3041;
  const t = charType(cp);
  return t === CT.KATA ? 2 : t === CT.DIGIT ? 3 : t === CT.LATIN ? 4 : 5;
};

const sig = (x: number) => 1 / (1 + Math.exp(-x));

/** One GRU direction over the sequence (PyTorch gate order r, z, n). */
const gru = (xs: Float32Array[], [Wih, Whh, bih, bhh]: Float32Array[], E: number, H: number, reverse: boolean): Float32Array[] => {
  const out: Float32Array[] = new Array(xs.length);
  let h = new Float32Array(H);
  const gi = new Float32Array(3 * H), gh = new Float32Array(3 * H);
  for (let s = 0; s < xs.length; s++) {
    const t = reverse ? xs.length - 1 - s : s;
    const x = xs[t];
    for (let i = 0; i < 3 * H; i++) {
      let a = bih[i], c = bhh[i];
      for (let j = 0; j < E; j++) a += Wih[i * E + j] * x[j];
      for (let j = 0; j < H; j++) c += Whh[i * H + j] * h[j];
      gi[i] = a; gh[i] = c;
    }
    const nh = new Float32Array(H);
    for (let i = 0; i < H; i++) {
      const rr = sig(gi[i] + gh[i]);
      const z = sig(gi[H + i] + gh[H + i]);
      const n = Math.tanh(gi[2 * H + i] + rr * gh[2 * H + i]);
      nh[i] = (1 - z) * n + z * h[i];
    }
    h = nh;
    out[t] = h;
  }
  return out;
};

export const neuralReader = (net: Net): Read => (text) => {
  const cps = Array.from(normalize(text), (c) => c.codePointAt(0)!);
  if (cps.length === 0) return "";
  const { E, H, K } = net;
  const xs = cps.map((cp) => net.emb.subarray(token(net, cp) * E, token(net, cp) * E + E));
  const fw = gru(xs, net.f, E, H, false), bw = gru(xs, net.b, E, H, true);
  let out = "";
  cps.forEach((cp, t) => {
    if (!isKanji(cp)) { out += String.fromCodePoint(kataToHiraCp(cp)); return; }
    for (let k = 0; k < K; k++) {
      let best = 0, bs = -Infinity;
      for (let a = 0; a < A; a++) {
        const row = (k * A + a) * 2 * H;
        let s = net.outB[k * A + a];
        for (let j = 0; j < H; j++) s += net.outW[row + j] * fw[t][j] + net.outW[row + H + j] * bw[t][j];
        if (s > bs) { bs = s; best = a; }
      }
      if (best === 0) break;
      out += best === A - 1 ? "ー" : String.fromCharCode(0x3040 + best);
    }
  });
  return out;
};
