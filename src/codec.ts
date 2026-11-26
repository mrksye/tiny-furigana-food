// Binary model format v2 (decoder). The encoder lives in train/encode.ts.
//
// Every symbol sequence is stored as a canonical-Huffman stream ("H"):
//   varint alphabetSize, alphabetSize x 4-bit code lengths, varint count, bit stream (byte aligned).
// Layout:
//   "TG2P" u8 version(3)
//   readings : H shared-prefix lengths (one per reading), H kana codes (0 terminates a reading)
//   kanji    : H code-point deltas (255 = escape -> varint delta in the escape list), varint escapes...
//   cands    : H count-1 per kanji, H reading id, H class, f32 priorMin, f32 priorStep, H prior level
//   lexicon  : H shared-prefix chars, H suffix lengths, H chars (kanjiIdx+1, 0 = escape), varint escaped cps...,
//              H reading kana codes (0 terminates). Keys starting with U+0001 are counters (see lattice.ts)
//   scorer   : u8 feats, u8 bits, u8 dim, u8 transBits, f32 lexScore, f32[8] varPen,
//              f32 emitScale, f32 emitZero, levels (dense or sparse, see `levels`), f32[dim] out, f32[dim] bias,
//              f32 transScale, f32 transZero, H trans levels
//   chunker  : u8 bits; if bits > 0: u8 flags, f32 bias, levels  (read by src/reader2.ts only) (pass-2 chunk boundaries; bits = 0 = none)   [format v3]
// Kana codes: 1..0x5f = U+3041..U+309F, 0x60 = ー, 0x7f = escape (code point taken from the escape list).

import type { Model, Scorer } from "./model";

export type Reader = { readonly u8: Uint8Array; readonly dv: DataView; pos: number };
export const makeReader = (bytes: Uint8Array): Reader => ({ u8: bytes, dv: new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), pos: 0 });

export const varint = (r: Reader): number => {
  let x = 0, shift = 0, b: number;
  do { b = r.u8[r.pos++]; x += (b & 0x7f) * 2 ** shift; shift += 7; } while (b & 0x80);
  return x;
};
export const u8 = (r: Reader) => r.u8[r.pos++];
export const f32 = (r: Reader) => { const v = r.dv.getFloat32(r.pos, true); r.pos += 4; return v; };

export const huff = (r: Reader): Int32Array => {
  const nSym = varint(r);
  const lens = new Uint8Array(nSym);
  for (let i = 0; i < nSym; i += 2) { const b = u8(r); lens[i] = b & 15; if (i + 1 < nSym) lens[i + 1] = b >> 4; }
  const count = new Uint16Array(16);
  for (const l of lens) count[l]++;
  count[0] = 0;
  const offs = new Uint16Array(16);
  for (let l = 1; l < 15; l++) offs[l + 1] = offs[l] + count[l];
  const sorted = new Int32Array(nSym);
  for (let s = 0; s < nSym; s++) if (lens[s]) sorted[offs[lens[s]]++] = s;
  const n = varint(r);
  const out = new Int32Array(n);
  let acc = 0, nb = 0;
  const bit = () => { if (nb === 0) { acc = u8(r); nb = 8; } const b = acc & 1; acc >>= 1; nb--; return b; };
  for (let i = 0; i < n; i++) {
    let code = 0, first = 0, index = 0;
    for (let l = 1; l < 16; l++) {
      code |= bit();
      const c = count[l];
      if (code - first < c) { out[i] = sorted[index + code - first]; break; }
      index += c; first = (first + c) << 1; code <<= 1;
    }
  }
  return out;
};

const kanaFrom = (code: number, esc: () => number): string =>
  code === 0x7f ? String.fromCodePoint(esc()) : code === 0x60 ? "ー" : String.fromCharCode(0x3040 + code);

// Weight levels: u8 mode. 0 = dense (H levels); 1 = sparse: varint length, H gaps (255 = +255 and continue),
// H levels of the non-zero entries.
export const levels = (r: Reader): Float32Array => {
  const scale = f32(r), zero = f32(r);
  if (u8(r) === 0) {
    const q = huff(r);
    const w = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) w[i] = (q[i] - zero) * scale;
    return w;
  }
  const w = new Float32Array(varint(r));
  const gaps = huff(r), vals = huff(r);
  let pos = -1, gi = 0;
  for (let i = 0; i < vals.length; i++) {
    let g = 0;
    while (gaps[gi] === 255) { g += 255; gi++; }
    pos += g + gaps[gi++] + 1;
    w[pos] = (vals[i] - zero) * scale;
  }
  return w;
};

export const decodeModel = (bytes: Uint8Array): Model => {
  const r = makeReader(bytes);
  const noEsc = () => { throw new Error("tiny-furigana-food: bad model"); }; // wrong magic, unknown version or corrupt data
  if (String.fromCharCode(u8(r), u8(r), u8(r), u8(r)) !== "TG2P") noEsc();
  const ver = u8(r);
  if (ver !== 2 && ver !== 3) noEsc(); // v3 = v2 + optional chunker

  const shared = huff(r), pc = huff(r);
  const readings: string[] = [];
  for (let i = 0, j = 0; i < shared.length; i++) {
    let s = i ? readings[i - 1].slice(0, shared[i]) : "";
    for (; pc[j] !== 0; j++) s += kanaFrom(pc[j], noEsc);
    j++;
    readings.push(s);
  }

  const deltas = huff(r);
  const kanji = new Map<number, number>();
  const kanjiCps: number[] = [];
  let cp = 0;
  deltas.forEach((d, i) => { cp += d === 255 ? varint(r) : d; kanji.set(cp, i); kanjiCps.push(cp); });
  const nK = kanjiCps.length;

  const counts = huff(r), rid = huff(r), cls = huff(r);
  const pMin = f32(r), pStep = f32(r);
  const pl = huff(r);
  const candStart = new Uint32Array(nK + 1);
  for (let k = 0; k < nK; k++) candStart[k + 1] = candStart[k] + counts[k] + 1;
  const candPrior = new Float32Array(rid.length);
  for (let c = 0; c < rid.length; c++) candPrior[c] = pMin + (pl[c] ?? 0) * pStep;

  const lShared = huff(r), lLen = huff(r), lChars = huff(r);
  const escN = lChars.reduce((a, x) => a + (x === 0 ? 1 : 0), 0);
  const esc: number[] = [];
  for (let i = 0; i < escN; i++) esc.push(varint(r));
  const lRead = huff(r);
  const lexicon = new Map<string, string>();
  let lexMaxLen = 0, ci = 0, ei = 0, ri = 0;
  let prev: string[] = [];
  for (let i = 0; i < lShared.length; i++) {
    const chars = prev.slice(0, lShared[i]);
    for (let j = 0; j < lLen[i]; j++) {
      const x = lChars[ci++];
      chars.push(String.fromCodePoint(x === 0 ? esc[ei++] : kanjiCps[x - 1]));
    }
    let rd = "";
    for (; lRead[ri] !== 0; ri++) rd += kanaFrom(lRead[ri], noEsc);
    ri++;
    lexicon.set(chars.join(""), rd);
    lexMaxLen = Math.max(lexMaxLen, chars.length);
    prev = chars;
  }

  const feats = u8(r), bits = u8(r), dim = u8(r), transBits = u8(r);
  const lexScore = f32(r);
  const varPen = new Float32Array(8);
  for (let j = 0; j < 8; j++) varPen[j] = f32(r);
  const emit = levels(r);
  const out = new Float32Array(dim), bias = new Float32Array(dim);
  for (let j = 0; j < dim; j++) out[j] = f32(r);
  for (let j = 0; j < dim; j++) bias[j] = f32(r);
  const trans = levels(r);
  const scorer: Scorer = { feats, bits, dim, emit, out, bias, transBits, trans, lexScore, varPen };


  return {
    kanji, candStart,
    candReading: Uint16Array.from(rid), candClass: Uint8Array.from(cls), candPrior,
    readings, lexicon, lexMaxLen, scorer, end: r.pos,
  };
};
