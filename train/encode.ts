// Binary model encoder (mirror of src/codec.ts, format v2). Build-time only.

export type CandSpec = { reading: string; cls: number; prior: number };
export type ModelSpec = {
  kanji: Array<{ cp: number; cands: CandSpec[] }>;
  lexicon: Array<[string, string]>;
  priorBits: number;
  chunker?: { bits: number; bias: number; w: Float32Array; quantBits: number; flags: number };
  scorer: {
    feats: number; bits: number; dim: number; transBits: number; quantBits: number; lexScore: number;
    emit: Float32Array; out: Float32Array; bias: Float32Array; trans: Float32Array; varPen: Float32Array;
  };
};

export const pushVarint = (out: number[], x: number) => {
  do { let b = x % 128; x = Math.floor(x / 128); if (x) b |= 0x80; out.push(b); } while (x);
};
const pushF32 = (out: number[], v: number) => { out.push(...new Uint8Array(new Float32Array([v]).buffer)); };

export const kanaCode = (ch: string): number => {
  const c = ch.codePointAt(0)!;
  return c >= 0x3041 && c <= 0x309f ? c - 0x3040 : ch === "ー" ? 0x60 : -1;
};
export const isKanaString = (s: string) => Array.from(s).every((ch) => kanaCode(ch) > 0);

/** Huffman code lengths (two-queue construction) limited to 15 bits by flattening frequencies. */
const huffLengths = (freq: number[]): Uint8Array => {
  const n = freq.length;
  let f = freq.slice();
  for (;;) {
    const lens = new Uint8Array(n);
    const leaves = f.map((w, i) => [w, i] as const).filter(([w]) => w > 0).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (leaves.length === 0) return lens;
    if (leaves.length === 1) { lens[leaves[0][1]] = 1; return lens; }
    const L = leaves.length;
    const weight: number[] = leaves.map(([w]) => w);
    const parent: number[] = new Array(2 * L - 1).fill(-1);
    let li = 0, qi = L; // leaf cursor, internal-queue cursor
    const pick = () => {
      const takeLeaf = li < L && (qi >= weight.length || weight[li] <= weight[qi]);
      return takeLeaf ? li++ : qi++;
    };
    while (weight.length < 2 * L - 1) {
      const a = pick(), b = pick();
      const id = weight.length;
      weight.push(weight[a] + weight[b]);
      parent[a] = id; parent[b] = id;
    }
    const depth = new Array(2 * L - 1).fill(0);
    for (let i = 2 * L - 3; i >= 0; i--) depth[i] = depth[parent[i]] + 1;
    let mx = 0;
    for (let i = 0; i < L; i++) { lens[leaves[i][1]] = depth[i]; mx = Math.max(mx, depth[i]); }
    if (mx <= 15) return lens;
    f = f.map((x) => (x > 0 ? Math.max(1, x >> 1) : 0));
  }
};

/** Canonical Huffman stream: alphabet size, 4-bit lengths, count, bits (LSB-first bytes, codes MSB-first). */
export const pushHuff = (out: number[], syms: ArrayLike<number>) => {
  let nSym = 0;
  for (let i = 0; i < syms.length; i++) nSym = Math.max(nSym, syms[i] + 1);
  const freq = new Array(nSym).fill(0);
  for (let i = 0; i < syms.length; i++) freq[syms[i]]++;
  const lens = huffLengths(freq);
  pushVarint(out, nSym);
  for (let i = 0; i < nSym; i += 2) out.push(lens[i] | ((lens[i + 1] ?? 0) << 4));
  const codes = new Int32Array(nSym);
  let code = 0;
  for (let l = 1; l < 16; l++) {
    for (let s = 0; s < nSym; s++) if (lens[s] === l) codes[s] = code++;
    code <<= 1;
  }
  pushVarint(out, syms.length);
  let acc = 0, nb = 0;
  for (let i = 0; i < syms.length; i++) {
    const s = syms[i], l = lens[s], c = codes[s];
    for (let b = l - 1; b >= 0; b--) {
      acc |= ((c >> b) & 1) << nb;
      if (++nb === 8) { out.push(acc); acc = 0; nb = 0; }
    }
  }
  if (nb) out.push(acc);
};

/** Uniform quantiser with an MSE-optimal clipping scale. Returns levels in [0, 2^bits - 1]. */
export const quantize = (w: Float32Array, bits: number): { q: Int32Array; scale: number; zero: number } => {
  const zero = 1 << (bits - 1);
  const lim = zero - 1;
  let mx = 0;
  for (const x of w) mx = Math.max(mx, Math.abs(x));
  const nz = Array.from(w).filter((x) => x !== 0);
  const sample = nz.length > 20000 ? nz.filter((_, i) => i % Math.ceil(nz.length / 20000) === 0) : nz;
  let best = mx / lim || 1, bestErr = Infinity;
  for (let k = 1; k <= 20; k++) {
    const scale = (mx * k) / 20 / lim;
    if (!(scale > 0)) break;
    let err = 0;
    for (const x of sample) {
      const qq = Math.max(-lim, Math.min(lim, Math.round(x / scale)));
      err += (x - qq * scale) ** 2;
    }
    if (err < bestErr) { bestErr = err; best = scale; }
  }
  const q = new Int32Array(w.length);
  for (let i = 0; i < w.length; i++) q[i] = zero + Math.max(-lim, Math.min(lim, Math.round(w[i] / best)));
  return { q, scale: best, zero };
};

export const pushLevels = (out: number[], w: Float32Array, bits: number) => {
  const { q, scale, zero } = quantize(w, bits);
  pushF32(out, scale); pushF32(out, zero);
  const dense: number[] = [0];
  pushHuff(dense, q);
  const sparse: number[] = [1];
  pushVarint(sparse, q.length);
  const gaps: number[] = [], vals: number[] = [];
  let last = -1;
  for (let i = 0; i < q.length; i++) {
    if (q[i] === zero) continue;
    let g = i - last - 1;
    while (g >= 255) { gaps.push(255); g -= 255; }
    gaps.push(g); vals.push(q[i]); last = i;
  }
  pushHuff(sparse, gaps); pushHuff(sparse, vals);
  out.push(...(sparse.length < dense.length ? sparse : dense));
};

const kanaSyms = (s: string): number[] => Array.from(s, (ch) => {
  const k = kanaCode(ch);
  if (k < 0) throw new Error(`non-kana reading: ${s}`);
  return k;
});

export const encodeModel = (spec: ModelSpec): Uint8Array => {
  const out: number[] = [];
  out.push(..."TG2P".split("").map((c) => c.charCodeAt(0)), 3);

  const kanji = [...spec.kanji].sort((a, b) => a.cp - b.cp);
  const pool = [...new Set(kanji.flatMap((k) => k.cands.map((c) => c.reading)))].sort();
  const rid = new Map(pool.map((r, i) => [r, i]));
  const shared: number[] = [], pchars: number[] = [];
  pool.forEach((r, i) => {
    const p = i ? pool[i - 1] : "";
    let sh = 0;
    while (sh < r.length && sh < p.length && r[sh] === p[sh]) sh++;
    shared.push(sh);
    pchars.push(...kanaSyms(r.slice(sh)), 0);
  });
  pushHuff(out, shared);
  pushHuff(out, pchars);

  const deltas: number[] = [], esc: number[] = [];
  let prevCp = 0;
  for (const k of kanji) {
    const d = k.cp - prevCp;
    if (d >= 255) { deltas.push(255); esc.push(d); } else deltas.push(d);
    prevCp = k.cp;
  }
  pushHuff(out, deltas);
  for (const d of esc) pushVarint(out, d);

  const counts: number[] = [], rids: number[] = [], cls: number[] = [], pl: number[] = [];
  const pb = spec.priorBits;
  const priors = kanji.flatMap((k) => k.cands.map((c) => c.prior));
  const pMin = Math.min(...priors, 0), pMax = Math.max(...priors, 0);
  const pStep = pb > 0 && pMax > pMin ? (pMax - pMin) / ((1 << pb) - 1) : 0;
  for (const k of kanji) {
    if (k.cands.length < 1) throw new Error(`no candidates for ${String.fromCodePoint(k.cp)}`);
    counts.push(k.cands.length - 1);
    for (const c of k.cands) {
      rids.push(rid.get(c.reading)!);
      cls.push(c.cls);
      pl.push(pStep ? Math.round((c.prior - pMin) / pStep) : 0);
    }
  }
  pushHuff(out, counts); pushHuff(out, rids); pushHuff(out, cls);
  pushF32(out, pMin); pushF32(out, pStep);
  pushHuff(out, pb ? pl : []);

  const kidx = new Map(kanji.map((k, i) => [k.cp, i]));
  const lex = spec.lexicon.filter(([, r]) => isKanaString(r))
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  const lShared: number[] = [], lLen: number[] = [], lChars: number[] = [], lEsc: number[] = [], lRead: number[] = [];
  let prev: string[] = [];
  for (const [surf, reading] of lex) {
    const chars = Array.from(surf);
    let sh = 0;
    while (sh < chars.length && sh < prev.length && chars[sh] === prev[sh]) sh++;
    lShared.push(sh); lLen.push(chars.length - sh);
    for (const ch of chars.slice(sh)) {
      const cp = ch.codePointAt(0)!;
      const i = kidx.get(cp);
      if (i !== undefined) lChars.push(i + 1); else { lChars.push(0); lEsc.push(cp); }
    }
    lRead.push(...kanaSyms(reading), 0);
    prev = chars;
  }
  pushHuff(out, lShared); pushHuff(out, lLen); pushHuff(out, lChars);
  for (const cp of lEsc) pushVarint(out, cp);
  pushHuff(out, lRead);

  const s = spec.scorer;
  out.push(s.feats, s.bits, s.dim, s.transBits);
  pushF32(out, s.lexScore);
  for (let j = 0; j < 8; j++) pushF32(out, s.varPen[j] ?? 0);
  pushLevels(out, s.bits ? s.emit : new Float32Array(0), s.quantBits);
  for (let j = 0; j < s.dim; j++) pushF32(out, s.out[j] ?? 0);
  for (let j = 0; j < s.dim; j++) pushF32(out, s.bias[j] ?? 0);
  pushLevels(out, s.transBits ? s.trans : new Float32Array(0), s.quantBits);
  const ch = spec.chunker;
  out.push(ch ? ch.bits : 0);
  if (ch && ch.bits) {
    out.push(ch.flags);
    pushF32(out, ch.bias);
    pushLevels(out, ch.w, ch.quantBits);
  }
  return Uint8Array.from(out);
};
