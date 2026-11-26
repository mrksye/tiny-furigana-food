// Hashed feature extraction for candidate scoring. Shared verbatim by training tools.

import { CT, ITER_MARK } from "./kana";
import type { Ctx, Node } from "./lattice";
import { K_KANJI, K_LEX } from "./lattice";
import { F_ATOM, F_CAND_CHAR, F_CTX, F_OKURI, F_TRANS_CLS, F_TRANS_LEX } from "./model";

const mix = (h: number, x: number): number => {
  h = Math.imul(h ^ x, 0x9e3779b1);
  return (h ^ (h >>> 15)) >>> 0;
};
const h2 = (a: number, b: number) => mix(mix(0x811c9dc5, a), b);
const h3 = (a: number, b: number, c: number) => mix(h2(a, b), c);
const h4 = (a: number, b: number, c: number, d: number) => mix(h3(a, b, c), d);
const h5 = (a: number, b: number, c: number, d: number, e: number) => mix(h4(a, b, c, d), e);

const typeAt = (ctx: Ctx, i: number) => (i < 0 ? CT.BOS : i >= ctx.cps.length ? CT.EOS : ctx.types[i]);
const cpAt = (ctx: Ctx, i: number) => (i < 0 || i >= ctx.cps.length ? -1 : ctx.cps[i]);
const hiraAt = (ctx: Ctx, i: number) => (typeAt(ctx, i) === CT.HIRA ? ctx.cps[i] : 0);

/**
 * Emission features of a node. Writes raw 32-bit hashes into `buf`, returns count.
 * Literal nodes have no features (they are shared by every path).
 */
export const emitFeatures = (node: Node, ctx: Ctx, feats: number, buf: Uint32Array): number => {
  let k = 0;
  if (node.kind === K_LEX) {
    buf[k++] = h2(1, Math.min(node.e - node.s, 6));
    return k;
  }
  if (node.kind !== K_KANJI) return 0;
  const i = node.s;
  const c = node.cand, cls = node.cls, v = node.v;
  const pt = typeAt(ctx, i - 1), nt = typeAt(ctx, i + 1);
  const rl = Math.min(ctx.runLen[i], 4), rp = ctx.runPos[i];
  if (feats & F_CTX) {
    buf[k++] = h5(2, cls, v, pt, nt);
    buf[k++] = h4(3, cls, rl, rp);
    buf[k++] = h4(4, v, rl, rp);
    buf[k++] = h4(5, c, pt, nt);
    buf[k++] = h4(6, c, rl, rp);
    buf[k++] = h3(7, c, v);
    buf[k++] = h4(12, node.rank, cls, v);
    buf[k++] = h5(13, node.rank, pt, nt, rl);
    if (ctx.cps[i] === ITER_MARK) buf[k++] = h3(14, c, v);
  }
  if (feats & F_CAND_CHAR) {
    buf[k++] = h3(8, c, cpAt(ctx, i + 1));
    buf[k++] = h3(9, c, cpAt(ctx, i - 1));
  }
  if (feats & F_OKURI) {
    const n1 = hiraAt(ctx, i + 1);
    if (n1) {
      buf[k++] = h4(10, c, n1, hiraAt(ctx, i + 2));
      buf[k++] = h4(11, cls, n1, hiraAt(ctx, i + 2));
    }
  }
  if (feats & F_ATOM) {
    buf[k++] = h2(30, c);
    buf[k++] = h2(31, cpAt(ctx, i + 1));
    buf[k++] = h2(32, cpAt(ctx, i - 1));
    buf[k++] = h3(33, pt, nt);
    buf[k++] = h3(34, cls, v);
    buf[k++] = h3(35, rl, rp);
    buf[k++] = h2(36, hiraAt(ctx, i + 1));
    buf[k++] = h2(37, cpAt(ctx, i));
  }
  return k;
};

/** Transition features between adjacent kanji-ish nodes a -> b. */
export const transFeatures = (a: Node, b: Node, ctx: Ctx, feats: number, buf: Uint32Array): number => {
  if (a.kind !== K_KANJI || b.kind !== K_KANJI) return 0;
  let k = 0;
  if (feats & F_TRANS_CLS) {
    buf[k++] = h5(20, a.cls, b.cls, b.v, a.v);
    buf[k++] = h4(21, a.cls, b.cls, Math.min(ctx.runLen[b.s], 4));
    buf[k++] = h3(22, a.cand, b.cls);
    buf[k++] = h3(23, a.cls, b.cand);
    buf[k++] = h4(25, a.v, a.cls, b.out.charCodeAt(0));
    buf[k++] = h4(26, a.out.charCodeAt(a.out.length - 1), b.v, b.cls);
  }
  if (feats & F_TRANS_CLS && ctx.cps[b.s] === ITER_MARK) buf[k++] = h4(28, a.cand === b.cand ? 1 : 0, b.v, a.cls);
  if (feats & F_TRANS_LEX) {
    buf[k++] = h3(24, a.cand, b.cand);
    buf[k++] = h3(27, a.cand, b.out.charCodeAt(0));
  }
  return k;
};
