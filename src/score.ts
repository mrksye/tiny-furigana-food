// Tiny neural / linear candidate scoring.

import { emitFeatures, transFeatures } from "./features";
import type { Ctx, Node } from "./lattice";
import { K_KANJI, K_LEX } from "./lattice";
import type { Model } from "./model";

const buf = new Uint32Array(64);

/** Node score: candidate prior + hashed context model (linear or 1-hidden-layer MLP). */
export const emitScore = (m: Model, node: Node, ctx: Ctx, hidden: Float32Array): number => {
  if (node.kind === K_LEX) return m.scorer.lexScore * (node.e - node.s);
  if (node.kind !== K_KANJI) return 0;
  const sc = m.scorer;
  let s = m.candPrior[node.cand] + sc.varPen[node.v];
  if (sc.bits === 0) return s;
  const n = emitFeatures(node, ctx, sc.feats, buf);
  const mask = (1 << sc.bits) - 1;
  const d = sc.dim;
  if (d === 1) {
    for (let i = 0; i < n; i++) s += sc.emit[buf[i] & mask];
    return s;
  }
  hidden.set(sc.bias);
  for (let i = 0; i < n; i++) {
    const row = (buf[i] & mask) * d;
    for (let j = 0; j < d; j++) hidden[j] += sc.emit[row + j];
  }
  for (let j = 0; j < d; j++) if (hidden[j] > 0) s += hidden[j] * sc.out[j];
  return s;
};

export const transScore = (m: Model, a: Node, b: Node, ctx: Ctx): number => {
  const sc = m.scorer;
  if (sc.transBits === 0) return 0;
  const n = transFeatures(a, b, ctx, sc.feats, buf);
  const mask = (1 << sc.transBits) - 1;
  let s = 0;
  for (let i = 0; i < n; i++) s += sc.trans[buf[i] & mask];
  return s;
};
