// Viterbi decoder over the reading lattice.

import type { Ctx, Node } from "./lattice";
import { buildLattice, K_KANJI, makeCtx } from "./lattice";
import type { Model } from "./model";
import { compatible } from "./phonology";
import { emitScore, transScore } from "./score";


export const canFollow = (a: Node, b: Node): boolean =>
  compatible(a.out, a.kind === K_KANJI ? a.v : 0, b.out, b.kind === K_KANJI ? b.v : 0);

/** Best path through the lattice; returns the chosen nodes in order. */
export const viterbi = (m: Model, ctx: Ctx, byStart: readonly Node[][]): Node[] => {
  const n = byStart.length;
  if (n === 0) return [];
  const hidden = new Float32Array(Math.max(1, m.scorer.dim));
  const nodes: Node[] = [];
  const score: number[] = [];
  const back: number[] = [];
  const ends: number[][] = Array.from({ length: n + 1 }, () => []);
  for (let s = 0; s < n; s++) {
    const preds = ends[s];
    if (s > 0 && preds.length === 0) continue;
    for (const node of byStart[s]) {
      let best = s === 0 ? 0 : -Infinity, bp = -1;
      for (const p of preds) {
        if (!canFollow(nodes[p], node)) continue;
        const sc = score[p] + transScore(m, nodes[p], node, ctx);
        if (sc > best) { best = sc; bp = p; }
      }
      if (best === -Infinity) continue;
      const id = nodes.length;
      nodes.push(node);
      score.push(best + emitScore(m, node, ctx, hidden));
      back.push(bp);
      ends[node.e].push(id);
    }
  }
  let bi = -1, bs = -Infinity;
  for (const id of ends[n]) if (score[id] > bs) { bs = score[id]; bi = id; }
  const path: Node[] = [];
  for (let i = bi; i >= 0; i = back[i]) path.push(nodes[i]);
  return path.reverse();
};

export const decodeSegment = (m: Model, cps: readonly number[]): string => {
  const ctx = makeCtx(cps);
  return viterbi(m, ctx, buildLattice(m, ctx)).map((nd) => nd.out).join("");
};
