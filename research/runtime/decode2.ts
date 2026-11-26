// Experimental two-loop decoder (REPORT §6.12): symbolic lattice -> neural Viterbi (pass 1) ->
// symbolic chunking (src/chunk.ts) -> neural Viterbi on chunk-local contexts (pass 2).
// Kept out of the default bundle; use src/reader2.ts with a model that carries a chunker.

import { chunkBreaks } from "./chunk";
import { decodeSegment, viterbi } from "../../src/decode";
import type { Ctx, Node } from "../../src/lattice";
import { buildLattice, makeCtx } from "../../src/lattice";
import type { Model } from "../../src/model";
import { CH_GATE } from "../../src/model";
import { emitScore, transScore } from "../../src/score";

/** Total score of a path under a context (used to gate pass 2 against pass 1). */
const pathScore = (m: Model, ctx: Ctx, path: readonly Node[]): number => {
  const hidden = new Float32Array(Math.max(1, m.scorer.dim));
  let s = 0;
  path.forEach((nd, i) => { s += emitScore(m, nd, ctx, hidden); if (i) s += transScore(m, path[i - 1], nd, ctx); });
  return s;
};

/** Pass 2 decodes every chunk as an independent input (chunks behave exactly like standalone words). */
export const decodeSegment2 = (m: Model, cps: readonly number[]): string => {
  if (cps.length === 0) return "";
  const ctx = makeCtx(cps);
  const pass1 = viterbi(m, ctx, buildLattice(m, ctx));
  const brk = chunkBreaks(m, ctx, pass1);
  const out1 = pass1.map((nd) => nd.out).join("");
  if (!brk) return out1;
  const parts: number[][] = [[]];
  cps.forEach((c, i) => { if (brk[i]) parts.push([]); parts[parts.length - 1].push(c); });
  const out2 = parts.map((p) => decodeSegment(m, p)).join("");
  if (m.chunker && m.chunker.flags & CH_GATE) {
    let s2 = 0;
    for (const p of parts) { const c = makeCtx(p); s2 += pathScore(m, c, viterbi(m, c, buildLattice(m, c))); }
    if (s2 <= pathScore(m, ctx, pass1)) return out1;
  }
  return out2;
};
