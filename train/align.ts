// Forced alignment of whole-word readings (KEN_ALL) against the symbolic lattice.
// Each kanji takes one of its candidate readings (cost 0) or an arbitrary 1-4 kana reading (cost 1);
// kana / literals must match. Alignments with total cost <= maxCost are accepted.

import { CT } from "../src/kana";
import { buildLattice, makeCtx } from "../src/lattice";
import type { Model } from "../src/model";
import type { Example, Seg } from "./data";

export const align = (m: Model, e: Example, maxCost = 1): Seg[] | null => {
  const ctx = makeCtx(e.cps);
  const lat = buildLattice(m, ctx, false);
  const n = e.cps.length, r = e.reading, R = r.length;
  const INF = 1e9;
  // cost[i][j]: best cost having consumed i chars and j reading chars
  const cost = Array.from({ length: n + 1 }, () => new Float64Array(R + 1).fill(INF));
  const back: Array<Array<[number, string] | null>> = Array.from({ length: n + 1 }, () => new Array(R + 1).fill(null));
  cost[0][0] = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j <= R; j++) {
    const c0 = cost[i][j];
    if (c0 >= INF) continue;
    const opts = new Map<string, number>();
    for (const nd of lat[i]) if (nd.e === i + 1 && r.startsWith(nd.out, j)) opts.set(nd.out, Math.min(opts.get(nd.out) ?? INF, 0));
    if (ctx.types[i] === CT.KANJI) for (let L = 1; L <= 4 && j + L <= R; L++) { const s = r.slice(j, j + L); if (!opts.has(s)) opts.set(s, 1); }
    for (const [s, c] of opts) {
      const nc = c0 + c, nj = j + s.length;
      if (nc < cost[i + 1][nj]) { cost[i + 1][nj] = nc; back[i + 1][nj] = [j, s]; }
    }
  }
  if (cost[n][R] > maxCost) return null;
  const segs: Seg[] = [];
  for (let i = n, j = R; i > 0; i--) { const [pj, s] = back[i][j]!; segs.push({ s: i - 1, e: i, r: s }); j = pj; }
  return segs.reverse();
};
