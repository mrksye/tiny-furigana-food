// Lattice CRF trainer for the candidate scorer (linear or 1-hidden-layer MLP).
// Gold paths are latent: every path that spells the aligned reading counts as gold.

import { canFollow } from "../src/decode";
import { emitFeatures, transFeatures } from "../src/features";
import { buildLattice, K_KANJI, makeCtx, type Node } from "../src/lattice";
import type { Model, Scorer } from "../src/model";
import { rng, type Example } from "./data";
import { quantize } from "./encode";

export type CrfConfig = {
  feats: number; bits: number; dim: number; transBits: number;
  epochs: number; lr: number; seed: number; l2: number;
  qat?: { bits: number; epochs: number }; // extra epochs with quantised forward pass (straight-through), linear only
  noPrior?: boolean; // keep per-candidate priors at 0 (they are not stored; rank features replace them)
  log?: (s: string) => void;
};

export type Trained = { prior: Float32Array; scorer: Scorer };

const lse = (a: number, b: number) => (a === -Infinity ? b : b === -Infinity ? a : a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b)));

type Prepared = {
  w: number;
  nodes: Node[];
  gold: Uint8Array;
  rows: Uint32Array[]; // hashed emission feature rows per node
  edges: Int32Array; //  pairs (a, b)
  erows: Uint32Array[];
  ends: number[][];
  n: number;
};

const featBuf = new Uint32Array(64);

export const prepare = (m: Model, e: Example, cfg: CrfConfig): Prepared | null => {
  if (!e.segs || e.segs.some((s) => s.e - s.s !== 1)) return null;
  const ctx = makeCtx(e.cps);
  const lat = buildLattice(m, ctx, false);
  const n = e.cps.length;
  const nodes: Node[] = [];
  const gold: number[] = [];
  const goldAt = e.segs.map((s) => s.r);
  for (let s = 0; s < n; s++) {
    let any = false;
    for (const nd of lat[s]) {
      const g = nd.out === goldAt[s] ? 1 : 0;
      any ||= g === 1;
      nodes.push(nd); gold.push(g);
    }
    if (!any) return null;
  }
  const mask = (1 << cfg.bits) - 1, tmask = (1 << cfg.transBits) - 1;
  const rows = nodes.map((nd) => {
    if (!cfg.bits || nd.kind !== K_KANJI) return new Uint32Array(0);
    const k = emitFeatures(nd, ctx, cfg.feats, featBuf);
    return featBuf.slice(0, k).map((x) => x & mask);
  });
  const starts: number[][] = Array.from({ length: n }, () => []);
  const ends: number[][] = Array.from({ length: n + 1 }, () => []);
  nodes.forEach((nd, i) => { starts[nd.s].push(i); ends[nd.e].push(i); });
  const edges: number[] = [];
  const erows: Uint32Array[] = [];
  for (let s = 1; s < n; s++) for (const a of ends[s]) for (const b of starts[s]) {
    if (!canFollow(nodes[a], nodes[b])) continue;
    edges.push(a, b);
    const k = cfg.transBits ? transFeatures(nodes[a], nodes[b], ctx, cfg.feats, featBuf) : 0;
    erows.push(featBuf.slice(0, k).map((x) => x & tmask));
  }
  return { w: e.weight, nodes, gold: Uint8Array.from(gold), rows, edges: Int32Array.from(edges), erows, ends, n };
};

export const trainCrf = (m: Model, examples: readonly Example[], cfg: CrfConfig, dev?: (t: Trained) => number): Trained => {
  const log = cfg.log ?? (() => {});
  const nC = m.candPrior.length, d = cfg.dim;
  const rand = rng(cfg.seed);
  const prior = cfg.noPrior ? new Float32Array(nC) : Float32Array.from(m.candPrior), gPrior = new Float32Array(nC);
  const varPen = Float32Array.from(m.scorer.varPen), gVar = new Float32Array(8);
  const rowsN = cfg.bits ? 1 << cfg.bits : 0;
  const emit = new Float32Array(rowsN * d), gEmit = new Float32Array(rowsN * d);
  if (d > 1) for (let i = 0; i < emit.length; i++) emit[i] = (rand() - 0.5) * 0.02;
  const out = new Float32Array(d).fill(d > 1 ? 0.1 : 0), gOut = new Float32Array(d);
  const bias = new Float32Array(d).fill(d > 1 ? 0.01 : 0), gBias = new Float32Array(d);
  const trans = new Float32Array(cfg.transBits ? 1 << cfg.transBits : 0), gTrans = new Float32Array(trans.length);
  const eps = 1e-6;
  const upd = (w: Float32Array, G: Float32Array, i: number, g: number) => {
    g -= cfg.l2 * w[i];
    G[i] += g * g;
    w[i] += (cfg.lr * g) / Math.sqrt(G[i] + eps);
  };
  // QAT state: forward pass reads qe/qt (aliases of emit/trans until QAT starts)
  let qe = emit, qt = trans, se = 0, st = 0, lim = 0;
  const qsnap = (x: number, s: number) => Math.max(-lim, Math.min(lim, Math.round(x / s))) * s;
  const snapshot = (): Trained => ({
    prior: Float32Array.from(prior),
    scorer: {
      feats: cfg.feats, bits: cfg.bits, dim: d, emit: Float32Array.from(qe), out: Float32Array.from(out),
      bias: Float32Array.from(bias), transBits: cfg.transBits, trans: Float32Array.from(qt),
      lexScore: m.scorer.lexScore, varPen: Float32Array.from(varPen),
    },
  });

  const order = examples.map((_, i) => i);
  const hidden = new Float32Array(d);
  let best: Trained | null = null, bestDev = -1;
  const qatEpochs = cfg.qat && d === 1 ? cfg.qat.epochs : 0;
  for (let ep = 0; ep < cfg.epochs + qatEpochs; ep++) {
    if (qatEpochs && ep === cfg.epochs) {
      lim = (1 << (cfg.qat!.bits - 1)) - 1;
      se = quantize(emit, cfg.qat!.bits).scale; st = quantize(trans, cfg.qat!.bits).scale;
      qe = emit.map((x) => qsnap(x, se)); qt = trans.map((x) => qsnap(x, st));
      bestDev = -1; // only quantised snapshots compete from here on
      log(`QAT start: bits=${cfg.qat!.bits}`);
    }
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    let nll = 0, used = 0, skipped = 0;
    const t0 = performance.now();
    for (const oi of order) {
      const p = prepare(m, examples[oi], cfg);
      if (!p) { skipped++; continue; }
      used++;
      const N = p.nodes.length;
      // emission scores (keep MLP hidden activations)
      const es = new Float64Array(N);
      const hs: Float32Array[] = [];
      for (let v = 0; v < N; v++) {
        const nd = p.nodes[v];
        if (nd.kind !== K_KANJI) { hs.push(hidden); continue; }
        let s = prior[nd.cand] + varPen[nd.v];
        const r = p.rows[v];
        if (d === 1) for (const x of r) s += qe[x];
        else if (r.length) {
          const h = Float32Array.from(bias);
          for (const x of r) for (let j = 0; j < d; j++) h[j] += emit[x * d + j];
          for (let j = 0; j < d; j++) if (h[j] > 0) s += h[j] * out[j];
          hs.push(h); es[v] = s; continue;
        }
        hs.push(hidden);
        es[v] = s;
      }
      const E = p.edges.length / 2;
      const ts = new Float64Array(E);
      for (let k = 0; k < E; k++) { let s = 0; for (const x of p.erows[k]) s += qt[x]; ts[k] = s; }

      const marg = (goldOnly: boolean) => {
        const alpha = new Float64Array(N).fill(-Infinity), beta = new Float64Array(N).fill(-Infinity);
        const ok = (v: number) => !goldOnly || p.gold[v] === 1;
        for (let v = 0; v < N; v++) if (p.nodes[v].s === 0 && ok(v)) alpha[v] = es[v];
        // edges are generated in increasing boundary order -> one forward pass suffices
        for (let k = 0; k < E; k++) {
          const a = p.edges[2 * k], b = p.edges[2 * k + 1];
          if (!ok(a) || !ok(b) || alpha[a] === -Infinity) continue;
          alpha[b] = lse(alpha[b], alpha[a] + ts[k] + es[b]);
        }
        let Z = -Infinity;
        for (const v of p.ends[p.n]) if (ok(v)) { beta[v] = 0; Z = lse(Z, alpha[v]); }
        for (let k = E - 1; k >= 0; k--) {
          const a = p.edges[2 * k], b = p.edges[2 * k + 1];
          if (!ok(a) || !ok(b) || beta[b] === -Infinity) continue;
          beta[a] = lse(beta[a], ts[k] + es[b] + beta[b]);
        }
        const mu = new Float64Array(N), me = new Float64Array(E);
        if (Z === -Infinity) return { Z, mu, me };
        for (let v = 0; v < N; v++) if (alpha[v] > -Infinity && beta[v] > -Infinity) mu[v] = Math.exp(alpha[v] + beta[v] - Z);
        for (let k = 0; k < E; k++) {
          const a = p.edges[2 * k], b = p.edges[2 * k + 1];
          if (alpha[a] > -Infinity && beta[b] > -Infinity && ok(a) && ok(b)) me[k] = Math.exp(alpha[a] + ts[k] + es[b] + beta[b] - Z);
        }
        return { Z, mu, me };
      };
      const all = marg(false), gd = marg(true);
      if (gd.Z === -Infinity) { skipped++; continue; }
      nll += p.w * (all.Z - gd.Z);

      for (let v = 0; v < N; v++) {
        const nd = p.nodes[v];
        if (nd.kind !== K_KANJI) continue;
        const g = p.w * (gd.mu[v] - all.mu[v]);
        if (Math.abs(g) < 1e-7) continue;
        if (!cfg.noPrior) upd(prior, gPrior, nd.cand, g);
        upd(varPen, gVar, nd.v, g);
        const r = p.rows[v];
        if (d === 1) for (const x of r) { upd(emit, gEmit, x, g); if (qe !== emit) qe[x] = qsnap(emit[x], se); }
        else if (r.length) {
          const h = hs[v];
          for (let j = 0; j < d; j++) {
            if (h[j] <= 0) continue;
            const dh = g * out[j];
            upd(out, gOut, j, g * h[j]);
            upd(bias, gBias, j, dh);
            for (const x of r) upd(emit, gEmit, x * d + j, dh);
          }
        }
      }
      for (let k = 0; k < E; k++) {
        const g = p.w * (gd.me[k] - all.me[k]);
        if (Math.abs(g) < 1e-7) continue;
        for (const x of p.erows[k]) { upd(trans, gTrans, x, g); if (qt !== trans) qt[x] = qsnap(trans[x], st); }
      }
    }
    const snap = snapshot();
    const dv = dev ? dev(snap) : NaN;
    log(`epoch ${ep + 1}: nll/ex=${(nll / Math.max(1, used)).toFixed(4)} used=${used} skipped=${skipped} dev=${(dv * 100).toFixed(2)}% ${((performance.now() - t0) / 1000).toFixed(1)}s`);
    if (!dev || dv > bestDev) { bestDev = dv; best = snap; }
  }
  return best!;
};
