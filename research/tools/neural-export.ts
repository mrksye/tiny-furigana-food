// Quantise + encode a trained neural-only model (JSON from research/neural/neural_only.py) and evaluate it.
// usage: bun run research/tools/neural-export.ts <name> <quantBits>

import { appendFileSync, writeFileSync } from "node:fs";
import { gzipSync } from "bun";
import { decodeNeural, neuralReader } from "../runtime/neural";
import { pushHuff, pushLevels, pushVarint } from "../../train/encode";
import { evaluateAll, formatResults } from "./evaluate";

const [name, qb = "4"] = process.argv.slice(2);
const j = await Bun.file(new URL(`../../.cache/nn-${name}.json`, import.meta.url).pathname).json();
const st = j.state as Record<string, number[] | number[][]>;
const flat = (x: number[] | number[][]) => Float32Array.from((x as number[][]).flat ? (x as number[][]).flat() as number[] : x as number[]);
const kanji: string[] = j.kanji;
const nSpec = j.specials.length;
const order = kanji.map((c, i) => [c.codePointAt(0)!, i] as const).sort((a, b) => a[0] - b[0]);
const E = j.emb, H = j.hid;
const embAll = st["emb.weight"] as number[][];
const emb = Float32Array.from([...embAll.slice(0, nSpec), ...order.map(([, i]) => embAll[nSpec + i])].flat());

const out: number[] = [..."TGNN"].map((c) => c.charCodeAt(0));
out.push(1, E, H, j.K);
const deltas: number[] = [], esc: number[] = [];
let prev = 0;
for (const [cp] of order) { const d = cp - prev; if (d >= 255) { deltas.push(255); esc.push(d); } else deltas.push(d); prev = cp; }
pushHuff(out, deltas);
for (const d of esc) pushVarint(out, d);
const q = +qb;
pushLevels(out, emb, q);
for (const suf of ["", "_reverse"]) for (const k of ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"]) pushLevels(out, flat(st[`gru.${k}${suf}`]), q);
pushLevels(out, flat(st["out.weight"]), q);
pushLevels(out, flat(st["out.bias"]), 8);
const bytes = Uint8Array.from(out);
writeFileSync(new URL(`../models/B-${name}-q${q}.bin`, import.meta.url).pathname, bytes);
const t0 = performance.now();
const read = neuralReader(decodeNeural(bytes));
const results = await evaluateAll(read);
const gz = gzipSync(bytes).length;
console.log(`# B-${name}-q${q} bytes=${bytes.length} gzip=${gz} params=${j.params} (${((performance.now() - t0) / 1000).toFixed(0)}s)`);
console.log(formatResults(results));
appendFileSync(new URL("../results/results.jsonl", import.meta.url).pathname,
  JSON.stringify({ name: `B-${name}-q${q}`, date: new Date().toISOString(), config: { emb: E, hid: H, vocab: kanji.length, epochs: j.epochs, seed: j.seed, quantBits: q }, modelBytes: bytes.length, modelGzip: gz, breakdown: { total: bytes.length }, results }) + "\n");
