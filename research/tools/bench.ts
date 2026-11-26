// Runtime benchmark: init (decode) time, per-input latency, heap growth.
// usage: bun run research/tools/bench.ts [dist/model.bin] [--neural]

import { heapStats } from "bun:jsc";
import { createReader } from "../../src/reader";
import { createNeuralReader } from "../runtime/neural-index";
import { loadEval } from "../../train/data";

const neural = process.argv.includes("--neural");
const path = process.argv.slice(2).find((a) => !a.startsWith("--")) ?? new URL("../../dist/model.bin", import.meta.url).pathname;
const make = neural ? createNeuralReader : createReader;
const bytes = new Uint8Array(await Bun.file(path).arrayBuffer());
const inputs = [...(await loadEval("words_unknown_common")), ...(await loadEval("places_unknown")), ...(await loadEval("place_products_synth"))].map((x) => x.input);

Bun.gc(true);
const heap0 = heapStats().heapSize;
const t0 = performance.now();
const read = await make({ data: bytes });
const tInit = performance.now() - t0;
Bun.gc(true);
const heap1 = heapStats().heapSize;

const inits: number[] = [];
for (let i = 0; i < 5; i++) { const s = performance.now(); await make({ data: bytes }); inits.push(performance.now() - s); }

for (const s of inputs.slice(0, 500)) read(s); // warm-up
const t1 = performance.now();
let chars = 0;
for (let rep = 0; rep < 3; rep++) for (const s of inputs) { read(s); chars += s.length; }
const tRun = performance.now() - t1;
const n = inputs.length * 3;
const det = inputs.every((s) => read(s) === read(s));

console.log(JSON.stringify({
  model: path, modelBytes: bytes.length,
  initMsFirst: +tInit.toFixed(1), initMsMedian: +inits.sort((a, b) => a - b)[2].toFixed(1),
  heapGrowthKB: Math.round((heap1 - heap0) / 1024),
  inputs: n, usPerInput: +((tRun * 1000) / n).toFixed(1), usPerChar: +((tRun * 1000) / chars).toFixed(2),
  deterministic: det, runtime: `bun ${Bun.version}`,
}, null, 2));
