// Build named configurations (train/configs.ts), evaluate them, write research/models/<name>.bin and append to
// research/results/results.jsonl. usage: bun run research/tools/exp.ts <config-name> [...]

import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { gzipSync } from "bun";
import { loadKanjidic, loadTrain } from "../../train/data";
import "../runtime/register"; // two-loop decoder for configs with `chunk`
import { evaluateAll, formatResults } from "./evaluate";
import { readerOf } from "../../train/read";
import { build } from "../../train/pipeline";
import { CONFIGS } from "../../train/configs";

const names = process.argv.slice(2);
const dic = await loadKanjidic();
const cache = new Map<string, Awaited<ReturnType<typeof loadTrain>>>();
const exFor = async (f: string) => { if (!cache.has(f)) cache.set(f, await loadTrain(f)); return cache.get(f)!; };
mkdirSync(new URL("../models/", import.meta.url).pathname, { recursive: true });
for (const name of names) {
  const cfg = CONFIGS[name];
  if (!cfg) throw new Error(`unknown config ${name}`);
  const t0 = performance.now();
  const r = await build(cfg, await exFor(cfg.trainFile ?? "train"), dic);
  const secs = (performance.now() - t0) / 1000;
  const results = await evaluateAll(readerOf(r.model), r.model);
  const gz = gzipSync(r.bytes as Uint8Array<ArrayBuffer>).length;
  console.log(`\n# ${name}  bytes=${r.bytes.length} gzip=${gz} ${JSON.stringify(r.breakdown)} (${secs.toFixed(0)}s)`);
  console.log(formatResults(results));
  writeFileSync(new URL(`../models/${name}.bin`, import.meta.url).pathname, r.bytes);
  appendFileSync(new URL("../results/results.jsonl", import.meta.url).pathname,
    JSON.stringify({ name, date: new Date().toISOString(), config: cfg, modelBytes: r.bytes.length, modelGzip: gz, breakdown: r.breakdown, results, buildSeconds: secs }) + "\n");
}
