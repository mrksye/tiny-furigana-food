// Re-evaluate saved models on given sets without retraining. usage: bun run research/tools/reeval.ts <set> <model-name>...
import { decodeModelFull as decodeModel } from "../runtime/reader2";
import { readerOf } from "../../train/read";
import { loadEval } from "../../train/data";
import { evaluateSet } from "./evaluate";
const [set, ...names] = process.argv.slice(2);
const items = await loadEval(set);
for (const n of names) {
  const m = decodeModel(new Uint8Array(await Bun.file(new URL(`../models/${n}.bin`, import.meta.url).pathname).arrayBuffer()));
  const r = evaluateSet(readerOf(m), items, set, m);
  console.log(`${n.padEnd(14)} ${set} n=${r.n} exact=${(r.exact * 100).toFixed(1)}% cer=${(r.cer * 100).toFixed(1)}% oracle=${(r.oracle * 100).toFixed(1)}%`);
}
