// Dump errors of a built model on an eval set. usage: bun run research/tools/errors.ts <model.bin> <set> [n]
import { decodeModelFull as decodeModel } from "../runtime/reader2";
import { readerOf } from "../../train/read";
import { loadEval } from "../../train/data";
import { oracle, sameReading } from "./evaluate";

const [bin, set, n = "40"] = process.argv.slice(2);
const m = decodeModel(new Uint8Array(await Bun.file(bin).arrayBuffer()));
const read = readerOf(m);
let shown = 0;
for (const it of await loadEval(set)) {
  const raw = read(it.input);
  const got = /\s/.test(it.expected) ? raw : raw.replace(/\s+/g, ""); // same rules as evaluate.ts
  if (sameReading(got, it.expected) || it.alts.some((a) => sameReading(got, a))) continue;
  console.log(`${it.input}\t${it.expected}\t${got}\t${oracle(m, it.input, it.expected) ? "" : "NO-ORACLE"}`);
  if (++shown >= +n) break;
}
