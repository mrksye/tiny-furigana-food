// Regression check between two models: per eval set, how many items were fixed (wrong -> right) and
// broken (right -> wrong). Correct = exact match or a listed alternative reading; whitespace in the output
// is ignored when the gold has none (same rule as research/tools/evaluate.ts).
// usage: bun run research/tools/regress.ts <before-model> <after-model> [--show N] [--json out.json]

import { decodeModelFull as decodeModel } from "../runtime/reader2";
import { readerOf } from "../../train/read";
import { loadEval } from "../../train/data";
import { EVAL_SETS, PRIVATE_SETS, sameReading } from "./evaluate";

const args = process.argv.slice(2);
const [before, after] = args.filter((a) => !a.startsWith("--") && !/^\d+$/.test(a));
const showN = +(args[args.indexOf("--show") + 1] ?? 0) || 0;
const load = async (n: string) => readerOf(decodeModel(new Uint8Array(await Bun.file(new URL(`../models/${n}.bin`, import.meta.url).pathname).arrayBuffer())));
const [ra, rb] = await Promise.all([load(before), load(after)]);
const ok = (got: string, exp: string, alts: string[]) => {
  const g = /\s/.test(exp) ? got : got.replace(/\s+/g, "");
  return sameReading(g, exp) || alts.some((a) => sameReading(g, a));
};
const sets: string[] = [...EVAL_SETS];
for (const s of PRIVATE_SETS) if (await Bun.file(new URL(`../data/eval/${s}.tsv`, import.meta.url).pathname).exists()) sets.push(s);
console.log(`${before} -> ${after}`);
console.log(`| set | n | before | after | fixed | broken | net |`);
console.log(`|---|---|---|---|---|---|---|`);
const rows: Record<string, unknown>[] = [];
let tf = 0, tb = 0;
for (const set of sets) {
  const items = await loadEval(set);
  let a = 0, b = 0;
  const fixed: string[] = [], broken: string[] = [];
  for (const it of items) {
    const ga = ra(it.input), gb = rb(it.input);
    const oa = ok(ga, it.expected, it.alts), ob = ok(gb, it.expected, it.alts);
    if (oa) a++;
    if (ob) b++;
    if (!oa && ob) fixed.push(`${it.input}: ${ga} -> ${gb}`);
    if (oa && !ob) broken.push(`${it.input}: ${ga} -> ${gb} (gold ${it.expected})`);
  }
  tf += fixed.length; tb += broken.length;
  console.log(`| ${set} | ${items.length} | ${a} | ${b} | +${fixed.length} | -${broken.length} | ${fixed.length - broken.length >= 0 ? "+" : ""}${fixed.length - broken.length} |`);
  rows.push({ set, n: items.length, before: a, after: b, fixed: fixed.length, broken: broken.length, brokenItems: PRIVATE_SETS.includes(set as never) ? [] : broken });
  if (showN && broken.length) for (const x of broken.slice(0, showN)) console.log(`    BROKEN ${x}`);
}
console.log(`total fixed +${tf}, broken -${tb}`);
const ji = args.indexOf("--json");
if (ji >= 0) await Bun.write(args[ji + 1], JSON.stringify({ before, after, rows }, null, 2));
