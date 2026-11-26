// Keeps the hand-written dist/tiny-g2p.d.ts in sync with src/index.ts, src/node.ts and the built bundles.
// Type-level checks run under `bun run typecheck`; the export-name check runs under `bun test`.
import { expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import type * as Dts from "../dist/tiny-g2p";
import * as Src from "../src/index";
import * as SrcNode from "../src/node";

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
const assert = <T extends true>() => {};

assert<Equal<Dts.Read, Src.Read>>();
assert<Equal<Dts.ReaderOptions, Src.ReaderOptions>>();
assert<Equal<typeof Dts.createReader, typeof Src.createReader>>();
assert<Equal<typeof Dts.normalize, typeof Src.normalize>>();
assert<Equal<typeof Dts.createReader, typeof SrcNode.createReader>>();
// Model is opaque in the declarations, so compare the functions with Model swapped in.
assert<Equal<typeof Dts.readerFromModel, (m: Dts.Model) => Src.Read>>();
assert<Equal<typeof Dts.decodeModel, (bytes: Uint8Array) => Dts.Model>>();
assert<Equal<Parameters<typeof Src.readerFromModel>[0], ReturnType<typeof Src.decodeModel>>>();

const dts = readFileSync(new URL("../dist/tiny-g2p.d.ts", import.meta.url).pathname, "utf8");
const declared = [...dts.matchAll(/^export function (\w+)/gm)].map((m) => m[1]).sort();

test("d.ts declares exactly the functions src/index.ts exports", () => {
  expect(declared).toEqual(Object.keys(Src).sort());
});

test("the node entry exports the same functions", () => {
  expect(Object.keys(SrcNode).sort()).toEqual(declared);
});

for (const file of ["tiny-g2p.min.js", "node.js"]) {
  const bundle = new URL(`../dist/${file}`, import.meta.url).pathname;
  test.skipIf(!existsSync(bundle))(`d.ts declares exactly the functions dist/${file} exports`, async () => {
    expect(declared).toEqual(Object.keys(await import(bundle)).sort());
  });
}
