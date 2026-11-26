// Build the distributable: minified ESM reader + a chosen model.bin, and report sizes.
// Models are taken from research/models/<name>.bin (written by research/tools/exp.ts).
// usage: bun run train/bundle.ts <model-name> [budget-bytes] [extra-model-name:out-file ...]

import { brotliCompressSync, gzipSync } from "node:zlib";
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";

const name = process.argv[2] ?? "release";
const root = new URL("..", import.meta.url).pathname;
mkdirSync(root + "dist", { recursive: true });

const build = async (entry: string, out: string, target: "browser" | "node" = "browser") => {
  const r = await Bun.build({ entrypoints: [root + entry], minify: true, target, format: "esm" });
  if (!r.success) throw new Error(r.logs.join("\n"));
  writeFileSync(root + out, await r.outputs[0].text());
};
await build("src/index.ts", "dist/tiny-g2p.min.js");
await build("src/node.ts", "dist/node.js", "node"); // "node" export condition; not counted in the size budget
copyFileSync(root + `research/models/${name}.bin`, root + "dist/model.bin");
const budget = +(process.argv[3] ?? 100_000);
for (const extra of process.argv.slice(4)) { const [m, out] = extra.split(":"); copyFileSync(root + `research/models/${m}.bin`, root + `dist/${out}`); }

const sizes = (files: string[]) => {
  const rows = files.map((f) => {
    const b = readFileSync(root + f);
    return { file: f, raw: b.length, gzip: gzipSync(b, { level: 9 }).length, brotli: brotliCompressSync(b).length };
  });
  const tot = rows.reduce((a, r) => ({ file: "TOTAL", raw: a.raw + r.raw, gzip: a.gzip + r.gzip, brotli: a.brotli + r.brotli }), { file: "TOTAL", raw: 0, gzip: 0, brotli: 0 });
  return [...rows, tot];
};
const rows = sizes(["dist/tiny-g2p.min.js", "dist/model.bin"]);
console.table(rows);
const total = rows[rows.length - 1].raw;
console.log(total < budget ? `OK: total ${total} bytes < ${budget}` : `OVER BUDGET: total ${total} bytes >= ${budget}`);
const extras = process.argv.slice(4).map((e) => { const [m, out] = e.split(":"); return { model: m, files: sizes(["dist/tiny-g2p.min.js", `dist/${out}`]) }; });
for (const e of extras) console.log(e.model, "total", e.files[e.files.length - 1].raw);
writeFileSync(root + "dist/SIZES.json", JSON.stringify({ model: name, budget, files: rows, variants: extras }, null, 2));
