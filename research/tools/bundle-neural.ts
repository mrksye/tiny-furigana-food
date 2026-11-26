// Minified bundle of the neural-only reader (system B, comparison only) -> research/models/tiny-g2p-neural.min.js
// usage: bun run research/tools/bundle-neural.ts
import { writeFileSync } from "node:fs";

const root = new URL("../..", import.meta.url).pathname;
const r = await Bun.build({ entrypoints: [root + "research/runtime/neural-index.ts"], minify: true, target: "browser", format: "esm" });
if (!r.success) throw new Error(r.logs.join("\n"));
writeFileSync(root + "research/models/tiny-g2p-neural.min.js", await r.outputs[0].text());
