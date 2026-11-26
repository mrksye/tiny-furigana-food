// Print readings for ad-hoc inputs. usage: bun run research/tools/try.ts <model.bin> 文字列...
import { createReader } from "../../src/reader";
const [bin, ...xs] = process.argv.slice(2);
const read = await createReader({ data: new Uint8Array(await Bun.file(bin).arrayBuffer()) });
for (const x of xs) console.log(`${x}\t${read(x)}`);
