// Node / Bun / Deno entry (the "node" export condition). fetch() cannot read file: URLs in Node,
// so local models are read from disk here; this keeps the browser bundle free of the extra code.

import { readFile } from "node:fs/promises";
import { createReader as createBrowserReader, type Read, type ReaderOptions } from "./reader";

export { readerFromModel, normalize, type Read, type ReaderOptions } from "./reader";
export { decodeModel } from "./codec";

export const createReader = async (opts: ReaderOptions = {}): Promise<Read> => {
  const url = opts.url ?? new URL("./model.bin", import.meta.url);
  if (opts.data || /^(https?|data|blob):/.test(String(url))) return createBrowserReader(opts);
  return createBrowserReader({ data: await readFile(url) });
};
