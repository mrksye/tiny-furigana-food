import { decodeNeural, neuralReader } from "./neural";
import type { Read } from "../../src/reader";

export const createNeuralReader = async (opts: { data?: ArrayBuffer | Uint8Array; url?: string | URL } = {}): Promise<Read> => {
  const bytes = opts.data
    ? opts.data instanceof Uint8Array ? opts.data : new Uint8Array(opts.data)
    : new Uint8Array(await (await fetch(opts.url ?? new URL("./model-nn.bin", import.meta.url))).arrayBuffer());
  return neuralReader(decodeNeural(bytes));
};
