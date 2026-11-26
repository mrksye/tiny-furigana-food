// Extension point for the experimental two-loop decoder (research/runtime).
// The build code never imports research code: research entry points call registerTwoLoop() first
// (research/runtime/register.ts), and configs with `chunk` fail loudly when it was not registered.

import type { Chunker, Model } from "../src/model";
import type { Read } from "../src/reader";
import type { Example } from "./data";

export type ChunkConfig = { bits: number; samples: number; epochs: number; seed: number; lr: number; flags?: number; margin?: number };

export type TwoLoop = {
  /** decode a model including the trailing chunker section */
  decode: (bytes: Uint8Array) => Model;
  /** reader for a model that carries a chunker */
  reader: (m: Model) => Read;
  /** train the pass-2 chunk-boundary scorer */
  trainChunker: (ex: readonly Example[], hiraWords: readonly string[], cfg: ChunkConfig, log?: (s: string) => void) => Chunker;
};

let ext: TwoLoop | undefined;
export const registerTwoLoop = (x: TwoLoop) => { ext = x; };
export const twoLoop = (): TwoLoop => {
  if (!ext) throw new Error("two-loop decoder not registered: import research/runtime/register.ts");
  return ext;
};
