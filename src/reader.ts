// Public API.

import { decodeModel } from "./codec";
import { decodeSegment } from "./decode";
import type { Model } from "./model";

export type Read = (text: string) => string;

export type ReaderOptions = {
  /** Model bytes. If omitted, `url` (default: ./model.bin next to this module) is fetched once (read from disk under Node). */
  readonly data?: ArrayBuffer | Uint8Array;
  readonly url?: string | URL;
};

/** Unicode normalisation: NFKC (full-width ASCII -> ASCII, half-width kana -> full-width). */
export const normalize = (s: string): string => s.normalize("NFKC");

export const readerFromModel = (m: Model): Read => (text) => {
  const cps = Array.from(normalize(text), (ch) => ch.codePointAt(0)!);
  return decodeSegment(m, cps);
};

export const createReader = async (opts: ReaderOptions = {}): Promise<Read> => {
  const bytes = opts.data
    ? opts.data instanceof Uint8Array ? opts.data : new Uint8Array(opts.data)
    : new Uint8Array(await (await fetch(opts.url ?? new URL("./model.bin", import.meta.url))).arrayBuffer());
  return readerFromModel(decodeModel(bytes));
};
