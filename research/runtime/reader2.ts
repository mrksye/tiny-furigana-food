// Reader with the experimental two-loop decoder (src/decode2.ts). Not part of the default bundle.

import { decodeModel, f32, levels, makeReader, u8 } from "../../src/codec";
import { decodeSegment2 } from "./decode2";
import type { Model } from "../../src/model";
import { normalize, type Read, type ReaderOptions } from "../../src/reader";

/** decodeModel plus the optional chunker section at the end of a format-v3 model. */
export const decodeModelFull = (bytes: Uint8Array): Model => {
  const m = decodeModel(bytes);
  const r = makeReader(bytes);
  r.pos = m.end ?? bytes.length;
  const bits = r.pos < bytes.length ? u8(r) : 0;
  return bits ? { ...m, chunker: { bits, flags: u8(r), bias: f32(r), w: levels(r) } } : m;
};

export const readerFromModel2 = (m: Model): Read => (text) =>
  decodeSegment2(m, Array.from(normalize(text), (ch) => ch.codePointAt(0)!));

export const createReader2Loop = async (opts: ReaderOptions = {}): Promise<Read> => {
  const bytes = opts.data
    ? opts.data instanceof Uint8Array ? opts.data : new Uint8Array(opts.data)
    : new Uint8Array(await (await fetch(opts.url ?? new URL("./model.bin", import.meta.url))).arrayBuffer());
  return readerFromModel2(decodeModelFull(bytes));
};
