// Type declarations for tiny-g2p.min.js (public API of src/index.ts).

export type Read = (text: string) => string;

export type ReaderOptions = {
  /** Model bytes. If omitted, `url` (default: ./model.bin next to this module) is fetched once (read from disk under Node). */
  readonly data?: ArrayBuffer | Uint8Array;
  readonly url?: string | URL;
};

/** Decoded model. Its layout is internal; pass it to `readerFromModel`. */
export type Model = { readonly __tinyG2pModel: unique symbol };

/** Load the model once and return a synchronous reader: text -> hiragana reading. */
export function createReader(opts?: ReaderOptions): Promise<Read>;
export function readerFromModel(m: Model): Read;
export function decodeModel(bytes: Uint8Array): Model;
/** Unicode normalisation: NFKC (full-width ASCII -> ASCII, half-width kana -> full-width). */
export function normalize(s: string): string;
