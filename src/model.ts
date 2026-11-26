// In-memory model representation (decoded from the binary blob in codec.ts).

export const CLS_ON = 0;
export const CLS_KUN = 1;
export const CLS_OTHER = 2;
export const CLS_NOT_COUNTER = 4; // flag bit: candidate dropped right after a digit (種=たね in 3種); masked out of features

/** Feature-group switches (bitmask) so that experiments can ablate groups. */
export const F_CTX = 1; //        class/variant x neighbour character types
export const F_CAND_CHAR = 2; //  candidate x neighbouring characters (hashed)
export const F_OKURI = 4; //      candidate x following hiragana (okurigana)
export const F_TRANS_CLS = 8; //  class/variant transitions
export const F_TRANS_LEX = 16; // candidate-pair transitions (hashed)
export const F_ATOM = 32; //      atomic (non-conjoined) features for the MLP to combine

export type Scorer = {
  readonly feats: number; //        feature-group mask
  readonly bits: number; //         log2 of hashed table rows (0 = no hashed model)
  readonly dim: number; //          1 = linear, >1 = MLP hidden width
  readonly emit: Float32Array; //   rows x dim (dequantised)
  readonly out: Float32Array; //    dim (MLP output weights)
  readonly bias: Float32Array; //   dim (MLP hidden bias)
  readonly transBits: number;
  readonly trans: Float32Array; //  1 << transBits (dequantised)
  readonly lexScore: number; //     score per character of a lexicon node
  readonly varPen: Float32Array; // 8, additive score per variant mask
};

export type Model = {
  readonly kanji: ReadonlyMap<number, number>; // code point -> kanji index
  readonly candStart: Uint32Array; //            kanji index -> first candidate (length = nKanji + 1)
  readonly candReading: Uint16Array; //          candidate -> reading id
  readonly candClass: Uint8Array; //             candidate -> CLS_*
  readonly candPrior: Float32Array; //           candidate -> prior score
  readonly readings: readonly string[]; //       reading pool (hiragana)
  readonly lexicon: ReadonlyMap<string, string>;
  readonly lexMaxLen: number;
  readonly scorer: Scorer;
  readonly chunker?: Chunker; // parsed only by src/reader2.ts (experimental two-loop decoder)
  readonly end?: number; //     byte offset after the core sections (where the chunker section starts)
};

/** Pass-2 chunk-boundary scorer (src/chunk.ts). bits = 0 disables it. */
export type Chunker = { readonly bits: number; readonly bias: number; readonly w: Float32Array; readonly flags: number };
export const emptyChunker = (): Chunker => ({ bits: 0, bias: 0, w: new Float32Array(0), flags: 0 });
export const CH_KANA = 1; //    also split kanji|hiragana-word gaps (焼|そば)
export const CH_GATE = 2; //    keep pass 2 only when its path scores higher than pass 1
export const CH_ISOLATE = 4; // also block sound changes / transitions across chunks

export const emptyScorer = (): Scorer => ({
  feats: 0,
  bits: 0,
  dim: 1,
  emit: new Float32Array(0),
  out: new Float32Array(0),
  bias: new Float32Array(0),
  transBits: 0,
  trans: new Float32Array(0),
  lexScore: 4,
  varPen: new Float32Array(8),
});
