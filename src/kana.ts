// Character classes and kana conversions. Pure functions over code points.

export const enum CT {
  BOS = 0,
  KANJI = 1,
  HIRA = 2,
  KATA = 3,
  DIGIT = 4,
  LATIN = 5,
  OTHER = 6,
  EOS = 7,
}

export const ITER_MARK = 0x3005; // 々

export const isKanji = (c: number): boolean =>
  (c >= 0x4e00 && c <= 0x9fff) ||
  (c >= 0x3400 && c <= 0x4dbf) ||
  (c >= 0xf900 && c <= 0xfaff) ||
  (c >= 0x20000 && c <= 0x3ffff) ||
  c === ITER_MARK ||
  c === 0x3006; // 〆

export const isHira = (c: number): boolean => (c >= 0x3041 && c <= 0x3096) || c === 0x309d || c === 0x309e;
export const isKata = (c: number): boolean => (c >= 0x30a1 && c <= 0x30fa) || c === 0x30fd || c === 0x30fe || c === 0x30fc;

export const charType = (c: number): CT =>
  isKanji(c) ? CT.KANJI
  : isHira(c) ? CT.HIRA
  : isKata(c) ? CT.KATA
  : c >= 0x30 && c <= 0x39 ? CT.DIGIT
  : (c >= 0x41 && c <= 0x5a) || (c >= 0x61 && c <= 0x7a) ? CT.LATIN
  : CT.OTHER;

/** Katakana code point -> hiragana code point (ヷ-ヺ and ー are kept). */
export const kataToHiraCp = (c: number): number =>
  (c >= 0x30a1 && c <= 0x30f6) || c === 0x30fd || c === 0x30fe ? c - 0x60 : c;

export const toHiragana = (s: string): string => {
  let out = "";
  for (const ch of s) out += String.fromCodePoint(kataToHiraCp(ch.codePointAt(0)!));
  return out;
};
