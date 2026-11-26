// Symbolic sound-change rules, applied as *optional* lattice candidates.
//
// A candidate surface reading is a base reading plus a variant bitmask:
//   RENDAKU  (連濁)  first mora voiced          か->が, は->ば, ...
//   HANDAKU  (半濁)  first mora h -> p          は->ぱ (only after ん / っ)
//   SOKUON   (促音)  last mora つちくき -> っ    がく->がっ (only before k/s/t/p)
// Constraints between neighbours are checked in the decoder (see `compatible`).

export const RENDAKU = 1;
export const HANDAKU = 2;
export const SOKUON = 4;

const VOICED: Record<string, string> = {
  か: "が", き: "ぎ", く: "ぐ", け: "げ", こ: "ご",
  さ: "ざ", し: "じ", す: "ず", せ: "ぜ", そ: "ぞ",
  た: "だ", ち: "ぢ", つ: "づ", て: "で", と: "ど",
  は: "ば", ひ: "び", ふ: "ぶ", へ: "べ", ほ: "ぼ",
};
const SEMI: Record<string, string> = { は: "ぱ", ひ: "ぴ", ふ: "ぷ", へ: "ぺ", ほ: "ぽ" };
const GEMINABLE = new Set(["つ", "ち", "く", "き"]);
// Morae that may follow a sokuon: k/s/t/p rows (voiceless obstruents).
const AFTER_SOKUON = new Set([..."かきくけこさしすせそたちつてとぱぴぷぺぽはひふへほ"]);

/** Variant masks applicable to a base reading, given positional permissions. */
export const variantsOf = (base: string, allowInitial: boolean, allowFinal: boolean): number[] => {
  const first = base[0] ?? "";
  const last = base[base.length - 1] ?? "";
  const inits = [0];
  if (allowInitial && VOICED[first]) inits.push(RENDAKU);
  if (allowInitial && SEMI[first]) inits.push(HANDAKU);
  const fins = [0];
  if (allowFinal && base.length >= 2 && GEMINABLE.has(last)) fins.push(SOKUON);
  const out: number[] = [];
  for (const i of inits) for (const f of fins) out.push(i | f);
  return out;
};

export const applyVariant = (base: string, v: number): string => {
  if (v === 0) return base;
  let s = base;
  if (v & RENDAKU) s = VOICED[s[0]] + s.slice(1);
  else if (v & HANDAKU) s = SEMI[s[0]] + s.slice(1);
  if (v & SOKUON) s = s.slice(0, -1) + "っ";
  return s;
};

/** Hard phonotactic constraints between adjacent kanji readings (surface forms). */
export const compatible = (prevSurface: string, prevVariant: number, curSurface: string, curVariant: number): boolean => {
  if (prevVariant & SOKUON && !AFTER_SOKUON.has(curSurface[0])) return false;
  if (curVariant & HANDAKU) {
    const l = prevSurface[prevSurface.length - 1];
    if (l !== "ん" && l !== "っ") return false;
  }
  return true;
};

/** Inverse of applyVariant over a set of bases: which (base, variant) pairs yield `surface`. */
export const explain = (surface: string, bases: readonly string[]): Array<[number, number]> => {
  const out: Array<[number, number]> = [];
  bases.forEach((b, i) => {
    for (const v of variantsOf(b, true, true)) if (applyVariant(b, v) === surface) out.push([i, v]);
  });
  return out;
};
