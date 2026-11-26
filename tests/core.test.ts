import { describe, expect, test } from "bun:test";
import { charType, CT, toHiragana } from "../src/kana";
import { applyVariant, compatible, explain, HANDAKU, RENDAKU, SOKUON, variantsOf } from "../src/phonology";
import { buildLattice, makeCtx } from "../src/lattice";
import { decodeModel } from "../src/codec";
import { emptyScorer } from "../src/model";
import { normalize, readerFromModel } from "../src/reader";
import { encodeModel } from "../train/encode";

describe("kana", () => {
  test("katakana -> hiragana keeps the long-vowel mark", () => {
    expect(toHiragana("ビール")).toBe("びーる");
    expect(toHiragana("ヴァ")).toBe("ゔぁ");
  });
  test("character classes", () => {
    expect(charType("大".codePointAt(0)!)).toBe(CT.KANJI);
    expect(charType("々".codePointAt(0)!)).toBe(CT.KANJI);
    expect(charType("ー".codePointAt(0)!)).toBe(CT.KATA);
    expect(charType("5".codePointAt(0)!)).toBe(CT.DIGIT);
  });
  test("NFKC normalisation", () => {
    expect(normalize("ＡＢＣ１２３ｶﾀｶﾅ")).toBe("ABC123カタカナ");
  });
});

describe("phonology", () => {
  test("variants are optional candidates", () => {
    expect(variantsOf("がく", false, true)).toEqual([0, SOKUON]);
    expect(variantsOf("かわ", true, false)).toEqual([0, RENDAKU]);
    expect(variantsOf("ほん", true, false)).toEqual([0, RENDAKU, HANDAKU]);
    expect(variantsOf("あめ", true, true)).toEqual([0]);
  });
  test("apply / explain are inverse", () => {
    expect(applyVariant("がく", SOKUON)).toBe("がっ");
    expect(applyVariant("かわ", RENDAKU)).toBe("がわ");
    expect(applyVariant("はつ", HANDAKU | SOKUON)).toBe("ぱっ");
    expect(explain("がっ", ["がく", "まなぶ"])).toEqual([[0, SOKUON]]);
  });
  test("sokuon needs a voiceless obstruent next; handaku needs ん/っ before", () => {
    expect(compatible("がっ", SOKUON, "こう", 0)).toBe(true);
    expect(compatible("がっ", SOKUON, "えん", 0)).toBe(false);
    expect(compatible("さん", 0, "ぽ", HANDAKU)).toBe(true);
    expect(compatible("さ", 0, "ぽ", HANDAKU)).toBe(false);
  });
});

const toyModel = () => decodeModel(encodeModel({
  kanji: [
    { cp: "大".codePointAt(0)!, cands: [{ reading: "だい", cls: 0, prior: 0 }, { reading: "おお", cls: 1, prior: 0 }] },
    { cp: "根".codePointAt(0)!, cands: [{ reading: "こん", cls: 0, prior: 0 }, { reading: "ね", cls: 1, prior: 0 }] },
    { cp: "学".codePointAt(0)!, cands: [{ reading: "がく", cls: 0, prior: 0 }] },
    { cp: "校".codePointAt(0)!, cands: [{ reading: "こう", cls: 0, prior: 0 }] },
  ],
  lexicon: [["海老", "えび"]],
  priorBits: 0,
  scorer: { ...emptyScorer(), quantBits: 4, varPen: Float32Array.from([0, -2, -2, -9, 1, -6, -6, -9]) },
}));

describe("lattice + decoder", () => {
  test("every position is covered; unknown characters are kept verbatim", () => {
    const m = toyModel();
    const cps = Array.from("大𠮷X根", (c) => c.codePointAt(0)!);
    const lat = buildLattice(m, makeCtx(cps));
    expect(lat.every((nodes) => nodes.length > 0)).toBe(true);
    expect(lat[1][0].out).toBe("𠮷");
    const read = readerFromModel(m);
    expect(read("大𠮷X根")).toBe("だい𠮷xこん".replace("x", "X"));
  });
  test("lexicon nodes and sound-change rules", () => {
    const read = readerFromModel(toyModel());
    expect(read("海老")).toBe("えび");
    expect(read("学校")).toBe("がっこう"); // sokuon is preferred by the toy variant weights
    expect(read("")).toBe("");
    expect(read("ビール500ml")).toBe("びーる500ml");
  });
});
