import { expect, test } from "bun:test";
import { decodeModel } from "../src/codec";
import { emptyScorer } from "../src/model";
import { encodeModel } from "../train/encode";

const spec = (lexicon: Array<[string, string]> = []) => ({
  kanji: [
    { cp: "大".codePointAt(0)!, cands: [{ reading: "だい", cls: 0, prior: -0.2 }, { reading: "おお", cls: 1, prior: -1.5 }] },
    { cp: "根".codePointAt(0)!, cands: [{ reading: "こん", cls: 0, prior: -0.3 }, { reading: "ね", cls: 1, prior: -1 }] },
    { cp: "𠮷".codePointAt(0)!, cands: [{ reading: "よし", cls: 2, prior: 0 }] },
  ],
  lexicon,
  priorBits: 4,
  scorer: { ...emptyScorer(), quantBits: 4, bits: 4, transBits: 3, emit: Float32Array.from({ length: 16 }, (_, i) => (i - 8) / 10), trans: Float32Array.from({ length: 8 }, (_, i) => i / 7) },
});

test("round-trips table, lexicon and weights", () => {
  const m = decodeModel(encodeModel(spec([["海老", "えび"], ["大根役者", "だいこんやくしゃ"], ["𠮷野家", "よしのや"]])));
  expect(m.readings).toEqual(["おお", "こん", "だい", "ね", "よし"]);
  expect(m.kanji.get("根".codePointAt(0)!)).toBe(1);
  const i = m.kanji.get("大".codePointAt(0)!)!;
  expect(Array.from(m.candReading.slice(m.candStart[i], m.candStart[i + 1]), (r) => m.readings[r])).toEqual(["だい", "おお"]);
  expect(m.lexicon.get("海老")).toBe("えび");
  expect(m.lexicon.get("大根役者")).toBe("だいこんやくしゃ");
  expect(m.lexicon.get("𠮷野家")).toBe("よしのや");
  expect(m.lexMaxLen).toBe(4);
  expect(m.scorer.emit.length).toBe(16);
  expect(Math.abs(m.scorer.trans[7] - 1)).toBeLessThan(0.1);
  expect(m.candPrior[m.candStart[i]]).toBeCloseTo(-0.2, 0);
});
