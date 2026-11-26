// Regression tests against the released model (dist/model.bin). Skipped if it has not been built.
import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { createReader } from "../src/reader";

const path = new URL("../dist/model.bin", import.meta.url).pathname;
const has = existsSync(path);

describe.skipIf(!has)("release model", async () => {
  const read = has ? await createReader({ data: readFileSync(path) }) : (s: string) => s;
  test("fits the distribution budgets (default < 200 kB, model-100k variant < 100,000 B)", () => {
    const js = readFileSync(new URL("../dist/tiny-g2p.min.js", import.meta.url).pathname).length;
    expect(js + readFileSync(path).length).toBeLessThan(200_000);
    const small = new URL("../dist/model-100k.bin", import.meta.url).pathname;
    if (existsSync(small)) expect(js + readFileSync(small).length).toBeLessThan(100_000);
  });
  test("non-kanji characters are preserved / converted", () => {
    expect(read("ビール")).toBe("びーる");
    expect(read("牛乳500ml")).toBe("ぎゅうにゅう500ml");
    expect(read("ＡＢＣ")).toBe("ABC");
    expect(read("")).toBe("");
  });
  test("unknown characters are never dropped", () => {
    expect(read("𠮷")).toBe("𠮷");
    expect(read("🍺生")).toStartWith("🍺");
  });
  test("deterministic", () => {
    const xs = ["豚バラ肉", "冷凍枝豆", "宇治抹茶", "北海道産じゃがいも"];
    expect(xs.map(read)).toEqual(xs.map(read));
  });
  test("a few common items", () => {
    expect(read("大根")).toBe("だいこん");
    expect(read("冷凍枝豆")).toBe("れいとうえだまめ");
    expect(read("生ビール")).toBe("なまびーる");
  });
});
