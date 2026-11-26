// Evaluation metrics: exact match, any-valid-reading match, CER, oracle accuracy.

import { buildLattice, makeCtx } from "../../src/lattice";
import type { Model } from "../../src/model";
import { canFollow } from "../../src/decode";
import { normalize } from "../../src/reader";
import { loadEval, type EvalItem } from "../../train/data";

export const EVAL_SETS = [
  "words_known", "words_unknown_common", "words_unknown_rare", "compounds_unknown",
  "food_unknown", "places_unknown", "places_known_famous", "place_products_synth",
  "kenall_towns_unknown", "kenall_municipal_known", "kenall_products_synth", "spec_examples",
] as const;

/** Private evaluation sets (not in the public repository); used only when present. */
export const PRIVATE_SETS = ["menu_items", "menu_items_clean"] as const;

const editDistance = (a: string, b: string): number => {
  const A = Array.from(a), B = Array.from(b);
  let prev = Array.from({ length: B.length + 1 }, (_, j) => j);
  for (let i = 1; i <= A.length; i++) {
    const cur = [i];
    for (let j = 1; j <= B.length; j++)
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (A[i - 1] === B[j - 1] ? 0 : 1));
    prev = cur;
  }
  return prev[B.length];
};

/** Can some lattice path spell exactly `target`? */
export const oracle = (m: Model, input: string, target: string): boolean => {
  const cps = Array.from(normalize(input), (c) => c.codePointAt(0)!);
  const ctx = makeCtx(cps);
  const lat = buildLattice(m, ctx);
  // state: node index ending at position, with target offset; BFS
  type St = { node: (typeof lat)[0][0] | null; pos: number; off: number };
  const seen = new Set<string>();
  const stack: St[] = [{ node: null, pos: 0, off: 0 }];
  while (stack.length) {
    const st = stack.pop()!;
    if (st.pos === cps.length) { if (st.off === target.length) return true; continue; }
    for (const nd of lat[st.pos]) {
      if (st.node && !canFollow(st.node, nd)) continue;
      if (!target.startsWith(nd.out, st.off)) continue;
      const key = `${nd.e}:${st.off + nd.out.length}:${nd.v & 4}:${nd.out.slice(-1)}`;
      if (seen.has(key)) continue;
      seen.add(key);
      stack.push({ node: nd, pos: nd.e, off: st.off + nd.out.length });
    }
  }
  return false;
};

export type SetResult = { name: string; n: number; exact: number; anyValid: number; cer: number; oracle: number };

/**
 * Long-vowel notation is not an error: ジュース may be written じゅーす or じゅうす.
 * "ー" on one side matches the vowel kana it stands for on the other (after o-row also う, after e-row also い).
 */
const VOWEL_ROWS = ["あかさたなはまやらわがざだばぱぁゃゎ", "いきしちにひみりぎじぢびぴぃ", "うくすつぬふむゆるぐずづぶぷぅゅゔ", "えけせてねへめれげぜでべぺぇ", "おこそとのほもよろをごぞどぼぽぉょ"];
const LONG_OK = ["あ", "い", "う", "えい", "おう"];
const longOk = (prev: string, v: string) => {
  const row = VOWEL_ROWS.findIndex((r) => r.includes(prev));
  return row >= 0 && LONG_OK[row].includes(v);
};
export const sameReading = (a: string, b: string): boolean => {
  if (a === b) return true;
  const x = Array.from(a), y = Array.from(b);
  if (x.length !== y.length) return false;
  for (let i = 0; i < x.length; i++) {
    if (x[i] === y[i]) continue;
    if (i === 0 || x[i - 1] !== y[i - 1]) return false;
    if (!(x[i] === "ー" ? longOk(x[i - 1], y[i]) : y[i] === "ー" && longOk(y[i - 1], x[i]))) return false;
  }
  return true;
};

export const evaluateSet = (read: (s: string) => string, items: EvalItem[], name: string, m?: Model): SetResult => {
  let exact = 0, any = 0, ed = 0, len = 0, orc = 0;
  for (const it of items) {
    const exp = it.expected;
    const raw = read(it.input);
    const got = /\s/.test(exp) ? raw : raw.replace(/\s+/g, "");
    if (sameReading(got, exp)) exact++;
    if (sameReading(got, exp) || it.alts.some((a) => sameReading(got, a))) any++;
    ed += editDistance(got, exp);
    len += Array.from(exp).length;
    if (m && (oracle(m, it.input, exp) || it.alts.some((a) => oracle(m, it.input, a)))) orc++;
  }
  const n = items.length;
  return { name, n, exact: exact / n, anyValid: any / n, cer: ed / len, oracle: m ? orc / n : NaN };
};

export const evaluateAll = async (read: (s: string) => string, m?: Model, sets: readonly string[] = EVAL_SETS) => {
  const out: SetResult[] = [];
  for (const s of sets) out.push(evaluateSet(read, await loadEval(s), s, m));
  if (sets === EVAL_SETS) for (const s of PRIVATE_SETS) {
    if (await Bun.file(new URL(`../data/eval/${s}.tsv`, import.meta.url).pathname).exists()) out.push(evaluateSet(read, await loadEval(s), s, m));
  }
  return out;
};

export const formatResults = (rs: SetResult[]): string =>
  rs.map((r) => `${r.name.padEnd(22)} n=${String(r.n).padStart(4)} exact=${(r.exact * 100).toFixed(1).padStart(5)}% any=${(r.anyValid * 100).toFixed(1).padStart(5)}% cer=${(r.cer * 100).toFixed(1).padStart(5)}%${Number.isNaN(r.oracle) ? "" : ` oracle=${(r.oracle * 100).toFixed(1).padStart(5)}%`}`).join("\n");
