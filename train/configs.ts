// Experiment configurations. Seeds are fixed for reproducibility.

import { F_ATOM, F_CAND_CHAR, F_CTX, F_OKURI, F_TRANS_CLS, F_TRANS_LEX } from "../src/model";
import type { BuildConfig } from "./pipeline";

const T = (maxKanji: number, maxCands: number, minShare: number, placeWeight = 1, dicPseudo = 0) =>
  ({ maxKanji, maxCands, minShare, placeWeight, alpha: 0.5, dicPseudo });
const ALL = F_CTX | F_CAND_CHAR | F_OKURI | F_TRANS_CLS | F_TRANS_LEX;
const crf = (bits: number, transBits: number, dim = 1, feats = ALL, epochs = 4, noPrior = false) =>
  ({ feats, bits, dim, transBits, epochs, lr: 0.1, seed: 1, l2: 0, noPrior });

const gen: Record<string, BuildConfig> = {};
// pruning sweep on a large hashed model
for (const q of [3, 4]) for (const keep of [5000, 10000, 20000, 40000, 80000]) {
  const name = `C-b18-q${q}-k${keep / 1000}k`;
  gen[name] = { name, table: T(3000, 8, 0.005), priorBits: 4, quantBits: q, crf: crf(18, 16, 1, ALL, 3), keep };
}

// D: compressed dictionary of frequent words + top-1 reading per kanji fallback
for (const [kb, maxKanji] of [[8, 1200], [16, 2136], [32, 2136], [64, 3000], [97, 3000]] as const) {
  const name = `D-${kb}k`;
  gen[name] = { name, table: T(maxKanji, 1, 0), priorBits: 0, quantBits: 4, lex: { budget: 0, maxLen: 8, minWeight: 0.25, onlyErrors: false }, target: kb === 97 ? 97000 : kb * 1024 };
}

// training-length / place-weight probes at a fixed 16 KB-ish scorer
gen["C-b15q4-e6"] = { name: "C-b15q4-e6", table: T(3000, 8, 0.005), priorBits: 4, quantBits: 4, crf: crf(15, 13, 1, ALL, 6) };
gen["C-b15q4-pw3"] = { name: "C-b15q4-pw3", table: T(3000, 8, 0.005, 3), priorBits: 4, quantBits: 4, crf: crf(15, 13, 1, ALL, 3) };

gen["C-b15q4-np"] = { name: "C-b15q4-np", table: T(3000, 8, 0.005, 3), priorBits: 0, quantBits: 4, crf: crf(15, 13, 1, ALL, 3, true) };
gen["C-b15q4-p4r"] = { name: "C-b15q4-p4r", table: T(3000, 8, 0.005, 3), priorBits: 4, quantBits: 4, crf: crf(15, 13, 1, ALL, 3) };

// ---- budget tiers (total distributed bytes = minified reader JS + model) ----
export const JS_READER = 8600; // upper bound of dist/tiny-g2p.min.js (verified by train/bundle.ts)
export const TIERS = { "16k": 16384, "32k": 32768, "64k": 65536, "100k": 99999 } as const;
const LEX = { budget: 0, maxLen: 8, minWeight: 0.25, onlyErrors: true };
type Cand = [string, ReturnType<typeof T>, number, number]; // label, table, emit bits, trans bits
const C_TIERS: Record<keyof typeof TIERS, Cand[]> = {
  "16k": [["a", T(1200, 3, 0.03), 10, 9], ["b", T(1200, 2, 0.05), 11, 10], ["c", T(1000, 2, 0.05), 11, 10], ["d", T(1400, 2, 0.05), 9, 8]],
  "32k": [["a", T(2136, 3, 0.03), 14, 12], ["b", T(2136, 4, 0.02), 13, 11], ["c", T(2500, 5, 0.01), 13, 11], ["d", T(1800, 3, 0.03), 14, 12]],
  "64k": [["a", T(3000, 8, 0.005, 3), 15, 13], ["b", T(3000, 6, 0.01, 3), 16, 13], ["c", T(2500, 5, 0.01, 3), 15, 13], ["d", T(3000, 6, 0.01, 3), 14, 12]],
  "100k": [["a", T(3000, 8, 0.005, 3), 16, 14], ["b", T(3500, 8, 0.005, 3), 16, 14], ["c", T(3000, 8, 0.005, 3), 17, 14], ["d", T(3500, 8, 0.005, 3), 15, 13]],
};
for (const [tier, total] of Object.entries(TIERS) as Array<[keyof typeof TIERS, number]>) {
  for (const [lab, table, b, tb] of C_TIERS[tier]) {
    const name = `C-${tier}-${lab}`;
    gen[name] = { name, table, priorBits: 0, quantBits: 4, crf: crf(b, tb, 1, ALL, 3, true), lex: LEX, target: total - JS_READER };
  }
  const nameD = `D-${tier}`;
  gen[nameD] = { name: nameD, table: T(tier === "16k" ? 1200 : 2136, 1, 0), priorBits: 0, quantBits: 4, lex: { ...LEX, onlyErrors: false }, target: total - JS_READER };
}
gen["C-100k-e"] = { name: "C-100k-e", table: T(3000, 8, 0.005, 3), priorBits: 0, quantBits: 4, crf: crf(17, 14, 1, ALL, 5, true), lex: LEX, target: TIERS["100k"] - JS_READER };
gen["C-100k-f"] = { name: "C-100k-f", table: T(3500, 8, 0.005, 3), priorBits: 0, quantBits: 4, crf: crf(17, 15, 1, ALL, 3, true), lex: LEX, target: TIERS["100k"] - JS_READER };
gen["C-100k-g"] = { name: "C-100k-g", table: T(3000, 8, 0.005, 3), priorBits: 0, quantBits: 4, crf: crf(17, 14, 1, ALL, 3, true), lex: LEX, target: TIERS["100k"] - JS_READER, foodBoost: 3 };
gen["C-100k-h"] = { name: "C-100k-h", table: T(3000, 8, 0.005, 3), priorBits: 0, quantBits: 4, crf: crf(17, 14, 1, ALL, 3, true), lex: LEX, target: TIERS["100k"] - JS_READER };
gen["C-100k-i"] = { ...gen["C-100k-h"], name: "C-100k-i", placeFame: true };
gen["C-100k-j"] = { ...gen["C-100k-h"], name: "C-100k-j", placeFameLex: true };
gen["C-64k-j"] = { ...gen["C-64k-a"], name: "C-64k-j", placeFameLex: true };
gen["C-32k-j"] = { ...gen["C-32k-a"], name: "C-32k-j", placeFameLex: true };
for (const seed of [2, 3]) {
  const base = gen["C-100k-h"];
  gen[`C-100k-h-s${seed}`] = { ...base, name: `C-100k-h-s${seed}`, crf: { ...base.crf!, seed } };
}
// ---- extended budgets (user relaxed the limit to < 200 KB) ----
const BIG = { "150k": 149999, "200k": 199999 } as const;
const big = (name: string, total: number, table: ReturnType<typeof T>, b: number, tb: number, q: number, fame: boolean, maxLen = 8) =>
  (gen[name] = { name, table, priorBits: 0, quantBits: q, crf: crf(b, tb, 1, ALL, 3, true), lex: { ...LEX, maxLen }, target: total - JS_READER, placeFameLex: fame });
big("C-150k-a", BIG["150k"], T(3000, 8, 0.005, 3), 17, 14, 4, false);
big("C-150k-b", BIG["150k"], T(3000, 8, 0.005, 3), 17, 14, 4, true);
big("C-150k-c", BIG["150k"], T(4000, 10, 0.003, 3), 18, 15, 3, true);
big("C-200k-a", BIG["200k"], T(4000, 10, 0.003, 3), 18, 15, 4, false);
big("C-200k-b", BIG["200k"], T(4000, 10, 0.003, 3), 18, 15, 4, true);
big("C-200k-c", BIG["200k"], T(3000, 8, 0.005, 3), 17, 14, 4, true);
big("C-200k-d", BIG["200k"], T(4000, 10, 0.003, 3), 18, 15, 3, true);

big("C-100k-k", TIERS["100k"], T(3000, 8, 0.005, 3), 16, 14, 4, true);
big("C-100k-l", TIERS["100k"], T(3000, 8, 0.005, 3), 15, 13, 4, true);
big("C-100k-m", TIERS["100k"], T(3000, 8, 0.005, 3), 17, 14, 3, true);
gen["C-100k-mq"] = { ...gen["C-100k-m"], name: "C-100k-mq", crf: { ...gen["C-100k-m"].crf!, qat: { bits: 3, epochs: 1 } } };
gen["C-100k-hq"] = { ...gen["C-100k-h"], name: "C-100k-hq", crf: { ...gen["C-100k-h"].crf!, qat: { bits: 4, epochs: 1 } } };
gen["C-100k-mf"] = { ...gen["C-100k-mq"], name: "C-100k-mf", foodLex: 5 };
gen["R-100k-mq"] = { ...gen["C-100k-mq"], name: "R-100k-mq", trainFile: "train_all" };
gen["R-100k-mf"] = { ...gen["C-100k-mf"], name: "R-100k-mf", trainFile: "train_all" };
gen["R-100k-hq"] = { ...gen["C-100k-hq"], name: "R-100k-hq", trainFile: "train_all" };
gen["R-100k-hf"] = { ...gen["C-100k-hq"], name: "R-100k-hf", trainFile: "train_all", foodLex: 5 };
// "135k" keeps the model budget of the first 135 kB release (built when JS_READER was 8,400): 126,599 + 8,600
for (const [tag, total] of [["115k", 114999], ["125k", 124999], ["135k", 135199], ["150k", 149999], ["175k", 174999]] as const) {
  big(`R-${tag}-mf`, total, T(3000, 8, 0.005, 3), 17, 14, 3, true);
  gen[`R-${tag}-mf`] = { ...gen[`R-${tag}-mf`], trainFile: "train_all", foodLex: 5, crf: { ...gen[`R-${tag}-mf`].crf!, qat: { bits: 3, epochs: 1 } } };
}
for (const tag of ["115k", "135k"]) gen[`C-${tag}-mf`] = { ...gen[`R-${tag}-mf`], name: `C-${tag}-mf`, trainFile: "train" };
// two-loop decoding (pass-2 chunking) and okurigana-omission augmentation
const CK = { bits: 13, samples: 300000, epochs: 2, seed: 1, lr: 0.1 };
for (const tag of ["100k", "135k"]) {
  for (const pre of ["C", "R"]) {
    const base = gen[`${pre}-${tag}-mf`];
    gen[`${pre}-${tag}-ck`] = { ...base, name: `${pre}-${tag}-ck`, chunk: CK, okuriAug: 0.2 };
  }
}
gen["C-135k-aug"] = { ...gen["C-135k-mf"], name: "C-135k-aug", okuriAug: 0.2 };
gen["C-135k-chunk"] = { ...gen["C-135k-mf"], name: "C-135k-chunk", chunk: CK };
// chunking variants (flags: 1 kana gaps, 2 score gate, 4 isolate phonology)
for (const [lab, flags] of [["iso", 4], ["soft", 0], ["gate", 2], ["kana", 1], ["kgate", 3]] as const)
  gen[`C-135k-ch-${lab}`] = { ...gen["C-135k-mf"], name: `C-135k-ch-${lab}`, chunk: { ...CK, flags } };
for (const mg of [1, 2, 3, 4]) gen[`C-135k-ch-iso-m${mg}`] = { ...gen["C-135k-mf"], name: `C-135k-ch-iso-m${mg}`, chunk: { ...CK, flags: 4, margin: mg } };
gen["C-135k-ch-lexonly"] = { ...gen["C-135k-mf"], name: "C-135k-ch-lexonly", chunk: { ...CK, flags: 4, margin: 1000 } };
gen["R-135k-aug"] = { ...gen["R-135k-mf"], name: "R-135k-aug", okuriAug: 0.2 };
for (const mg of [1, 2]) gen[`R-135k-aug-m${mg}`] = { ...gen["R-135k-aug"], name: `R-135k-aug-m${mg}`, chunk: { ...CK, flags: 4, margin: mg } };
gen["R-135k-mf-m1"] = { ...gen["R-135k-mf"], name: "R-135k-mf-m1", chunk: { ...CK, flags: 4, margin: 1 } };
for (const w of [0.2, 1]) gen[`R-135k-oklex${w}`] = { ...gen["R-135k-mf"], name: `R-135k-oklex${w}`, okuriLex: w };
for (const b of [1000, 2000]) gen[`R-135k-okx${b}`] = { ...gen["R-135k-mf"], name: `R-135k-okx${b}`, okuriExtra: b };
gen["R-100k-okx"] = { ...gen["R-100k-mf"], name: "R-100k-okx", okuriExtra: 1000 };
// strict 100 kB variants re-budgeted for the 8,733 B reader of this revision (JS bound 8,800)
gen["R-100k-v2"] = { ...gen["R-100k-mf"], name: "R-100k-v2", target: TIERS["100k"] - 8800 };
gen["R-100k-v2okx"] = { ...gen["R-100k-mf"], name: "R-100k-v2okx", target: TIERS["100k"] - 8800 - 1000, okuriExtra: 1000 };
gen["R-100k-v2okx500"] = { ...gen["R-100k-mf"], name: "R-100k-v2okx500", target: TIERS["100k"] - 8800 - 500, okuriExtra: 500 };
for (const [lab, budget, minFail] of [["a", 1500, 0.2], ["b", 1500, 0.1], ["c", 3000, 0.1]] as const)
  gen[`R-135k-cx${lab}`] = { ...gen["R-135k-okx1000"], name: `R-135k-cx${lab}`, compoundExtra: { budget, minFail, prefixes: 30 } };
for (const b of [1000, 2000]) gen[`R-135k-rp${b}`] = { ...gen["R-135k-cxb"], name: `R-135k-rp${b}`, repairExtra: b };
gen["R-100k-mf20"] = { ...gen["R-100k-mf"], name: "R-100k-mf20", foodLex: 20 };
gen["R-135k-dg"] = { ...gen["R-135k-rp1000"], name: "R-135k-dg", digitAug: 1 };
gen["R-135k-ct"] = { ...gen["R-135k-rp1000"], name: "R-135k-ct", counters: { minN: 3, minShare: 0.6 } }; // release
gen["R-125k-mf20"] = { ...gen["R-125k-mf"], name: "R-125k-mf20", foodLex: 20 };
big("C-115k-b", 114999, T(3000, 8, 0.005, 3), 17, 14, 4, true);
big("C-125k-b", 124999, T(3000, 8, 0.005, 3), 17, 14, 4, true);
for (const ls of [1, 2, 3, 0, -0.5, -1, -2]) gen[`C-150k-b-ls${ls}`] = { ...gen["C-150k-b"], name: `C-150k-b-ls${ls}`, lexScore: ls };

// ---- ablations under the final code: equal hashed parameter count (32,768) ----
const TA = T(3000, 8, 0.005, 3);
const abl = (name: string, bits: number, dim: number, feats: number, q: number) =>
  (gen[name] = { name, table: TA, priorBits: 0, quantBits: q, crf: crf(bits, 13, dim, feats, 3, true) });
abl("ABL-lin-q4", 15, 1, ALL, 4);
abl("ABL-lin-q3", 15, 1, ALL, 3);
abl("ABL-lin-q2", 15, 1, ALL, 2);
abl("ABL-lin-q8", 15, 1, ALL, 8);
abl("ABL-mlp4-q4", 13, 4, ALL, 4);
abl("ABL-mlp8-q4", 12, 8, ALL, 4);
abl("ABL-mlpatom16-q4", 11, 16, F_ATOM | F_TRANS_CLS | F_TRANS_LEX, 4);
abl("ABL-mlpatom16x-q4", 11, 16, ALL | F_ATOM, 4);
abl("ABL-lin-noctx", 15, 1, F_TRANS_CLS | F_TRANS_LEX, 4);
abl("ABL-lin-notrans", 15, 1, F_CTX | F_CAND_CHAR | F_OKURI, 4);
// A: symbolic only, largest useful table that fits (no word lexicon by definition)
gen["A-16k"] = { name: "A-16k", table: T(1600, 3, 0.03), priorBits: 0, quantBits: 4 };
gen["A-32k"] = { name: "A-32k", table: T(4000, 10, 0.003), priorBits: 0, quantBits: 4 };
gen["A-64k"] = { name: "A-64k", table: T(6000, 12, 0.002, 3), priorBits: 0, quantBits: 4 };

export const CONFIGS: Record<string, BuildConfig> = {
  ...gen,
  "A-30k": { name: "A-30k", table: T(3000, 8, 0.005), priorBits: 4, quantBits: 8 },
  "C-lin-test2": { name: "C-lin-test2", table: T(3000, 8, 0.005), priorBits: 4, quantBits: 8, crf: crf(14, 12, 1, ALL, 3) },
  "C-lin-test3": { name: "C-lin-test3", table: T(3000, 8, 0.005, 1, 0.5), priorBits: 4, quantBits: 8, crf: crf(14, 12, 1, ALL, 3) },
  "C-lin-lex10k": { name: "C-lin-lex10k", table: T(3000, 8, 0.005), priorBits: 4, quantBits: 8, crf: crf(14, 12, 1, ALL, 3), lex: { budget: 10000, maxLen: 6, minWeight: 0.25, onlyErrors: true } },
  "C-lin-lex40k": { name: "C-lin-lex40k", table: T(3000, 8, 0.005), priorBits: 4, quantBits: 8, crf: crf(14, 12, 1, ALL, 3), lex: { budget: 40000, maxLen: 6, minWeight: 0.25, onlyErrors: true } },
  "C-mlp8-b11": { name: "C-mlp8-b11", table: T(3000, 8, 0.005), priorBits: 4, quantBits: 8, crf: crf(11, 12, 8, ALL, 3) },
  "C-mlp4-b12": { name: "C-mlp4-b12", table: T(3000, 8, 0.005), priorBits: 4, quantBits: 8, crf: crf(12, 12, 4, ALL, 3) },
  "C-lin-b15q4": { name: "C-lin-b15q4", table: T(3000, 8, 0.005), priorBits: 4, quantBits: 4, crf: crf(15, 13, 1, ALL, 3) },
  "C-lin-b13": { name: "C-lin-b13", table: T(3000, 8, 0.005), priorBits: 4, quantBits: 8, crf: crf(13, 12, 1, ALL, 3) },
  "C-mlpatom16": { name: "C-mlpatom16", table: T(3000, 8, 0.005), priorBits: 4, quantBits: 4, crf: crf(11, 13, 16, F_ATOM | F_TRANS_CLS | F_TRANS_LEX, 4) },
  "C-mlpatom16x": { name: "C-mlpatom16x", table: T(3000, 8, 0.005), priorBits: 4, quantBits: 4, crf: crf(11, 13, 16, ALL | F_ATOM, 4) },
  "C-lin-b16q4": { name: "C-lin-b16q4", table: T(3000, 8, 0.005), priorBits: 4, quantBits: 4, crf: crf(16, 13, 1, ALL, 3) },
  "C-lin-b15q2": { name: "C-lin-b15q2", table: T(3000, 8, 0.005), priorBits: 4, quantBits: 2, crf: crf(16, 14, 1, ALL, 3) },
  "C-lin-test": { name: "C-lin-test", table: T(3000, 8, 0.005), priorBits: 4, quantBits: 8, crf: crf(14, 12, 1, ALL, 3) },
};
