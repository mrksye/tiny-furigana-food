// Symbolic candidate generator: input code points -> reading lattice.

import { charType, CT, ITER_MARK, kataToHiraCp } from "./kana";
import type { Model } from "./model";
import { CLS_OTHER } from "./model";
import { applyVariant, variantsOf } from "./phonology";

export const K_LIT = 0; //   literal / kana passthrough
export const K_KANJI = 1; // single kanji candidate
export const K_LEX = 2; //   multi-character lexicon entry

export type Node = {
  readonly s: number; //    start (inclusive)
  readonly e: number; //    end (exclusive)
  readonly out: string; //  surface reading
  readonly kind: number;
  readonly cand: number; // global candidate index (-1 if none)
  readonly v: number; //    variant mask
  readonly cls: number;
  readonly rank: number; // candidate rank within its kanji (0 = most frequent)
};

/** Per-position context shared by feature extraction. */
export type Ctx = {
  readonly cps: Int32Array;
  readonly types: Uint8Array;
  readonly runLen: Uint8Array; // length of the kanji run containing the position (0 if not kanji)
  readonly runPos: Uint8Array; // 0 first, 1 middle, 2 last, 3 single
};

export const makeCtx = (cps: readonly number[]): Ctx => {
  const n = cps.length;
  const types = new Uint8Array(n);
  const runLen = new Uint8Array(n);
  const runPos = new Uint8Array(n);
  for (let i = 0; i < n; i++) types[i] = charType(cps[i]);
  for (let i = 0; i < n; ) {
    if (types[i] !== CT.KANJI) { i++; continue; }
    let j = i;
    while (j < n && types[j] === CT.KANJI) j++;
    const len = j - i;
    for (let k = i; k < j; k++) {
      runLen[k] = Math.min(len, 255);
      runPos[k] = len === 1 ? 3 : k === i ? 0 : k === j - 1 ? 2 : 1;
    }
    i = j;
  }
  return { cps: Int32Array.from(cps), types, runLen, runPos };
};

const kanjiIndex = (m: Model, cps: Int32Array, i: number): number | undefined => {
  let k = i;
  while (k >= 0 && cps[k] === ITER_MARK) k--; // 々 repeats the previous kanji
  return k >= 0 ? m.kanji.get(cps[k]) : undefined;
};

export const literal = (cp: number): string => String.fromCodePoint(kataToHiraCp(cp));

/** Build lattice nodes grouped by start position. Every position gets >= 1 node. */
/** A node that is not a kanji candidate (literal character or lexicon entry). */
const plain = (s: number, e: number, out: string, kind: number): Node => ({ s, e, out, kind, cand: -1, v: 0, cls: CLS_OTHER, rank: 0 });

export const buildLattice = (m: Model, ctx: Ctx, useLexicon = true): Node[][] => {
  const { cps, types } = ctx;
  const n = cps.length;
  const byStart: Node[][] = Array.from({ length: n }, () => []);
  let surface = "";
  if (useLexicon && m.lexicon.size) for (let i = 0; i < n; i++) surface += String.fromCodePoint(cps[i]);
  const offs: number[] = []; // UTF-16 offsets for lexicon substring lookup
  if (surface) { let o = 0; for (let i = 0; i <= n; i++) { offs.push(o); if (i < n) o += cps[i] > 0xffff ? 2 : 1; } }

  // counter context: 2 = right after a digit, 1 = after the kanji that follows a digit or after a counter word
  // (3|種|盛り, 2|名様, 3点|盛り)
  const reach = new Uint8Array(n + 1);
  for (let i = 0; i < n; i++) {
    const out = byStart[i];
    if (types[i - 1] === CT.DIGIT) reach[i] = 2;
    if (reach[i] > 1 && types[i] === CT.KANJI) reach[i + 1] |= 1;
    const idx = types[i] === CT.KANJI ? kanjiIndex(m, cps, i) : undefined;
    if (idx === undefined && (cps[i] === 0x30f6 || cps[i] === 0x30b1) && types[i - 1] === CT.KANJI && types[i + 1] === CT.KANJI) {
      // ヶ / ケ between kanji (茅ヶ崎, 旭ケ丘, 一ヶ月) is a particle or counter, not the kana け
      for (const r of ["が", "け", "か"]) out.push(plain(i, i + 1, r, K_LIT));
    } else if (idx === undefined) {
      out.push(plain(i, i + 1, literal(cps[i]), K_LIT));
    } else {
      const prevT = i > 0 ? types[i - 1] : CT.BOS;
      const allowInit = cps[i] === ITER_MARK || prevT === CT.KANJI || prevT === CT.HIRA || prevT === CT.KATA;
      const allowFin = i + 1 < n && types[i + 1] === CT.KANJI;
      // a kanji right after a digit is a counter (3種, 2杯): drop the candidates flagged as non-counter (not 3切れ)
      const ctr = reach[i] > 1 && types[i + 1] !== CT.HIRA;
      for (let c = m.candStart[idx]; c < m.candStart[idx + 1]; c++) {
        if (ctr && m.candClass[c] & 4) continue;
        const base = m.readings[m.candReading[c]];
        for (const v of variantsOf(base, allowInit, allowFin)) {
          out.push({ s: i, e: i + 1, out: applyVariant(base, v), kind: K_KANJI, cand: c, v, cls: m.candClass[c] & 3, rank: c - m.candStart[idx] });
        }
      }
    }
    if (surface) {
      for (let len = 2; len <= m.lexMaxLen && i + len <= n; len++) {
        const w = surface.slice(offs[i], offs[i + len]);
        // counter words are stored behind a \x01 marker and only match in counter context (2|名様=めいさま,
        // 3種|盛り=もり). Pushed first so that they win ties against a plain entry (盛り=さかり).
        const r = m.lexicon.get(w), c = reach[i] ? m.lexicon.get("\x01" + w) : undefined;
        if (c !== undefined) reach[i + len] = 1;
        for (const o of [c, r]) if (o !== undefined) out.push(plain(i, i + len, o, K_LEX));
      }
    }
  }
  return byStart;
};
