// Okurigana-omitted spellings for training and the lexicon (焼きそば -> 焼そば).

import { CT, charType } from "../src/kana";
import type { Example, Seg } from "./data";

/**
 * Okurigana-omitted variants with a light weight: 焼きそば -> 焼そば, 焼き鳥 -> 焼鳥, 盛り合わせ -> 盛合せ.
 * Kana are dropped only when they are the 連用形 okurigana of one of the kanji's KANJIDIC kun readings
 * (焼 や.く -> き, 盛 も.る -> り, 合 あ.わせる -> わ(せ)), so 茶しぶ / 茶めし are never touched.
 * For a multi-kana okurigana the last kana is kept (合わせ -> 合せ), following common practice.
 */
const GODAN_I: Record<string, string> = { う: "い", く: "き", ぐ: "ぎ", す: "し", つ: "ち", ぬ: "に", ぶ: "び", む: "み", る: "り" };
const renyo = (okuri: string): string[] => {
  const last = okuri.slice(-1), head = okuri.slice(0, -1);
  const out: string[] = [];
  if (GODAN_I[last]) out.push(head + GODAN_I[last]);
  if (last === "る" && head.length) out.push(head); // ichidan: 合わせる -> 合わせ
  return out;
};

export const okuriganaVariants = (ex: readonly Example[], weight: number, kun: ReadonlyMap<string, string[]>): Example[] => {
  const known = new Set(ex.map((e) => e.surface));
  const out: Example[] = [];
  for (const e of ex) {
    if (!e.segs || e.flags.includes("verb") || e.flags.includes("adj") || e.cps.length > 6) continue;
    const segs = e.segs;
    const drop = new Map<number, number>(); // seg index of okurigana start -> number of kana to drop
    for (let j = 0; j < segs.length - 1; j++) {
      const sg = segs[j];
      if (sg.e - sg.s !== 1 || charType(e.cps[sg.s]) !== CT.KANJI) continue;
      let h = "";
      for (let i = sg.e; i < e.cps.length && charType(e.cps[i]) === CT.HIRA; i++) h += String.fromCodePoint(e.cps[i]);
      if (!h) continue;
      for (const k of kun.get(String.fromCodePoint(e.cps[sg.s])) ?? []) {
        const [stem, okuri] = k.split(".");
        if (!okuri || stem.replace(/-/g, "") !== sg.r) continue;
        for (const r of renyo(okuri)) {
          if (!h.startsWith(r)) continue;
          const n = r.length === 1 ? 1 : r.length - 1;
          if (sg.e + n >= e.cps.length) continue; // something must follow (焼き alone stays)
          drop.set(j + 1, n);
        }
      }
    }
    if (!drop.size) continue;
    const dropPos = new Set<number>();
    for (const [j, n] of drop) for (let t = 0; t < n; t++) dropPos.add(segs[j].s + t);
    const cps: number[] = [], nsegs: Seg[] = [];
    let pending = "";
    for (let j = segs.length - 1; j >= 0; j--) {
      const sg = segs[j];
      if (sg.e - sg.s === 1 && dropPos.has(sg.s)) { pending = sg.r + pending; continue; }
      nsegs.unshift({ s: sg.s, e: sg.e, r: sg.r + pending });
      pending = "";
    }
    // re-index after removals
    let off = 0;
    const fixed: Seg[] = nsegs.map((sg) => {
      const len = sg.e - sg.s;
      const out2 = { s: off, e: off + len, r: sg.r };
      for (let i = sg.s; i < sg.e; i++) cps.push(e.cps[i]);
      off += len;
      return out2;
    });
    const surface = String.fromCodePoint(...cps);
    const kana = cps.filter((c) => charType(c) === CT.HIRA).length;
    if (known.has(surface) || cps.length < 3 || kana > 3) continue;
    known.add(surface);
    out.push({ ...e, surface, cps, segs: fixed, weight: e.weight * weight, flags: e.flags + ",okuri-omit" });
  }
  return out;
};
