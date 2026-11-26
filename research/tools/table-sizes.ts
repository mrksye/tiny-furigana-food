// Quick grid of symbolic table sizes and system-A dev accuracy (no training).
import { decodeModelFull as decodeModel } from "../runtime/reader2";
import { emptyScorer } from "../../src/model";
import { loadKanjidic, loadTrain } from "../../train/data";
import { encodeModel } from "../../train/encode";
import { devAccuracy, splitDev } from "../../train/pipeline";
import { buildTable, collectStats, variantPenalties } from "../../train/table";
const [ex, dic] = await Promise.all([loadTrain(), loadKanjidic()]);
const { train, dev } = splitDev(ex, 2000, 7);
for (const pw of [1, 3]) {
  const stats = collectStats(train, dic, pw);
  for (const [k, c, s] of [[800, 2, 0.05], [1200, 2, 0.05], [1200, 3, 0.03], [1600, 3, 0.03], [2136, 3, 0.03], [2136, 4, 0.02], [2500, 5, 0.01], [3000, 6, 0.01], [3000, 8, 0.005], [3500, 8, 0.005], [4000, 10, 0.003]]) {
    const kanji = buildTable(stats, dic, { maxKanji: k, maxCands: c, minShare: s, placeWeight: pw, alpha: 0.5 });
    const sc = { ...emptyScorer(), quantBits: 4 };
    const m0 = decodeModel(encodeModel({ kanji, lexicon: [], priorBits: 4, scorer: sc }));
    const varPen = variantPenalties(m0, train);
    for (const pb of [0, 3, 4]) {
      const bytes = encodeModel({ kanji, lexicon: [], priorBits: pb, scorer: { ...sc, varPen } });
      if (pb === 0 && false) continue;
      console.log(`pw=${pw} kanji=${k} cands=${c} minShare=${s} priorBits=${pb} bytes=${bytes.length} devA=${(devAccuracy(decodeModel(bytes), dev) * 100).toFixed(1)}%`);
    }
  }
}
