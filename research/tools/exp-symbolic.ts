// Experiment A: symbolic only (candidate table + statistical priors, no context model).

import { decodeModelFull as decodeModel } from "../runtime/reader2";
import { emptyScorer } from "../../src/model";
import { readerFromModel } from "../../src/reader";
import { loadKanjidic, loadTrain } from "../../train/data";
import { encodeModel } from "../../train/encode";
import { evaluateAll, formatResults } from "./evaluate";
import { buildTable, collectStats, variantPenalties } from "../../train/table";

const t0 = performance.now();
const [ex, dic] = await Promise.all([loadTrain(), loadKanjidic()]);
console.log(`loaded ${ex.length} examples in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
const placeWeight = +(process.env.PW ?? 1);
const stats = collectStats(ex, dic, placeWeight);

for (const [maxKanji, maxCands, minShare, priorBits] of [[2136, 4, 0.02, 4], [2500, 6, 0.01, 4], [3000, 8, 0.005, 4], [4000, 10, 0.003, 4]]) {
  const kanji = buildTable(stats, dic, { maxKanji, maxCands, minShare, placeWeight, alpha: 0.5 });
  const sc = emptyScorer();
  const m0 = decodeModel(encodeModel({ kanji, lexicon: [], priorBits, scorer: { ...sc, quantBits: 8 } }));
  const varPen = variantPenalties(m0, ex);
  const bytes = encodeModel({ kanji, lexicon: [], priorBits, scorer: { ...sc, varPen, quantBits: 8 } });
  const m = decodeModel(bytes);
  console.log("varPen", Array.from(varPen, (x) => x.toFixed(2)).join(" "));
  const read = readerFromModel(m);
  const res = await evaluateAll(read, m);
  console.log(`\n# A symbolic  kanji=${kanji.length} maxCands=${maxCands} minShare=${minShare} cands=${m.candReading.length} readings=${m.readings.length} bytes=${bytes.length}`);
  console.log(formatResults(res));
}
