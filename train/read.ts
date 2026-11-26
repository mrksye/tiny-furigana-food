// Build-time helper: pick the decoder that matches the model (two-loop only when it carries a chunker).
import type { Model } from "../src/model";
import { readerFromModel } from "../src/reader";
import { twoLoop } from "./hooks";
export const readerOf = (m: Model) => (m.chunker?.bits ? twoLoop().reader(m) : readerFromModel(m));
