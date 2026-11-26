// Plug the experimental two-loop decoder into the build code (train/hooks.ts). Import for side effects.

import { registerTwoLoop } from "../../train/hooks";
import { trainChunker } from "../tools/chunker";
import { decodeModelFull, readerFromModel2 } from "./reader2";

registerTwoLoop({ decode: decodeModelFull, reader: readerFromModel2, trainChunker });
