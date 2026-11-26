# Data sources and licences

This file records every data source used, its licence as stated by the publisher, how it is used,
and what it implies for the files in this repository. Where something could not be confirmed,
it is marked **unconfirmed** instead of guessed.

All raw files were downloaded on 2026-11-25. SHA-256 checksums: [`RAW_SHA256SUMS.txt`](RAW_SHA256SUMS.txt).
Raw files are **not** committed (`data/raw/` is git-ignored); `train/prep/` regenerates everything from them.

| Source | Version / file | Publisher licence (as stated) | Used for | Shipped at runtime? |
|---|---|---|---|---|
| JMdict | `JMdict.gz`, file date 2026-11-24 | CC BY-SA 4.0, EDRDG General Dictionary Licence | word readings, frequency tags, food field | Indirectly (see below) |
| KANJIDIC2 | `kanjidic2.xml.gz`, file date 2026-11-24 | CC BY-SA 4.0, EDRDG General Dictionary Licence | on/kun reading classes, jōyō grade | Indirectly |
| JMnedict | `JMnedict.xml.gz`, file date 2026-11-24 | CC BY-SA 4.0, EDRDG General Dictionary Licence (covers "ENAMDICT", the Japanese names file) | place / station names | Indirectly |
| JmdictFurigana / JmnedictFurigana | release `2.3.1+2026-10-25` (github.com/Doublevil/JmdictFurigana) | Repository licence: MIT. The data is derived from JMdict/JMnedict, so the EDRDG licence also applies to it | per-character alignment of readings | No (training only) |
| Japan Post postal-code data (KEN_ALL) | `utf_ken_all.zip` (UTF-8 edition, archive date 2026-10-23; downloaded manually in a browser because automated downloads returned HTML) | Japan Post README: "郵便番号データに限っては日本郵便株式会社は著作権を主張しません。自由に配布していただいて結構です。" | official readings of prefectures, municipalities (and their stems) and town names | Indirectly (lexicon entries and learned weights) |
| Private menu data | 76 menu items of the author's food-truck app | Owner's private business data | **Evaluation only** (never trained on). The file and derived TSVs are git-ignored and not published | No |
| Jōyō kanji list | — | — | Taken from KANJIDIC2 grade fields (1–6 and 8), not from the official list itself | — |
| Wikipedia / Wiktionary | — | — | **Not used** in this study | — |

EDRDG licence text: <https://www.edrdg.org/edrdg/licence.html>.
Project pages: <https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project>,
<https://www.edrdg.org/wiki/index.php/KANJIDIC_Project>.

## What is derived from what

| Repository artefact | Derived from | Notes |
|---|---|---|
| `dist/model.bin`, `research/models/*.bin` (lattice systems A/C/D) | JMdict, KANJIDIC2, JMnedict (through the furigana alignment) | Contains a reading table (kanji → reading strings), an exception lexicon (word → reading) and learned weights |
| `research/models/B-*.bin` (neural only) | same | Contains only network weights and a kanji code-point list |
| `research/data/eval/*.tsv` | JMdict, JMnedict, KEN_ALL | Word/reading pairs; `spec_examples.tsv` comes from the project specification; `menu_items*.tsv` are private and git-ignored |
| `research/results/results.jsonl`, `docs/*.md` | measurements | — |

## Implications (please review before publishing)

1. **The EDRDG licence states that it applies to "any data files which are derived from" the dictionary files**, and the ShareAlike term says derived works may be distributed "only under the same, similar or a compatible licence".
   The reading table and exception lexicon inside `model.bin` are direct extracts, so the model files and `research/data/eval/*.tsv` should be distributed under **CC BY-SA 4.0** with attribution to EDRDG.
2. Whether purely learned weights (system B, or the hashed scorer weights of system C) are "derived data files" in the licence's sense is **unconfirmed**; this study does not take a position. Treating all model files as CC BY-SA 4.0 is the conservative choice and is what this repository assumes.
3. EDRDG requires software that uses the files to acknowledge the source in its documentation and to provide (links to) the licence; the demo page carries an acknowledgement footer, and the README links here. Applications that embed the model should do the same (e.g. an "About / Sources" screen).
4. Commercial use is permitted by EDRDG provided the conditions above are met. The EDRDG licence also says users must not claim copyright over material from the files.
5. The **source code** (everything under `src/`, `train/`, `research/` (code), `tests/`, `demo/`) contains no dictionary data and is licensed independently under the MIT licence (see `LICENSE`).

## Leakage-relevant facts

- The test splits are produced by `train/prep/make_splits.py` (salted MD5 group split, salt `tiny-g2p-split-v1`, 4 %) with superstring removal; see `docs/REPORT.md` §4.
- The 18 specification examples were **not** removed from training. Their membership in the training data is listed in REPORT §7.3; no example is special-cased in code or data.
