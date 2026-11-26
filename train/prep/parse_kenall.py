"""Parse Japan Post KEN_ALL (UTF-8 edition) into place-name units with official readings.

Input : data/raw/utf_ken_all.zip
Output: data/work/kenall.tsv  (surface, reading[hiragana], level, code)
  level = pref | city | town
Rules:
  - town rows that are notes ("以下に掲載がない場合", "の次に番地がくる場合", "一円") are skipped
  - parenthesised parts are removed; rows that only continue a multi-row parenthesis are skipped
  - town names with digits / 地割 / 番地 / 丁目 are skipped
  - "X郡Y町" additionally yields "Y町" when the reading splits uniquely at グン
Deterministic; no randomness.
"""
import csv
import io
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = ("以下に掲載がない場合", "の次に番地がくる場合", "一円", "次のビルを除く")
BAD = re.compile(r"[0-9０-９〇一二三四五六七八九十]+(地割|番地|丁目|線|号)|地割|番地|丁目|～|、|・")


def hira(s):
    s = unicodedata.normalize("NFKC", s)
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in s)


def clean(surface, reading):
    if "（" in surface or "(" in surface:
        surface = re.split(r"[（(]", surface)[0]
        reading = re.split(r"[（(]", reading)[0]
    elif "）" in surface or ")" in surface:
        return None
    if not surface or any(x in surface for x in SKIP) or BAD.search(surface):
        return None
    return surface, reading


def main():
    out = {}
    with zipfile.ZipFile(ROOT / "data/raw/utf_ken_all.zip") as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        text = z.read(name).decode("utf-8")
    for row in csv.reader(io.StringIO(text)):
        code, pk, ck, tk, pref, city, town = row[0], row[3], row[4], row[5], row[6], row[7], row[8]
        out.setdefault((pref, hira(pk)), ("pref", code[:2]))
        out.setdefault((city, hira(ck)), ("city", code))
        if "郡" in city:
            a, _, b = city.partition("郡")
            parts = hira(ck).split("ぐん")
            if b and len(parts) == 2 and parts[1]:
                out.setdefault((b, parts[1]), ("city", code))
        c = clean(town, tk)
        if c:
            s, r = c
            out.setdefault((s, hira(r)), ("town", code))
    # Product names use municipality stems (宇治抹茶, not 宇治市抹茶): add "X" for "X市/町/村" when the
    # reading ends with the suffix reading.
    SUF = [("市", "し"), ("町", "まち"), ("町", "ちょう"), ("村", "むら"), ("村", "そん")]
    for (s_, r_), (lv, cd) in list(out.items()):
        if lv != "city" or len(s_) < 3 or "郡" in s_ or "市" in s_[:-1]:
            continue
        for ks, rs in SUF:
            if s_.endswith(ks) and r_.endswith(rs):
                out.setdefault((s_[:-1], r_[: -len(rs)]), ("city", cd))
                break
    kana = re.compile(r"^[\u3041-\u3096ー]+$")
    rows = sorted((s, r, lv, cd) for (s, r), (lv, cd) in out.items() if s and r and kana.match(r))
    with open(ROOT / "data/work/kenall.tsv", "w", encoding="utf-8") as f:
        f.write("surface\treading\tlevel\tcode\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    from collections import Counter
    print("kenall units", len(rows), Counter(r[2] for r in rows), file=sys.stderr)


if __name__ == "__main__":
    main()
