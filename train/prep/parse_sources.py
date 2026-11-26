"""Parse raw EDRDG / JmdictFurigana sources into compact TSV work files.

Inputs (data/raw):  JMdict.gz, JMnedict.xml.gz, kanjidic2.xml.gz,
                    JmdictFurigana.txt, JmnedictFurigana.txt
Outputs (data/work): kanji.tsv, words.tsv, places.tsv

Deterministic: no randomness.
"""
import gzip
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
WORK = ROOT / "data" / "work"
ENT = re.compile(r"&([A-Za-z0-9_.-]+);")
STD = {"lt", "gt", "amp", "quot", "apos"}


def entries(path, tag):
    """Yield parsed XML elements for each <tag>...</tag> block (entity names kept as text)."""
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    buf, inside = [], False
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not inside:
                if line.startswith(open_t):
                    inside, buf = True, [line]
                continue
            buf.append(line)
            if line.startswith(close_t):
                inside = False
                text = ENT.sub(lambda m: m.group(0) if m.group(1) in STD else m.group(1), "".join(buf))
                yield ET.fromstring(text)


def load_furigana(path):
    """surface|reading|furigana -> {(surface, reading): 'i:yomi;j-k:yomi'}"""
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) == 3:
                out[(parts[0], parts[1])] = parts[2]
    return out


NF = re.compile(r"nf(\d\d)")


def priority(pris):
    """Map JMdict priority tags to a coarse frequency class 0 (none) .. 5 (very common)."""
    best = 0
    for p in pris:
        m = NF.fullmatch(p)
        if m:
            n = int(m.group(1))
            best = max(best, 5 if n <= 6 else 4 if n <= 16 else 3 if n <= 30 else 2)
        elif p in ("news1", "ichi1", "spec1", "gai1"):
            best = max(best, 3)
        elif p in ("news2", "ichi2", "spec2", "gai2"):
            best = max(best, 2)
    return best


def parse_kanjidic():
    rows = []
    for c in entries(RAW / "kanjidic2.xml.gz", "character"):
        lit = c.findtext("literal")
        misc = c.find("misc")
        grade = misc.findtext("grade") or ""
        freq = misc.findtext("freq") or ""
        on = [r.text for r in c.iter("reading") if r.get("r_type") == "ja_on"]
        kun = [r.text for r in c.iter("reading") if r.get("r_type") == "ja_kun"]
        nan = [n.text for n in c.iter("nanori")]
        rows.append((lit, grade, freq, ",".join(on), ",".join(kun), ",".join(nan)))
    with open(WORK / "kanji.tsv", "w", encoding="utf-8") as f:
        f.write("char\tgrade\tfreq\ton\tkun\tnanori\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print("kanji", len(rows), file=sys.stderr)


BAD_KINF = {"rK", "oK", "sK", "iK"}
# Food-domain signal from English glosses (JMdict's "food" field tag covers only ~1,600 words).
FOOD_RX = re.compile(r"\b(food|dish|cuisine|noodles?|soup|sauce|meat|beef|pork|chicken|fish|seafood|shellfish|squid|"
                     r"octopus|shrimp|prawn|crab|roe|scallops?|clam|oyster|eel|tuna|salmon|vegetables?|fruits?|beans?|"
                     r"tofu|rice|bread|cake|sweets|confectionery|candy|dessert|tea|coffee|sake|beer|wine|liquor|"
                     r"drink|beverage|juice|egg|cheese|butter|milk|pickles?|pickled|miso|soy|seaweed|mushrooms?|"
                     r"radish|onion|garlic|ginger|pepper|spice|seasoning|vinegar|dumplings?|fried|grilled|broth|"
                     r"gizzard|offal|liver|sushi|sashimi|tempura|curry|salad|snack|sesame|wheat|flour|potato|yam|"
                     r"taro|cabbage|cucumber|tomato|carrot|melon|apple|lemon|peach|grape|citrus|chestnut|nuts?|"
                     r"tentacles?|skewer|bowl of)\b", re.I)
BAD_MISC = {"arch", "obsc", "rare", "obs"}


def parse_jmdict(furi):
    n = 0
    with open(WORK / "words.tsv", "w", encoding="utf-8") as out:
        out.write("surface\treading\tfreq\tflags\tfurigana\n")
        for e in entries(RAW / "JMdict.gz", "entry"):
            kebs = []
            for k in e.findall("k_ele"):
                inf = {x.text for x in k.findall("ke_inf")}
                if inf & BAD_KINF:
                    continue
                kebs.append((k.findtext("keb"), priority([p.text for p in k.findall("ke_pri")]), inf))
            if not kebs:
                continue
            senses = e.findall("sense")
            misc = {m.text for s in senses for m in s.findall("misc")}
            fields = {m.text for s in senses for m in s.findall("field")}
            en = " ".join(g.text or "" for s in senses for g in s.findall("gloss")
                          if not g.get("{http://www.w3.org/XML/1998/namespace}lang"))
            if FOOD_RX.search(en):
                fields = fields | {"foodish"}
            pos = {m.text for s in senses for m in s.findall("pos")}
            sense_all_bad = senses and all(
                {m.text for m in s.findall("misc")} & BAD_MISC for s in senses if s.findall("gloss") and not s.findall("gloss")[0].get("{http://www.w3.org/XML/1998/namespace}lang")
            )
            emitted = set()
            for r in e.findall("r_ele"):
                if r.find("re_nokanji") is not None:
                    continue
                rinf = {x.text for x in r.findall("re_inf")}
                if rinf & {"ok", "ik", "sk", "rk"}:
                    continue
                restr = {x.text for x in r.findall("re_restr")}
                rpri = priority([p.text for p in r.findall("re_pri")])
                reb = r.findtext("reb")
                for keb, kpri, kinf in kebs:
                    if restr and keb not in restr:
                        continue
                    fg = furi.get((keb, reb))
                    if fg is None:
                        continue
                    flags = []
                    if "food" in fields:
                        flags.append("food")
                    elif "foodish" in fields:
                        flags.append("foodish")
                    if "uk" in misc:
                        flags.append("uk")
                    if sense_all_bad:
                        flags.append("rare")
                    if "ateji" in kinf:
                        flags.append("ateji")
                    if any(p.startswith("v") for p in pos):
                        flags.append("verb")
                    if any(p.startswith("adj") for p in pos):
                        flags.append("adj")
                    if "ctr" in pos:
                        flags.append("ctr")
                    fq = min(kpri, rpri) if (kpri and rpri) else max(kpri, rpri) if len(kebs) == 1 else kpri
                    out.write(f"{keb}\t{reb}\t{fq}\t{','.join(flags)}\t{fg}\n")
                    emitted.add(keb)
                    n += 1
            # Words whose readings are all katakana (餃子 ギョーザ, 烏龍茶 ウーロンちゃ) have no furigana alignment.
            # Keep them as whole-word spans; prefer a hiragana reading (including search-only "sK" ones).
            for keb, kpri, kinf in kebs:
                if keb in emitted:
                    continue
                rebs = []
                for r in e.findall("r_ele"):
                    restr = {x.text for x in r.findall("re_restr")}
                    rinf = {x.text for x in r.findall("re_inf")}
                    if r.find("re_nokanji") is not None or (restr and keb not in restr) or rinf & {"ok", "ik", "rk"}:
                        continue
                    reb = r.findtext("reb")
                    hira = all("\u3041" <= ch <= "\u309f" for ch in reb)
                    rebs.append((0 if hira else 1, 0 if "sk" not in rinf else 1 if hira else 2, reb))
                if not rebs:
                    continue
                reb = sorted(rebs, key=lambda x: (x[0], -x[1] if x[0] == 0 else x[1]))[0][2]
                flags = ["food"] if "food" in fields else ["foodish"] if "foodish" in fields else []
                flags.append("span")
                if sense_all_bad:
                    flags.append("rare")
                out.write(f"{keb}\t{reb}\t{kpri}\t{','.join(flags)}\t0-{len(keb) - 1}:{reb}\n")
                n += 1
    print("words", n, file=sys.stderr)


def parse_food_kana():
    """Kana-only JMdict entries tagged with the food field (used for synthetic product names)."""
    n = 0
    with open(WORK / "food_kana.tsv", "w", encoding="utf-8") as out:
        out.write("surface\treading\tfreq\n")
        for e in entries(RAW / "JMdict.gz", "entry"):
            if e.find("k_ele") is not None:
                continue
            fields = {m.text for s in e.findall("sense") for m in s.findall("field")}
            if "food" not in fields:
                continue
            for r in e.findall("r_ele"):
                reb = r.findtext("reb")
                out.write(f"{reb}\t{reb}\t{priority([p.text for p in r.findall('re_pri')])}\n")
                n += 1
    print("food_kana", n, file=sys.stderr)


def parse_jmnedict(furi):
    n = 0
    with open(WORK / "places.tsv", "w", encoding="utf-8") as out:
        out.write("surface\treading\ttypes\tfurigana\n")
        for e in entries(RAW / "JMnedict.xml.gz", "entry"):
            types = {t.text for t in e.iter("name_type")}
            keep = types & {"place", "station"}
            if not keep:
                continue
            kebs = [k.findtext("keb") for k in e.findall("k_ele")]
            for r in e.findall("r_ele"):
                reb = r.findtext("reb")
                restr = {x.text for x in r.findall("re_restr")}
                for keb in kebs:
                    if restr and keb not in restr:
                        continue
                    fg = furi.get((keb, reb))
                    if fg is None:
                        continue
                    out.write(f"{keb}\t{reb}\t{','.join(sorted(types))}\t{fg}\n")
                    n += 1
    print("places", n, file=sys.stderr)


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    parse_kanjidic()
    parse_jmdict(load_furigana(RAW / "JmdictFurigana.txt"))
    parse_jmnedict(load_furigana(RAW / "JmnedictFurigana.txt"))
    parse_food_kana()
