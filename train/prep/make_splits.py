"""Build leak-controlled train / eval splits.

Split unit is a *group*:
  - JMdict words: the surface string.
  - Place names : the surface with administrative suffixes stripped
                  (箕面 / 箕面市 / 箕面駅 -> 箕面), so variants of one place never
                  straddle train and test.
Groups are assigned to test by a salted MD5 hash (deterministic, seed = SALT).
After assignment, every training item whose surface *contains* a test surface
(superstring) is removed, so the test name cannot be memorised through a longer
entry (e.g. test 箕面 removes train 箕面市, 箕面駅前). Training items that are
*substrings* of a test item are kept on purpose: composing known parts into an
unknown compound is exactly the generalisation being measured.
"""
import csv
import hashlib
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data" / "work"
EVAL = ROOT / "research" / "data" / "eval"
SALT = "tiny-g2p-split-v1"
TEST_FRAC = 0.04
SEED = 20261125
KANJI = re.compile(r"[㐀-鿿豈-﫿々〆]")
SUFFIX = re.compile(r"(都|道|府|県|市|区|町|村|郡|駅|島|川|山|湖|港|橋|温泉|丁目)+$")
FREQ_W = {0: 1.0, 2: 3.0, 3: 5.0, 4: 8.0, 5: 12.0}
KW = {"pref": 5.0, "city": 3.0, "town": 0.1}  # KEN_ALL weights


def is_test(key, frac=TEST_FRAC):
    h = hashlib.md5((SALT + key).encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32 < frac


def read_tsv(name):
    with open(WORK / name, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def place_group(s):
    g = SUFFIX.sub("", s)
    return g if len(g) >= 1 else s


def main():
    words = [w for w in read_tsv("words.tsv") if KANJI.search(w["surface"]) and "rare" not in w["flags"]]
    places = [p for p in read_tsv("places.tsv") if KANJI.search(p["surface"])]

    # primary reading per surface: highest freq class, ties -> dictionary order
    prim = {}
    for w in words:
        s, fq = w["surface"], int(w["freq"])
        if s not in prim or fq > prim[s][1]:
            prim[s] = (w["reading"], fq)
    pprim = {}
    for p in places:
        pprim.setdefault(p["surface"], p["reading"])

    test_words = {s for s in prim if is_test("w:" + s)}
    test_places = {s for s in pprim if is_test("p:" + place_group(s))}
    kenall = read_tsv("kenall.tsv") if (WORK / "kenall.tsv").exists() else []
    kprim = {}
    for k in kenall:
        kprim.setdefault(k["surface"], (k["reading"], k["level"]))
    # KEN_ALL town names are split like other places; prefecture / municipality names are always training data
    test_ktowns = {s for s, (r, lv) in kprim.items() if lv == "town" and KANJI.search(s) and is_test("k:" + place_group(s))}
    test_surfaces = test_words | test_places | test_ktowns

    def leaks(s):
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n + 1):
                if s[i:j] in test_surfaces:
                    return True
        return False

    train = []
    dropped = 0
    for w in words:
        if leaks(w["surface"]):
            dropped += 1
            continue
        wt = FREQ_W[int(w["freq"])]
        if prim[w["surface"]][0] != w["reading"]:
            wt *= 0.3
        if "uk" in w["flags"]:
            wt *= 0.5
        train.append((w["surface"], w["reading"], w["furigana"], f"{wt:.2f}", "word", w["flags"]))
    for p in places:
        if leaks(p["surface"]):
            dropped += 1
            continue
        wt = 0.3 if pprim[p["surface"]] == p["reading"] else 0.1
        train.append((p["surface"], p["reading"], p["furigana"], f"{wt:.2f}", "place", p["types"]))
    kdrop = 0
    for srf, (r, lv) in sorted(kprim.items()):
        if not KANJI.search(srf) or srf in test_ktowns:
            continue
        if leaks(srf):
            kdrop += 1
            continue
        wt = KW[lv]
        # furigana column left empty: aligned later against the symbolic candidate table (train/align.ts)
        train.append((srf, r, "", f"{wt:.2f}", "place", "kenall-" + lv))
    print("kenall train", sum(1 for t in train if t[5].startswith("kenall")), "dropped(leak)", kdrop, file=sys.stderr)
    # release training set: identical weighting but nothing held out (test sets are not valid for models trained on it)
    full = []
    for w in words:
        wt = FREQ_W[int(w["freq"])] * (0.3 if prim[w["surface"]][0] != w["reading"] else 1) * (0.5 if "uk" in w["flags"] else 1)
        full.append((w["surface"], w["reading"], w["furigana"], f"{wt:.2f}", "word", w["flags"]))
    for p_ in places:
        full.append((p_["surface"], p_["reading"], p_["furigana"], f"{0.3 if pprim[p_['surface']] == p_['reading'] else 0.1:.2f}", "place", p_["types"]))
    for srf, (r, lv) in sorted(kprim.items()):
        if KANJI.search(srf):
            full.append((srf, r, "", f"{KW[lv]:.2f}", "place", "kenall-" + lv))
    full.sort()
    with open(WORK / "train_all.tsv", "w", encoding="utf-8") as f:
        f.write("surface\treading\tfurigana\tweight\tsource\tflags\n")
        for r in full:
            f.write("\t".join(r) + "\n")
    print("train_all", len(full), file=sys.stderr)
    train.sort()
    with open(WORK / "train.tsv", "w", encoding="utf-8") as f:
        f.write("surface\treading\tfurigana\tweight\tsource\tflags\n")
        for r in train:
            f.write("\t".join(r) + "\n")
    print("train", len(train), "dropped(leak)", dropped, file=sys.stderr)

    rng = random.Random(SEED)
    train_surf = {r[0] for r in train}
    all_readings = defaultdict(set)
    for w in words:
        all_readings[w["surface"]].add(w["reading"])
    for p in places:
        all_readings[p["surface"]].add(p["reading"])

    def write(name, rows):
        EVAL.mkdir(parents=True, exist_ok=True)
        with open(EVAL / name, "w", encoding="utf-8") as f:
            f.write("input\texpected\talternatives\n")
            for s, r in rows:
                alts = "|".join(sorted(all_readings.get(s, set()) - {r}))
                f.write(f"{s}\t{r}\t{alts}\n")
        print(name, len(rows), file=sys.stderr)

    def sample(xs, n):
        xs = sorted(xs)
        rng.shuffle(xs)
        return xs[:n]

    # known words: frequent training words (reproduction)
    known = [s for s, (r, fq) in prim.items() if s in train_surf and fq >= 2]
    write("words_known.tsv", [(s, prim[s][0]) for s in sample(known, 3000)])

    # unknown words, split into simple vs compound (decomposable into >=2 training words)
    tw = sorted(test_words)
    def decomposable(s):
        n = len(s)
        best = [None] * (n + 1)
        best[0] = 0
        for j in range(1, n + 1):
            for i in range(j):
                if best[i] is not None and (j - i >= 2 or KANJI.match(s[i:j]) is None) and s[i:j] in train_surf and s[i:j] != s:
                    if best[j] is None or best[i] + 1 < best[j]:
                        best[j] = best[i] + 1
        return best[n] is not None and best[n] >= 2
    common = [s for s in tw if prim[s][1] >= 2]
    rare = [s for s in tw if prim[s][1] < 2]
    comp = [s for s in tw if len(s) >= 3 and decomposable(s)]
    write("words_unknown_common.tsv", [(s, prim[s][0]) for s in sample(common, 1500)])
    write("words_unknown_rare.tsv", [(s, prim[s][0]) for s in sample(rare, 1500)])
    write("compounds_unknown.tsv", [(s, prim[s][0]) for s in sample(comp, 1500)])
    food = [w["surface"] for w in words if "food" in w["flags"] and w["surface"] in test_words]
    write("food_unknown.tsv", [(s, prim[s][0]) for s in sample(set(food), 500)])

    # unknown place names
    tp = sorted(test_places)
    write("places_unknown.tsv", [(s, pprim[s]) for s in sample(tp, 3000)])

    # KEN_ALL: held-out town names (official readings) and known municipality names
    if kprim:
        write_k = lambda name, xs: write(name, xs)
        write_k("kenall_towns_unknown.tsv", [(t, kprim[t][0]) for t in sample(test_ktowns, 1500)])
        ktrain = {t[0] for t in train if t[5].startswith("kenall")}
        cities = [t for t, (r, lv) in kprim.items() if lv == "city" and t in ktrain]
        write_k("kenall_municipal_known.tsv", [(t, kprim[t][0]) for t in sample(cities, 1000)])
        # realistic product names: known municipality stem (suffix stripped) + known food word
        SUF = [("市", "し"), ("区", "く"), ("町", "まち"), ("町", "ちょう"), ("村", "むら"), ("村", "そん")]
        stems = []
        for t in cities:
            r = kprim[t][0]
            for ks, rs in SUF:
                if t.endswith(ks) and r.endswith(rs) and len(t) >= 3:
                    stems.append((t[:-1], r[: -len(rs)]))
                    break
        foods_k = [(s2, prim[s2][0]) for s2 in sorted(train_surf & set(prim))
                   if len(s2) >= 2 and prim[s2][1] >= 2 and s2 in {w["surface"] for w in words if "food" in w["flags"]}]
        kana_food2 = [(r["surface"], r["reading"]) for r in read_tsv("food_kana.tsv") if re.fullmatch(r"[ァ-ヶー]{2,}", r["surface"])]
        rows2 = []
        for i, (st, sr) in enumerate(sample(stems, 1000)):
            pool = foods_k if i % 2 == 0 else kana_food2
            fs, fr = pool[rng.randrange(len(pool))]
            fr = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in fr)
            rows2.append((st + fs, sr + fr))
        with open(EVAL / "kenall_products_synth.tsv", "w", encoding="utf-8") as f:
            f.write("input\texpected\talternatives\n")
            for a1, b1 in rows2:
                f.write(f"{a1}\t{b1}\t\n")
        print("kenall_products_synth.tsv", len(rows2), file=sys.stderr)

    # famous known places: training place names contained in >= 10 other training place names
    tplaces = sorted({r[0] for r in train if r[4] == "place"})
    contain = defaultdict(int)
    for s in tplaces:
        subs = {s[i:j] for i in range(len(s)) for j in range(i + 2, min(len(s), i + 4) + 1)} - {s}
        for x in subs:
            contain[x] += 1
    famous = [s for s in tplaces if contain[s] >= 10 and s in pprim]
    write("places_known_famous.tsv", [(s, pprim[s]) for s in sample(famous, 1000)])

    # synthetic place + product names: unknown place + known food word
    foods = [(s, prim[s][0]) for s in sorted(train_surf & set(prim))
             if len(s) >= 2 and prim[s][1] >= 2 and s in {w["surface"] for w in words if "food" in w["flags"]}]
    kana_food = [(r["surface"], r["reading"]) for r in read_tsv("food_kana.tsv") if re.fullmatch(r"[ァ-ヶー]{2,}", r["surface"])]
    short_places = [s for s in tp if 2 <= len(s) <= 3]
    rows = []
    pl = sample(short_places, 1000)
    for i, s in enumerate(pl):
        f_s, f_r = (foods if i % 2 == 0 else kana_food)[rng.randrange(len(foods if i % 2 == 0 else kana_food))]
        f_r = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in f_r)
        rows.append((s + f_s, pprim[s] + f_r))
    with open(EVAL / "place_products_synth.tsv", "w", encoding="utf-8") as f:
        f.write("input\texpected\talternatives\n")
        for s, r in rows:
            f.write(f"{s}\t{r}\t\n")
    print("place_products_synth.tsv", len(rows), file=sys.stderr)


if __name__ == "__main__":
    main()
