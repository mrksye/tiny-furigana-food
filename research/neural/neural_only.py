"""System B: neural-only reading generation (no reading dictionary at runtime).

Per input character, a BiGRU predicts up to K kana (slot-wise softmax, 0 = end).
Characters of a multi-character aligned span (e.g. 海老 -> えび) put the whole
reading on the first character and "nothing" on the others. Non-kanji characters
are copied verbatim at inference (kana -> hiragana), so their loss is masked.

Outputs float weights as JSON; research/tools/neural-export.ts quantises + encodes them so
that the binary format is shared with the TypeScript decoder.

usage: python -I research/neural/neural_only.py --emb 16 --hid 32 --vocab 2500 --epochs 6 --out .cache/nn-small.json
"""
import argparse
import json
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
K = 4  # kana slots per input character
KANA = [chr(c) for c in range(0x3041, 0x3097)] + ["ー"]
KID = {c: i + 1 for i, c in enumerate(KANA)}  # 0 = end-of-reading
A = len(KANA) + 1
HIRA = [chr(c) for c in range(0x3041, 0x3097)]


def is_kanji(c):
    o = ord(c)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or o >= 0x20000 or c in "々〆"


def to_hira(s):
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in s)


def char_class(c):
    o = ord(c)
    if 0x3041 <= o <= 0x3096:
        return "H:" + c  # each hiragana is its own token (okurigana matter)
    if 0x30A1 <= o <= 0x30FC:
        return "<KATA>"
    if c.isdigit():
        return "<DIGIT>"
    if c.isascii() and c.isalpha():
        return "<LATIN>"
    return "<OTHER>"


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            surface, reading, furi, weight, source, flags = line.rstrip("\n").split("\t")
            s = unicodedata.normalize("NFKC", surface)
            if len(s) != len(surface) or len(s) > 16:
                continue
            tgt = [""] * len(s)
            ok = True
            covered = [False] * len(s)
            for part in furi.split(";"):
                pos, _, yomi = part.partition(":")
                a, _, b = pos.partition("-")
                a = int(a); b = int(b) if b else a
                if b >= len(s):
                    ok = False; break
                tgt[a] = to_hira(yomi)
                for i in range(a, b + 1):
                    covered[i] = True
            if not ok:
                continue
            for i, c in enumerate(s):
                if not covered[i]:
                    tgt[i] = to_hira(c)
            if any(len(t) > K for t, c in zip(tgt, s) if is_kanji(c)):
                continue
            if any(ch not in KID for t, c in zip(tgt, s) if is_kanji(c) for ch in t):
                continue
            rows.append((s, tgt, float(weight)))
    return rows


class Net(nn.Module):
    def __init__(self, V, E, H):
        super().__init__()
        self.emb = nn.Embedding(V, E)
        self.gru = nn.GRU(E, H, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * H, K * A)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.out(h).view(x.shape[0], x.shape[1], K, A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--hid", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=2500, help="number of kanji with their own embedding")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(3)

    rows = load(ROOT / "data/work/train.tsv")
    freq = Counter()
    for s, _, w in rows:
        for c in s:
            if is_kanji(c):
                freq[c] += w
    kanji = [c for c, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[: args.vocab]]
    specials = ["<PAD>", "<UNK_KANJI>", "<KATA>", "<DIGIT>", "<LATIN>", "<OTHER>"] + ["H:" + h for h in HIRA]
    vocab = {t: i for i, t in enumerate(specials + kanji)}

    def tok(c):
        if is_kanji(c):
            return vocab.get(c, 1)
        return vocab[char_class(c)]

    random.shuffle(rows)
    dev, train = rows[:2000], rows[2000:]

    def tensors(batch):
        L = max(len(s) for s, _, _ in batch)
        x = torch.zeros(len(batch), L, dtype=torch.long)
        y = torch.zeros(len(batch), L, K, dtype=torch.long)
        m = torch.zeros(len(batch), L)
        w = torch.tensor([b[2] for b in batch])
        for i, (s, tgt, _) in enumerate(batch):
            for j, c in enumerate(s):
                x[i, j] = tok(c)
                if is_kanji(c):
                    m[i, j] = 1
                    for k, ch in enumerate(tgt[j]):
                        y[i, j, k] = KID[ch]
        return x, y, m, w

    net = Net(len(vocab), args.emb, args.hid)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params={n_params} vocab={len(vocab)} train={len(train)}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss(reduction="none")
    train.sort(key=lambda r: len(r[0]))  # bucket by length
    batches = [train[i:i + 256] for i in range(0, len(train), 256)]

    def decode(s, logits):
        out = ""
        for j, c in enumerate(s):
            if not is_kanji(c):
                out += to_hira(c)
                continue
            for k in range(K):
                a = int(logits[j, k].argmax())
                if a == 0:
                    break
                out += KANA[a - 1]
        return out

    def evaluate():
        net.eval()
        ok = 0
        with torch.no_grad():
            for i in range(0, len(dev), 256):
                b = dev[i:i + 256]
                x, _, _, _ = tensors(b)
                lg = net(x)
                for bi, (s, tgt, _) in enumerate(b):
                    ok += decode(s, lg[bi]) == "".join(tgt)
        net.train()
        return ok / len(dev)

    for ep in range(args.epochs):
        random.shuffle(batches)
        t0, tot = time.time(), 0.0
        for b in batches:
            x, y, m, w = tensors(b)
            lg = net(x)
            l = lossf(lg.reshape(-1, A), y.reshape(-1)).view(y.shape).sum(-1)
            loss = (l * m * w[:, None]).sum() / (m * w[:, None]).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        for g in opt.param_groups:
            g["lr"] *= 0.7
        print(f"epoch {ep + 1} loss={tot / len(batches):.4f} dev={evaluate() * 100:.2f}% {time.time() - t0:.0f}s", flush=True)

    sd = {k: v.detach().numpy().tolist() for k, v in net.state_dict().items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"emb": args.emb, "hid": args.hid, "K": K, "kanji": kanji, "specials": specials, "state": sd,
                   "seed": args.seed, "epochs": args.epochs, "params": n_params}, f)


if __name__ == "__main__":
    main()
