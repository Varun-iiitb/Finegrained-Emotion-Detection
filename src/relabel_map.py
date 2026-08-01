"""Repaired emotion label map: recover the REAL emotions the wheel wrongly drops.

Diagnosis (2026-06-30, see CLAUDE.md / memory label-space-fragmentation): the
OV-MER wheel's case2 mapping drops 2,256/3,311 vocab words to None — but only
~113 of those are real frequent emotions (freq>=20), e.g. tension(1863),
gratitude(318), humorous(697), sarcasm(119); the other ~1,931 are rare auto-
annotation junk that SHOULD drop. This module fixes the bug WITHOUT touching the
established taxonomy: it routes each frequent dropped word (freq>=MIN_FREQ) to its
nearest of the existing wheel canonicals using BGE-M3 cosine, with a FLOOR so
genuine junk (far from any emotion) stays dropped. Validated routings:
  sarcasm->sarcastic(.80)  condescending->dismissive(.86)  sincere->open(.87)

Output: data/cache/repaired_label_map.json = {word: canonical} covering the
wheel-mapped words + the recovered ones. Words not in the map are still dropped.

Importable:
  m = load_repaired(path)              # -> RepairedGrouper, drop-in for WheelGrouper
  m.to_group_set(words, *_ , **_)      # same call signature; ignores level/oov args

Run:  python src/relabel_map.py                       # build + save + report
      python src/relabel_map.py --min-freq 5 --floor 0.55
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import WheelGrouper  # noqa: E402


class RepairedGrouper:
    """Drop-in replacement for WheelGrouper at scoring time: maps each word via the
    repaired dict (words absent from the dict are dropped, exactly like OOV)."""

    def __init__(self, repaired_map: dict, meta: dict = None):
        self.m = repaired_map
        self.meta = meta or {}

    def to_group_set(self, words, *args, **kwargs):
        return {self.m[w] for w in words if w in self.m}


def load_repaired(path: str) -> RepairedGrouper:
    d = json.load(open(path, encoding="utf-8"))
    return RepairedGrouper(d["map"], d.get("meta", {}))


def word_freq(manifest: str) -> Counter:
    wf = Counter()
    for line in open(manifest, encoding="utf-8"):
        line = line.strip()
        if line:
            for w in set(json.loads(line).get("label_words", [])):
                wf[w] += 1
    return wf


def build_repaired_map(vocab, emb, wf, grouper, min_freq=5, floor=0.55):
    """Returns (repaired_map, stats). repaired_map = wheel case2 map + recovered
    frequent dropped words routed to nearest canonical centroid (>= floor)."""
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    idx = {w: i for i, w in enumerate(vocab)}

    # base wheel case2 mapping
    w2c = {}
    for w in vocab:
        gs = grouper.to_group_set([w], "case2", "drop")
        if gs:
            w2c[w] = next(iter(gs))
    canon = sorted(set(w2c.values()))

    # canonical centroids = mean BGE embedding of each canonical's member words
    cmem = defaultdict(list)
    for w, c in w2c.items():
        cmem[c].append(w)
    Cemb = np.stack([E[[idx[w] for w in cmem[c]]].mean(0) for c in canon])
    Cemb = Cemb / (np.linalg.norm(Cemb, axis=1, keepdims=True) + 1e-8)

    dropped = [w for w in vocab if w not in w2c]
    recovered, recovered_inst, routes = {}, 0, []
    cand = [w for w in dropped if wf.get(w, 0) >= min_freq]
    if cand:
        sims = E[[idx[w] for w in cand]] @ Cemb.T
        for w, s in zip(cand, sims):
            j = int(s.argmax())
            if s[j] >= floor:
                recovered[w] = canon[j]
                recovered_inst += wf.get(w, 0)
                routes.append((w, canon[j], float(s[j]), wf.get(w, 0)))

    repaired = dict(w2c); repaired.update(recovered)
    stats = {"n_vocab": len(vocab), "n_canon": len(canon),
             "n_wheel_mapped": len(w2c), "n_recovered_words": len(recovered),
             "n_recovered_instances": recovered_inst,
             "n_still_dropped": len(vocab) - len(repaired),
             "min_freq": min_freq, "floor": floor,
             "routes": sorted(routes, key=lambda r: -r[3])}
    return repaired, stats


def apply_curation(amap, vocab_set, new_canonicals, drop):
    """Override the auto map with human-reviewed decisions: remove DROP words, and
    force NEW_CANONICALS members to their new canonical (pulled from vocab even if
    the wheel had dropped them). Returns (curated_map, added_by_canonical)."""
    m = {w: c for w, c in amap.items() if w not in drop}
    added = {}
    for canon, words in new_canonicals.items():
        for w in words:
            if w in vocab_set:
                m[w] = canon
                added.setdefault(canon, []).append(w)
    return m, added


def support_counts(manifest, word2grp, learn=20):
    sup = Counter()
    for line in open(manifest, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        ws = set(json.loads(line).get("label_words", []))
        for grp in {word2grp[w] for w in ws if w in word2grp}:
            sup[grp] += 1
    learnable = sum(v >= learn for v in sup.values())
    return sup, learnable


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the repaired emotion label map")
    ap.add_argument("--manifest", default=os.path.expanduser("~/SRIP/data/manifests/train.jsonl"))
    ap.add_argument("--vocab-emb", default=os.path.expanduser("~/SRIP/data/cache/vocab_emb.pt"))
    ap.add_argument("--wheel", default=os.path.expanduser(
        "~/SRIP/third_party/AffectGPT/OV-MER/emotion_wheel"))
    ap.add_argument("--out", default=os.path.expanduser("~/SRIP/data/cache/repaired_label_map.json"))
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--floor", type=float, default=0.55)
    ap.add_argument("--no-curate", action="store_true",
                    help="skip src/label_curation.py overrides (raw auto map only)")
    args = ap.parse_args()

    grouper = WheelGrouper(args.wheel)
    ve = torch.load(args.vocab_emb, weights_only=False)
    vocab = list(ve["vocab"]); emb = ve["word_emb"].float().numpy()
    wf = word_freq(args.manifest)

    repaired, stats = build_repaired_map(vocab, emb, wf, grouper,
                                         args.min_freq, args.floor)

    # apply human curation (new canonicals + drop non-emotions)
    added = {}
    if not args.no_curate:
        from label_curation import NEW_CANONICALS, DROP
        repaired, added = apply_curation(repaired, set(vocab), NEW_CANONICALS, DROP)
        stats["new_canonicals"] = {k: len(v) for k, v in added.items()}
        stats["n_dropped_by_curation"] = len(DROP)

    # before/after learnable counts
    w2c = {w: c for w, c in repaired.items()
           if grouper.to_group_set([w], "case2", "drop")}  # the wheel-mapped subset
    _, learn_before = support_counts(args.manifest, w2c)
    _, learn_after = support_counts(args.manifest, repaired)

    n_canon_final = len(set(repaired.values()))
    print(f"\n=== REPAIRED + CURATED LABEL MAP (min_freq={args.min_freq}, floor={args.floor}"
          f"{', NO curation' if args.no_curate else ''}) ===")
    print(f"  vocab={stats['n_vocab']}  wheel canonicals={stats['n_canon']}  "
          f"-> final canonicals={n_canon_final}")
    print(f"  wheel-mapped words : {stats['n_wheel_mapped']}")
    print(f"  auto-recovered     : {stats['n_recovered_words']} words")
    if added:
        for c, ws in added.items():
            present = [w for w in ws]  # already filtered to vocab
            print(f"  NEW canonical [{c}]: {len(present)} words -> "
                  f"+{sum(wf.get(w,0) for w in present)} clips  ({', '.join(present[:8])}…)")
        print(f"  dropped by curation: {stats.get('n_dropped_by_curation',0)} non-emotion words")
    print(f"  learnable canonicals (>=20 clips): {learn_before} -> {learn_after}")
    print(f"\n  top 20 auto routings (word -> canonical, cos, freq) "
          f"[curation may override]:")
    for w, c, s, f in stats["routes"][:20]:
        final = repaired.get(w, "DROPPED")
        flag = "" if final == c else f"  => {final}"
        print(f"     {f:5d}  {w:22s} -> {c:18s} ({s:.2f}){flag}")

    out = {"map": repaired, "meta": {k: v for k, v in stats.items() if k != "routes"},
           "routes": stats["routes"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
