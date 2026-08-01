"""Reconstruct per-epoch training history from OLD stdout logs.

train.py and train_hybrid.py only PRINTED their per-epoch metrics — they never
saved history. This parser turns those stdout lines back into the same
history.json schema that train_machine_cv.py writes, so the earlier runs can be
plotted by plot_training.py WITHOUT re-running them on the GPU.

Recognised line formats (emitted by the existing scripts):
  train.py      :  "  seed0 ep00 loss=1.234 val_Fs=27.30 thr=0.150"
                   "  seed0: OV-MERD F_s=28.30  (val_Fs=27.30 thr=0.150)"
                   -> each seed becomes a "fold" so curves overlay.
  train_hybrid  :  "  [pretrain] ep00 loss=2.100 human zero-shot F_s(oracle-thr)=27.10"
                   "  [finetune] fold0: human-test F_s=34.20 (val=33.10 thr=0.200)"
                   -> pretrain epochs become the curve (val_fs = human zero-shot);
                      finetune fold finals go into result.json (no per-epoch line
                      exists in the log).

Run:  python src/parse_logs.py --log ~/SRIP/logs/gate_zh.log --out ~/SRIP/runs/gate_zh
      python src/plot_training.py --run ~/SRIP/runs/gate_zh
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List

RE_TRAIN_EP = re.compile(
    r"seed(\d+)\s+ep(\d+)\s+loss=([\d.]+)\s+val_Fs=([\d.]+)\s+thr=([\d.]+)")
RE_TRAIN_FINAL = re.compile(
    r"seed(\d+):\s+OV-MERD F_s=([\d.]+)\s+\(val_Fs=([\d.]+)\s+thr=([\d.]+)\)")
RE_HYB_PRE = re.compile(
    r"\[pretrain\]\s+ep(\d+)\s+loss=([\d.]+)\s+human zero-shot F_s\(oracle-thr\)=([\d.]+)")
RE_HYB_FT = re.compile(
    r"\[finetune\]\s+fold(\d+):\s+human-test F_s=([\d.]+)\s+\(val=([\d.]+)\s+thr=([\d.]+)\)")


def parse(text: str) -> Dict:
    history: List[Dict] = []
    fold_results: List[Dict] = []
    kinds = set()

    # Namespace the "fold" series by kind so a (rare) mixed log can't merge a
    # train.py seed curve with a hybrid pretrain curve into the same series.
    for m in RE_TRAIN_EP.finditer(text):
        seed, ep, loss, vfs, thr = m.groups()
        history.append({"fold": f"seed{seed}", "epoch": int(ep),
                        "train_loss": float(loss), "val_fs": float(vfs),
                        "val_thr": float(thr)})
        kinds.add("train")
    for m in RE_TRAIN_FINAL.finditer(text):
        seed, ovfs, vfs, thr = m.groups()
        fold_results.append({"fold": f"seed{seed}", "test_fs": float(ovfs),
                             "best_val_fs": float(vfs), "best_thr": float(thr)})

    for m in RE_HYB_PRE.finditer(text):
        ep, loss, zs = m.groups()
        history.append({"fold": "pretrain", "epoch": int(ep), "train_loss": float(loss),
                        "val_fs": float(zs), "val_thr": None})
        kinds.add("hybrid_pretrain")
    for m in RE_HYB_FT.finditer(text):
        fold, fs, val, thr = m.groups()
        fold_results.append({"fold": f"ft_fold{fold}", "test_fs": float(fs),
                             "best_val_fs": float(val), "best_thr": float(thr)})
        kinds.add("hybrid_finetune")

    # de-dup epochs per fold (keep last occurrence — re-runs append)
    seen, dedup = {}, []
    for r in history:
        seen[(r["fold"], r["epoch"])] = r
    for r in sorted(seen.values(), key=lambda x: (x["fold"], x["epoch"])):
        dedup.append(r)

    test_scores = [r["test_fs"] for r in fold_results]
    result = {"run_name": "(parsed)", "kinds": sorted(kinds),
              "folds": sorted(fold_results, key=lambda r: r["fold"]),
              "n_folds": len(fold_results)}
    if test_scores:
        result["test_fs_mean"] = round(sum(test_scores) / len(test_scores), 4)
        result["test_fs_std"] = round(
            (sum((s - result["test_fs_mean"]) ** 2 for s in test_scores)
             / len(test_scores)) ** 0.5, 4)
    return {"history": dedup, "result": result}


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconstruct history.json from old stdout logs")
    ap.add_argument("--log", required=True, help="stdout log file to parse")
    ap.add_argument("--out", required=True, help="output run dir (history.json+result.json)")
    ap.add_argument("--name", default=None, help="run_name to stamp in result.json")
    args = ap.parse_args()

    with open(args.log, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    parsed = parse(text)
    if not parsed["history"]:
        print("WARNING: no recognised per-epoch lines found. Is this a train.py / "
              "train_hybrid.py log? (linear_probe.py prints no per-epoch curve.)")
    if args.name:
        parsed["result"]["run_name"] = args.name

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(parsed["history"], f, indent=2)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(parsed["result"], f, indent=2)

    nfolds = len({r["fold"] for r in parsed["history"]})
    print(f"parsed {len(parsed['history'])} epoch-records across {nfolds} fold/seed(s) "
          f"({', '.join(parsed['result']['kinds']) or 'none'}) -> {args.out}")
    if parsed["result"].get("test_fs_mean") is not None:
        print(f"  final F_s = {parsed['result']['test_fs_mean']:.2f} "
              f"± {parsed['result']['test_fs_std']:.2f}")
    print(f"  plot: python src/plot_training.py --run {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
