"""Training-curve plots for DG-DQA runs (headless / Agg backend).

Reads per-epoch history JSON (a list of records with at least `epoch`,
`train_loss`, `val_fs`, and optionally `fold`) and renders:
  - a loss-vs-epoch curve,
  - a val-F_s-vs-epoch curve (the selection metric), with the best epoch marked,
  - for CV runs (multiple folds), one line per fold + the per-epoch mean.

History comes from two sources, both producing the SAME record schema:
  - new runs: <run>/history.json written by train_machine_cv.py
  - old runs: reconstructed from stdout logs by parse_logs.py

Run (one run dir):   python src/plot_training.py --run ~/SRIP/runs/machine_cv_zh_canonical
Run (a history file): python src/plot_training.py --history path/to/history.json --out plot.png
Run (compare runs):  python src/plot_training.py --compare ~/SRIP/runs/A ~/SRIP/runs/B --out cmp.png
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")                      # headless server: no display
import matplotlib.pyplot as plt            # noqa: E402


def load_history(path: str) -> List[Dict]:
    """Accept either a history.json file or a run directory containing one."""
    if os.path.isdir(path):
        path = os.path.join(path, "history.json")
    with open(path) as f:
        return json.load(f)


def load_result(run_dir: str) -> Optional[Dict]:
    p = os.path.join(run_dir, "result.json")
    return json.load(open(p)) if os.path.exists(p) else None


def by_fold(history: List[Dict]) -> Dict[int, List[Dict]]:
    folds = defaultdict(list)
    for r in history:
        folds[r.get("fold", 0)].append(r)
    for k in folds:
        folds[k].sort(key=lambda r: r["epoch"])
    return dict(sorted(folds.items()))


def _mean_curve(folds: Dict[int, List[Dict]], key: str):
    """Per-epoch mean across folds (epochs that all folds share)."""
    per_ep = defaultdict(list)
    for recs in folds.values():
        for r in recs:
            if r.get(key) is not None:
                per_ep[r["epoch"]].append(r[key])
    eps = sorted(per_ep)
    return eps, [sum(per_ep[e]) / len(per_ep[e]) for e in eps]


def plot_run(history: List[Dict], out: str, title: str,
             result: Optional[Dict] = None) -> None:
    folds = by_fold(history)
    multi = len(folds) > 1
    fig, (axL, axF) = plt.subplots(1, 2, figsize=(13, 5))

    for k, recs in folds.items():
        eps = [r["epoch"] for r in recs]
        lab = f"fold{k}" if multi else "run"
        if any(r.get("train_loss") is not None for r in recs):
            axL.plot(eps, [r.get("train_loss") for r in recs], marker=".",
                     alpha=0.5 if multi else 1.0, label=lab)
        axF.plot(eps, [r.get("val_fs") for r in recs], marker=".",
                 alpha=0.5 if multi else 1.0, label=lab)

    if multi:
        e, m = _mean_curve(folds, "train_loss")
        if m:
            axL.plot(e, m, color="black", lw=2.5, label="mean")
        e, m = _mean_curve(folds, "val_fs")
        axF.plot(e, m, color="black", lw=2.5, label="mean")

    # mark best val epoch(s) from result.json
    if result:
        for fr in result.get("folds", []):
            be = fr.get("best_epoch")
            if be is not None and be >= 0:
                axF.axvline(be, color="red", ls=":", alpha=0.4)
        mean = result.get("test_fs_mean"); std = result.get("test_fs_std", 0.0)
        if mean is not None:
            axF.axhline(mean, color="green", ls="--", alpha=0.7,
                        label=f"test F_s={mean:.1f}±{std:.1f}")

    axL.set_xlabel("epoch"); axL.set_ylabel("train loss"); axL.set_title("Training loss")
    axL.grid(alpha=0.3); axL.legend(fontsize=8)
    axF.set_xlabel("epoch"); axF.set_ylabel("val F_s (%)")
    axF.set_title("Validation F_s (selection metric)")
    axF.grid(alpha=0.3); axF.legend(fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"  saved -> {out}")


def plot_compare(run_dirs: List[str], out: str) -> None:
    """Overlay the per-epoch mean val-F_s curve of several runs + a final-F_s bar."""
    fig, (axF, axB) = plt.subplots(1, 2, figsize=(13, 5))
    names, finals, errs = [], [], []
    for rd in run_dirs:
        hist = load_history(rd)
        res = load_result(rd)
        name = (res or {}).get("run_name", os.path.basename(rd.rstrip("/")))
        eps, m = _mean_curve(by_fold(hist), "val_fs")
        axF.plot(eps, m, marker=".", label=name)
        names.append(name)
        finals.append((res or {}).get("test_fs_mean", m[-1] if m else 0.0))
        errs.append((res or {}).get("test_fs_std", 0.0))
    axF.set_xlabel("epoch"); axF.set_ylabel("val F_s (%)")
    axF.set_title("Validation F_s vs epoch"); axF.grid(alpha=0.3); axF.legend(fontsize=8)
    axB.bar(range(len(names)), finals, yerr=errs, capsize=4)
    axB.set_xticks(range(len(names))); axB.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axB.set_ylabel("test / final F_s (%)"); axB.set_title("Final F_s by run"); axB.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"  saved -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="DG-DQA training-curve plots")
    ap.add_argument("--run", help="a run dir (with history.json [+ result.json])")
    ap.add_argument("--history", help="a history.json path directly")
    ap.add_argument("--compare", nargs="+", help="several run dirs to overlay")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.compare:
        plot_compare(args.compare, args.out or "compare.png")
        return 0
    if args.run:
        hist = load_history(args.run)
        res = load_result(args.run)
        out = args.out or os.path.join(args.run, "plot.png")
        title = args.title or (res or {}).get("run_name", os.path.basename(args.run.rstrip("/")))
        plot_run(hist, out, title, res)
        return 0
    if args.history:
        hist = load_history(args.history)
        out = args.out or os.path.splitext(args.history)[0] + ".png"
        plot_run(hist, out, args.title or os.path.basename(args.history))
        return 0
    ap.error("pass one of --run / --history / --compare")


if __name__ == "__main__":
    raise SystemExit(main())
