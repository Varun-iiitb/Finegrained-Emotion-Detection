"""OV-MER set-based emotion metric: the emotion-wheel-grouped set metric F_s.

THE ONLY model-selection metric for this project (see CLAUDE.md section 6).
This is a faithful, standalone port of the grouping + metric in Lian et al.,
"OV-MER: Towards Open-Vocabulary Multimodal Emotion Recognition" (ICML 2025),
specifically AffectGPT's `my_affectgpt/evaluation/wheel.py`
(`func_backward_case{1,2,3}`, `calculate_openset_overlap_rate`). No dependency
on the AffectGPT package — only pandas (to read the emotion_wheel files).

Per sample, every predicted / gold emotion word is mapped to a GROUP ID and the
two sides deduplicated to sets (gold Y, pred P). Then:
    precision_s = |Y∩P|/|P|,   recall_s = |Y∩P|/|Y|     (per sample)
    - a sample with empty Y is SKIPPED;
    - a sample with empty P (after mapping) contributes precision=recall=0.
AGGREGATION is the CORPUS harmonic mean (NOT the mean of per-sample f):
    P = mean_samples(precision_s),  R = mean_samples(recall_s),
    F_s = 2·P·R/(P+R)   (0 if P+R==0).
This is why HM(92.2, 51.1) = 65.75 reproduces the ~65.7 one-hot OV-MERD anchor.

Grouping levels (`metric=` argument):
    case1                 word-form normalization only            (Level 1)
    case2                 + synonym -> canonical wheel label       (Level 2) ***
    case3_wheelN_levelL   + emotion-wheel cluster                  (Level 3)
We SELECT on case2 (Level 2). case3 is logged (mean over the 5 wheels).

OOV handling (`oov=`):
    'drop'       remove out-of-vocab words from pred AND gold (paper-faithful;
                 this is what we SELECT on).
    'singleton'  keep an out-of-vocab word as its own group (stricter
                 open-vocab eval; logged as a diagnostic only).

Public API:
    score(pred_lists, gold_lists, wheel_path) -> {precision_s, recall_s, f_s}
        (Level-2, drop  ==  the official selection metric)
    score_report(pred_lists, gold_lists, wheel_path) -> full breakdown dict
    WheelGrouper(wheel_path)  -> reusable loader (build mappings once)
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Set-math core  (no heavy deps; fully unit-testable via --selftest)
# ---------------------------------------------------------------------------

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def score_group_sets(
    gold_sets: Sequence[Set[str]],
    pred_sets: Sequence[Set[str]],
) -> Dict[str, float]:
    """Corpus-level (precision_s, recall_s, f_s) over already-grouped SETS.

    Mirrors AffectGPT `calculate_openset_overlap_rate` + the f-score combine:
    skip empty-gold samples; empty-pred samples score 0/0; average precision and
    recall over samples; F_s = harmonic mean of those two averages.
    """
    if len(gold_sets) != len(pred_sets):
        raise ValueError(
            f"length mismatch: {len(gold_sets)} gold vs {len(pred_sets)} pred")
    precisions: List[float] = []
    recalls: List[float] = []
    for g, p in zip(gold_sets, pred_sets):
        g, p = set(g), set(p)
        if not g:
            continue  # empty gold -> sample skipped (paper)
        if not p:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        inter = len(g & p)
        precisions.append(inter / len(p))
        recalls.append(inter / len(g))
    P, R = _mean(precisions), _mean(recalls)
    F = 2.0 * P * R / (P + R) if (P + R) > 0 else 0.0
    return {"precision_s": P, "recall_s": R, "f_s": F, "n": len(precisions)}


# ---------------------------------------------------------------------------
# Text helpers (string_to_list ported verbatim from AffectGPT functions.py)
# ---------------------------------------------------------------------------

def string_to_list(s) -> List[str]:
    """Parse a synonym/format/openset field into a list of strings.

    Verbatim behaviour of AffectGPT `string_to_list`: pass lists through; treat
    NaN/'' as empty; else strip a single pair of surrounding brackets and split
    on quotes and commas, dropping blanks. Multi-word items keep their spaces.
    """
    if isinstance(s, list):
        return s
    if s is None:
        return []
    try:
        import pandas as pd  # local: keep set-math core import-light
        if not isinstance(s, str) and pd.isna(s):
            return []
    except ImportError:
        pass
    if s == "":
        return []
    s = str(s)
    if s and s[0] == "[":
        s = s[1:]
    if s and s[-1] == "]":
        s = s[:-1]
    return [item.strip() for item in re.split(r"['\",]", s)
            if item.strip() not in ["", ","]]


def _norm(word: str) -> str:
    """Lowercase + strip, matching the `item.lower().strip()` used everywhere."""
    return str(word).lower().strip()


def _merge_map(m1: Dict[str, list], m2: Dict[str, list]) -> Dict[str, list]:
    """Port of AffectGPT `func_merge_map` (set-union list values across runs)."""
    out: Dict[str, list] = {}
    for k in set(m1) | set(m2):
        if k in m1 and k in m2:
            out[k] = list(set(m1[k] + m2[k]))
        elif k in m2:
            out[k] = list(m2[k])
        else:
            out[k] = list(m1[k])
    return out


# ---------------------------------------------------------------------------
# Emotion-wheel grouping (ported from AffectGPT wheel.py)
# ---------------------------------------------------------------------------

class WheelGrouper:
    """Loads the emotion_wheel mappings ONCE and maps emotion-word lists to
    group-id sets at a chosen level (case1/2/3) and OOV policy.

    `wheel_path` is the emotion_wheel directory (or its format.csv; the
    directory is inferred). case3 wheel clusters are loaded lazily and cached.
    """

    WHEELS = ["wheel1", "wheel2", "wheel3", "wheel4", "wheel5"]

    def __init__(self, wheel_path: str):
        self.wheel_dir = (os.path.dirname(wheel_path)
                          if os.path.isfile(wheel_path) else wheel_path)
        self.format_mapping = self._build_format_mapping()   # surface -> [base]
        self.raw_mapping = self._build_raw_mapping()         # base -> [canonical]
        self._wheel_cache: Dict[Tuple[str, str], Dict[str, str]] = {}
        print(f"[wheel] format_mapping: {len(self.format_mapping)} surface forms; "
              f"raw_mapping: {len(self.raw_mapping)} words")

    # ---- file loaders -----------------------------------------------------
    def _build_format_mapping(self) -> Dict[str, List[str]]:
        """format.csv -> surface form -> [base names]  (AffectGPT read_format2raws)."""
        import pandas as pd
        path = os.path.join(self.wheel_dir, "format.csv")
        df = pd.read_csv(path)
        fmap: Dict[str, List[str]] = {}
        for _, row in df.iterrows():
            raw = _norm(row["name"])
            for fitem in (_norm(x) for x in string_to_list(row["format"])):
                fmap.setdefault(fitem, []).append(raw)
            fmap.setdefault(raw, []).append(raw)  # self-map
        return fmap

    def _build_raw_mapping(self) -> Dict[str, List[str]]:
        """synonym.xlsx (8 runs merged) -> word -> [canonical wheel labels].

        Port of `read_candidate_synonym_merge`: each run self-maps the base word
        and maps every synonym -> base; runs merged by set-union.
        """
        import pandas as pd
        path = os.path.join(self.wheel_dir, "synonym.xlsx")
        df = pd.read_excel(path, dtype=str)
        merged: Dict[str, List[str]] = {}
        for run in range(1, 9):
            wcol, scol = f"word_run{run}", f"synonym_run{run}"
            if wcol not in df.columns or scol not in df.columns:
                continue
            onerun: Dict[str, List[str]] = {}
            for _, row in df.iterrows():
                raw = _norm(row[wcol])
                if not raw or raw == "nan":
                    continue
                onerun.setdefault(raw, []).append(raw)
                for syn in (_norm(x) for x in string_to_list(row[scol])):
                    if syn:
                        onerun.setdefault(syn, []).append(raw)
            merged = _merge_map(merged, onerun)
        return merged

    def _read_wheel_to_map(self, xlsx_path: str) -> Dict[str, Dict[str, List[str]]]:
        """wheelN.xlsx (level1/level2/level3, forward-filled) -> nested dict."""
        import pandas as pd
        df = pd.read_excel(xlsx_path)
        store: Dict[str, Dict[str, List[str]]] = {}
        l1 = l2 = l3 = ""
        for _, row in df.iterrows():
            if not pd.isna(row["level1"]):
                l1 = row["level1"]
            if not pd.isna(row["level2"]):
                l2 = row["level2"]
            if not pd.isna(row["level3"]):
                l3 = row["level3"]
            l1, l2, l3 = l1.lower().strip(), l2.lower().strip(), l3.lower().strip()
            store.setdefault(l1, {}).setdefault(l2, []).append(l3)
        return store

    def _get_wheel_cluster(self, wheel: str, level: str) -> Dict[str, str]:
        """canonical label -> wheel cluster center (AffectGPT func_get_wheel_cluster)."""
        key = (wheel, level)
        if key in self._wheel_cache:
            return self._wheel_cache[key]
        store = self._read_wheel_to_map(
            os.path.join(self.wheel_dir, f"{wheel}.xlsx"))
        wmap: Dict[str, str] = {}
        if level == "level1":
            for l1 in store:
                wmap[l1] = l1
                for l2 in store[l1]:
                    wmap[l2] = l1
                    for l3 in store[l1][l2]:
                        wmap[l3] = l1
        elif level == "level2":
            for l1 in store:
                wmap[l1] = sorted(store[l1])[0]
                for l2 in store[l1]:
                    wmap[l2] = l2
                    for l3 in store[l1][l2]:
                        wmap[l3] = l2
        self._wheel_cache[key] = wmap
        return wmap

    # ---- backward (word -> group id) -------------------------------------
    def _backward(self, label: str, metric: str,
                  wmap: Optional[Dict[str, str]]) -> str:
        """Return the group id for one word, or '' if out-of-vocab."""
        fmap = self.format_mapping
        if label not in fmap:
            return ""
        if metric.startswith("case1"):
            return sorted(fmap[label])[0]
        if metric.startswith("case2"):
            s1 = sorted(fmap[label])[0]
            cans = self.raw_mapping.get(s1)
            return sorted(cans)[0] if cans else ""
        if metric.startswith("case3"):
            cands = [r for f in fmap[label] for r in self.raw_mapping.get(f, [])]
            for c in sorted(cands):
                if wmap and c in wmap:
                    return wmap[c]
            return ""
        raise ValueError(f"unknown metric {metric!r}")

    def to_group_set(self, words: Sequence[str], metric: str = "case2",
                     oov: str = "drop",
                     wmap: Optional[Dict[str, str]] = None) -> Set[str]:
        """Map one sample's emotion-word list to a SET of group ids.

        oov='drop' removes out-of-vocab words; oov='singleton' keeps them as
        their own group (the normalized word).
        """
        out: Set[str] = set()
        for w in words:
            key = _norm(w)
            if not key:
                continue
            gid = self._backward(key, metric, wmap)
            if gid:
                out.add(gid)
            elif oov == "singleton":
                out.add(key)
        return out

    def score(self, pred_lists: Sequence[Sequence[str]],
              gold_lists: Sequence[Sequence[str]],
              metric: str = "case2", oov: str = "drop") -> Dict[str, float]:
        """Set metric for case1/case2 (single grouping). For case3 use
        `score_case3_mean` to average over the 5 wheels."""
        wmap = None
        if metric.startswith("case3"):
            _, wheel, level = metric.split("_")
            wmap = self._get_wheel_cluster(wheel, level)
        gold = [self.to_group_set(g, metric, oov, wmap) for g in gold_lists]
        pred = [self.to_group_set(p, metric, oov, wmap) for p in pred_lists]
        return score_group_sets(gold, pred)

    def score_case3_mean(self, pred_lists, gold_lists,
                         level: str = "level1", oov: str = "drop") -> Dict[str, float]:
        """case3 reported as the mean of (P, R, F) over the 5 emotion wheels
        (AffectGPT `wheel_metric_calculation` averages [f, p, r] across wheels)."""
        ps, rs, fs = [], [], []
        for w in self.WHEELS:
            r = self.score(pred_lists, gold_lists, metric=f"case3_{w}_{level}", oov=oov)
            ps.append(r["precision_s"]); rs.append(r["recall_s"]); fs.append(r["f_s"])
        return {"precision_s": _mean(ps), "recall_s": _mean(rs), "f_s": _mean(fs),
                "n_wheels": len(self.WHEELS)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score(pred_lists: Sequence[Sequence[str]],
          gold_lists: Sequence[Sequence[str]],
          wheel_path: str) -> Dict[str, float]:
    """THE official selection metric: Level-2 (case2) synonym grouping, drop OOV.
    Returns {"precision_s", "recall_s", "f_s"} (means / corpus-HM over samples).
    """
    g = WheelGrouper(wheel_path)
    r = g.score(pred_lists, gold_lists, metric="case2", oov="drop")
    return {k: r[k] for k in ("precision_s", "recall_s", "f_s")}


def score_report(pred_lists: Sequence[Sequence[str]],
                 gold_lists: Sequence[Sequence[str]],
                 wheel_path: str) -> Dict[str, Dict[str, float]]:
    """Full breakdown: Level-2 (case2) and Level-3 (case3 mean over wheels),
    each under drop (primary) and singleton (diagnostic). `select` is the number
    the project selects on (case2 / drop)."""
    g = WheelGrouper(wheel_path)
    rep = {
        "case2_drop": g.score(pred_lists, gold_lists, "case2", "drop"),
        "case2_singleton": g.score(pred_lists, gold_lists, "case2", "singleton"),
        "case3_drop": g.score_case3_mean(pred_lists, gold_lists, "level1", "drop"),
        "case3_singleton": g.score_case3_mean(pred_lists, gold_lists, "level1", "singleton"),
    }
    rep["select"] = {"metric": "case2", "oov": "drop", **rep["case2_drop"]}
    return rep


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------

def _selftest_setmath() -> None:
    """Check 1: corpus set-math on raw group-id sets (no wheel files needed)."""
    r = score_group_sets([{"a", "b"}], [{"a", "b"}])
    assert abs(r["f_s"] - 1.0) < 1e-9, r
    assert abs(r["precision_s"] - 1.0) < 1e-9 and abs(r["recall_s"] - 1.0) < 1e-9, r

    r = score_group_sets([{"a"}], [{"b"}])           # disjoint
    assert (r["precision_s"], r["recall_s"], r["f_s"]) == (0.0, 0.0, 0.0), r

    r = score_group_sets([{"a", "b"}], [{"a"}])      # 1-of-2 gold
    assert abs(r["precision_s"] - 1.0) < 1e-9, r
    assert abs(r["recall_s"] - 0.5) < 1e-9, r
    assert abs(r["f_s"] - (2.0 / 3.0)) < 1e-9, r

    r = score_group_sets([{"a"}], [set()])           # empty pred -> 0
    assert (r["precision_s"], r["recall_s"], r["f_s"]) == (0.0, 0.0, 0.0), r

    r = score_group_sets([set()], [{"a"}])           # empty gold -> skipped
    assert r["n"] == 0 and r["f_s"] == 0.0, r

    r = score_group_sets([{"a"}, {"a"}], [{"a"}, {"b"}])  # corpus over 2 samples
    assert abs(r["precision_s"] - 0.5) < 1e-9 and abs(r["f_s"] - 0.5) < 1e-9, r

    print("[selftest] set-math unit tests PASSED")


def _selftest_synonym(wheel_path: str) -> None:
    """Check 2: 'joyful' (pred) vs 'happy' (gold) must match at Level-2 (case2)."""
    g = WheelGrouper(wheel_path)
    gj = g.to_group_set(["joyful"], "case2")
    gh = g.to_group_set(["happy"], "case2")
    print(f"[selftest] case2 group('joyful')={gj}  group('happy')={gh}")
    r = g.score([["joyful"]], [["happy"]], metric="case2", oov="drop")
    assert r["f_s"] > 0.999, (
        f"synonym test FAILED: joyful vs happy did not match -> {r}. "
        f"Expected both to map to the same canonical wheel label at case2.")
    print(f"[selftest] synonym test PASSED: {r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="OV-MER set metric F_s")
    ap.add_argument("--selftest", action="store_true",
                    help="set-math unit tests (no files needed)")
    ap.add_argument("--synonym-test", action="store_true",
                    help="joyful/happy synonym test (needs --wheel)")
    ap.add_argument("--wheel", type=str, default=None,
                    help="emotion_wheel dir (or its format.csv)")
    args = ap.parse_args()

    ran = False
    if args.selftest:
        _selftest_setmath(); ran = True
    if args.synonym_test:
        if not args.wheel:
            print("[error] --synonym-test requires --wheel"); return 2
        _selftest_synonym(args.wheel); ran = True
    if not ran:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
