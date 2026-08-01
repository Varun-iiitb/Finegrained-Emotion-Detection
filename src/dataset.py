"""Dataset / collate over the per-clip feature cache (DG-DQA, see CLAUDE.md §5).

Lazily loads each clip's cached sequences and collates a batch into padded
tensors + boolean pad masks (True = pad) per modality. A missing/masked modality
is already a length-1 zero token in the cache, so it just pads like any short
sequence. `text_branch` selects the Chinese ('zh') or English ('en') subtitle
stream as the text modality.
"""

import glob
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class CacheDataset(Dataset):
    """Per-clip cache reader. names defaults to every .pt under cache/<split>/."""

    def __init__(self, cache_dir: str, split: str, text_branch: str = "zh",
                 names: Optional[List[str]] = None):
        self.dir = os.path.join(cache_dir, split)
        self.text_branch = text_branch
        if names is None:
            names = [os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(os.path.join(self.dir, "*.pt"))]
        self.names = sorted(names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> Dict:
        name = self.names[i]
        r = torch.load(os.path.join(self.dir, name + ".pt"), weights_only=False)
        t = r["X_text_zh"] if self.text_branch == "zh" else r["X_text_en"]
        item = {"name": name, "t": t.float(),
                "a": r["X_audio"].float(), "v": r["X_visual"].float(),
                "label_words": r["label_words"]}
        if "desc_emb" in r:
            item["desc_emb"] = r["desc_emb"].float()
        return item


def collate(batch: List[Dict]) -> Dict:
    """Pad each modality's ragged sequences to the batch max; build pad masks."""
    out = {"name": [b["name"] for b in batch],
           "label_words": [b["label_words"] for b in batch]}
    for m in ("t", "a", "v"):
        seqs = [b[m] for b in batch]
        t_max = max(s.size(0) for s in seqs)
        d = seqs[0].size(1)
        X = torch.zeros(len(batch), t_max, d)
        pad = torch.ones(len(batch), t_max, dtype=torch.bool)
        for i, s in enumerate(seqs):
            X[i, :s.size(0)] = s
            pad[i, :s.size(0)] = False
        out[m] = X
        out[m + "_pad"] = pad
    if "desc_emb" in batch[0]:
        out["desc_emb"] = torch.stack([b["desc_emb"] for b in batch])
    return out
