"""
batch_encoding.py — turn a cached batch into model inputs (shared by train.py + evaluate.py).

Before this file, train.py and evaluate.py each had their own copy of
encode_batch_cached ("mirror of train.py's ...") — two copies that had to be kept
identical by hand. Both now import this one.

TextCache  — the pre-encoded queries from scripts/precompute_text_embeddings.py
             (text_pooled*.pt, and text_tokens*.pt for word-level). With it,
             training/eval load NO SigLIP at all.
encode_cached_batch — pads the per-video frame embeddings, builds the fractional
             timestamps + the real-frame mask, and fetches the query either from
             the TextCache or (no cache) the live frozen text tower.
"""

from __future__ import annotations

from pathlib import Path

import torch

DATASET_REPO = "shaunmarvell/qvhighlights-1fps"


class TextCache:
    """
    Pre-encoded SigLIP 2 queries, looked up by the ORIGINAL annotation string.

    Args:
        data_dir:   where the .pt files live (downloaded from HF if missing)
        lowercase:  use the *_lower.pt variant (queries were encoded as q.lower())
        word_level: also load the per-token file (~1.3 GB in RAM) + its mask
    """

    def __init__(self, data_dir: str | Path, lowercase: bool = False, word_level: bool = False,
                 repo_id: str = DATASET_REPO, token: str | None = None):
        sfx = "_lower" if lowercase else ""
        data_dir = Path(data_dir)
        self.word_level = word_level

        pooled = torch.load(self._fetch(data_dir, f"text_pooled{sfx}.pt", repo_id, token),
                            map_location="cpu")
        self.queries = pooled["queries"]
        self.index   = {q: i for i, q in enumerate(self.queries)}
        self.pooled  = pooled["pooled"]                      # (Q, D) fp16
        self.tokens = self.mask = None

        if word_level:
            tok = torch.load(self._fetch(data_dir, f"text_tokens{sfx}.pt", repo_id, token),
                             map_location="cpu")
            assert tok["queries"] == self.queries, "pooled/tokens files list queries in different order"
            self.tokens = tok["tokens"]                      # (Q, L, D) fp16
            self.mask   = tok["mask"]                        # (Q, L) bool

        print(f"TextCache: {len(self.queries)} queries | lowercase={lowercase} | "
              f"word_level={word_level}"
              + (f" | tokens {tuple(self.tokens.shape)}" if word_level else ""))

    @staticmethod
    def _fetch(data_dir: Path, name: str, repo_id: str, token: str | None) -> Path:
        path = data_dir / name
        if not path.exists():
            from huggingface_hub import hf_hub_download
            print(f"  Downloading {name} from {repo_id} ...")
            hf_hub_download(repo_id, name, repo_type="dataset", local_dir=str(data_dir), token=token)
        return path

    def lookup(self, queries: list, device) -> tuple:
        """
        Returns (query, query_mask, query_pooled) for a list of query strings:
            pooled mode: query = (B, 1, D),  query_mask = None
            word mode:   query = (B, L, D),  query_mask = (B, L) bool
            query_pooled (B, 1, D) always — the contrastive loss needs one vector per query.
        """
        try:
            idx = torch.tensor([self.index[q] for q in queries], dtype=torch.long)
        except KeyError as e:
            raise KeyError(f"query not in text cache (re-run precompute_text_embeddings.py?): {e}")
        pooled = self.pooled[idx].to(device).float().unsqueeze(1)        # (B, 1, D)
        if not self.word_level:
            return pooled, None, pooled
        tokens = self.tokens[idx].to(device).float()                     # (B, L, D)
        mask   = self.mask[idx].to(device)                               # (B, L)
        return tokens, mask, pooled


def encode_cached_batch(model, batch, device, text_cache: TextCache | None = None,
                        lowercase: bool = False, d_model: int = 1152) -> tuple:
    """
    Cached-feature batch → model inputs. Call inside autocast (as before).

    Returns:
        frame_embs   : (B, N_max, d_model) float32, zero-padded
        timestamps   : (B, N_max)          fractional t/duration, padded 1.0
        frame_mask   : (B, N_max) bool     True = real frame
        query        : (B, 1, D) pooled, or (B, L, D) tokens (word-level)
        query_mask   : (B, L) bool or None
        query_pooled : (B, 1, D)           for the contrastive loss
    """
    B      = len(batch["frame_embs"])
    counts = [e.shape[0] for e in batch["frame_embs"]]
    N_max  = max(counts)

    frame_embs = torch.zeros(B, N_max, d_model, device=device)
    timestamps = torch.ones(B, N_max, device=device)   # 1.0 = end-of-video pad position
    frame_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)

    for i in range(B):
        emb = batch["frame_embs"][i].to(device).float()   # (N_i, d_model), fp16→fp32
        n   = emb.shape[0]
        frame_embs[i, :n] = emb
        frame_mask[i, :n] = True
        dur_i = batch["duration"][i].item()
        frac  = torch.tensor(
            [t / dur_i for t in batch["timestamps"][i]],
            dtype=torch.float32, device=device,
        )
        timestamps[i, :n] = frac

    if text_cache is not None:
        query, query_mask, query_pooled = text_cache.lookup(batch["query"], device)
    else:
        texts = [q.lower() for q in batch["query"]] if lowercase else batch["query"]
        query = model.text_encoder.encode_queries(texts).float().unsqueeze(1)   # (B,1,D)
        query_mask, query_pooled = None, query

    return frame_embs, timestamps, frame_mask, query, query_mask, query_pooled
