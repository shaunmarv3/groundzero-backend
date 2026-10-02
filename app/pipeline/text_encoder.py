"""
text_encoder.py — SigLIP 2 So400m Text Encoder (fully frozen).
Phase 3.3 (Chunk E).

Architecture:
  Backbone : SigLIP 2 So400m text tower — same model as visual_encoder.py.
             Fully frozen, no LoRA (text queries don't need video-specific adaptation).
  Output   : (1, 1152) float16 tensor for a single query.
             (N, 1152) float16 tensor for a batch of queries.

Note: TextEncoder and VisualEncoder both load the full SigLIP 2 model for
independent testing. In GroundZeroModel (Phase 3.7) they will share one
backbone instance to avoid loading 400M params twice.
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel


class TextEncoder(nn.Module):
    """
    SigLIP 2 So400m text tower — fully frozen.

    Trainable: 0 params (text queries need no video-specific adaptation)
    Frozen:    entire SigLIP 2 backbone

    Pass shared_model to reuse an already-loaded backbone (e.g. from VisualEncoder)
    so GroundZeroModel doesn't load SigLIP 2 twice into VRAM.
    """

    def __init__(
        self,
        model_id: str = "google/siglip2-so400m-patch14-384",
        device: str = "cuda",
        shared_model=None,
    ):
        super().__init__()
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_id)

        if shared_model is not None:
            # Reuse already-loaded backbone — saves ~1.1 GB VRAM in GroundZeroModel
            self.model = shared_model
        else:
            full_model = AutoModel.from_pretrained(model_id)
            for param in full_model.parameters():
                param.requires_grad = False
            self.model = full_model.to(device)

    def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a single text query.

        Args:
            query : query string, e.g. "person opens a door"

        Returns:
            Tensor (1, 1152)
        """
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self.device)

        out = self.model.get_text_features(**inputs)
        # transformers >=4.50 returns a BaseModelOutputWithPooling instead of a tensor
        features = out if isinstance(out, torch.Tensor) else out.pooler_output  # (1, 1152)
        return features

    def encode_queries(self, queries: list) -> torch.Tensor:
        """
        Encode a batch of text queries.

        Args:
            queries : list of query strings

        Returns:
            Tensor (N, 1152)
        """
        inputs = self.processor(
            text=queries,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self.device)

        out = self.model.get_text_features(**inputs)
        # transformers >=4.50 returns a BaseModelOutputWithPooling instead of a tensor
        features = out if isinstance(out, torch.Tensor) else out.pooler_output  # (N, 1152)
        return features

    def encode_query_tokens(self, query: str) -> tuple:
        """
        Per-token text states for word-level cross-attention (Fix 3). Same processor call
        as encode_query / scripts/precompute_text_embeddings.py.

        Returns:
            tokens : (1, L, 1152) — last_hidden_state of the text tower (L = 64 for SigLIP 2)
            mask   : (1, L) bool  — True = real token, False = padding
        """
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self.device)
        tokens = self.model.text_model(**inputs).last_hidden_state          # (1, L, 1152)
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"].bool()
        else:
            mask = inputs["input_ids"] != self.processor.tokenizer.pad_token_id
        return tokens, mask

    def count_params(self) -> dict:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return {"trainable": trainable, "total": total, "frozen": total - trainable}
