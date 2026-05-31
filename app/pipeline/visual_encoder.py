"""
visual_encoder.py — SigLIP 2 So400m Visual Encoder with LoRA adapters.
Phase 3.2 (Chunk D).

Architecture:
  Backbone : SigLIP 2 So400m (google/siglip2-so400m-patch14-384)
             ~0.9B params, 1152-d embeddings, 27 transformer blocks
             Loaded in float16 (frozen) so the backbone + activation graph fit a
             16 GB GPU. Compute already runs in fp16 under autocast, so numerics
             are unchanged; only the LoRA adapters are kept in fp32 (for stable
             optimisation) — see __init__.
  Adapters : LoRA on last 4 blocks (layers 23-26), q_proj + v_proj only.
             get_peft_model automatically freezes the base model.
  Output   : (N, 1152) float32 tensor — one embedding per frame.
"""

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModel
from peft import LoraConfig, get_peft_model


class VisualEncoder(nn.Module):
    """
    SigLIP 2 So400m vision tower with LoRA adapters on the last 4 blocks.

    Frozen:     entire SigLIP 2 backbone (~0.9B params, float32)
    Trainable:  LoRA A/B matrices on layers [23,24,25,26], q_proj + v_proj
    """

    def __init__(
        self,
        model_id: str = "google/siglip2-so400m-patch14-384",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_layers: list = [23, 24, 25, 26],
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # Processor: resizes images to 384×384 and normalises pixel values
        self.processor = AutoProcessor.from_pretrained(model_id)

        # fp16 frozen backbone: halves weight storage (~3.5 GB → ~1.75 GB) and the
        # checkpointed activation graph, which is what lets a 150-frame video fit a
        # 16 GB T4. Compute already runs in fp16 under autocast, so this changes
        # storage only, not numerics.
        full_model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)

        # Freeze text tower — visual_encoder.py only handles the vision side
        for param in full_model.text_model.parameters():
            param.requires_grad = False

        # Gradient checkpointing: recomputes activations during backward instead of
        # storing all 27 layers simultaneously. Cuts activation memory from ~7 GB to
        # ~500 MB for 150-frame videos — essential for training on T4 (16 GB VRAM).
        # Must be called on full_model (a PreTrainedModel) — the SiglipVisionTransformer
        # submodule doesn't expose this method. This also wires up _gradient_checkpointing_func,
        # which setting the bare `.gradient_checkpointing = True` flag does NOT do.
        # use_reentrant=False lets gradients reach the LoRA adapters even though the frozen
        # patch-embedding input has no grad — so no enable_input_require_grads() hack needed.
        full_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Don't checkpoint the frozen text tower — wasted recompute, and it would warn
        # "none of the inputs require grad" on every text forward.
        full_model.text_model.encoder.gradient_checkpointing = False

        # Apply LoRA to the last 4 transformer blocks of the vision tower.
        # get_peft_model automatically freezes all base model params inside vision_model.
        # Only the LoRA A/B matrices (initialised to near-zero) are trainable.
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            layers_to_transform=lora_layers,
            lora_dropout=0.1,
            bias="none",
        )
        full_model.vision_model = get_peft_model(full_model.vision_model, lora_config)

        # Keep the trainable LoRA matrices in fp32 while the frozen backbone stays fp16
        # (the QLoRA pattern). Autocast casts them to fp16 for the matmul, but the master
        # weights + AdamW states stay fp32, which avoids fp16 underflow during updates.
        for p in full_model.vision_model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        self.model = full_model.to(device)

    def encode_frames(self, frames: list, chunk_size: int = 8) -> torch.Tensor:
        """
        Encode a list of PIL Images into frame embeddings.

        Processes frames in chunks to avoid OOM — encoding all 150 frames at once
        creates 150×729 patches × 27 layers of activations which exceeds T4 VRAM.
        chunk_size=8 keeps peak activation memory ~200 MB per chunk.

        Args:
            frames     : list of PIL.Image
            chunk_size : frames processed at once (lower = less VRAM, more steps)

        Returns:
            Tensor (N, 1152) — one 1152-d vector per frame
        """
        all_features = []
        for i in range(0, len(frames), chunk_size):
            chunk = frames[i : i + chunk_size]
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            out = self.model.get_image_features(**inputs)
            # transformers >=4.50 returns a BaseModelOutputWithPooling here instead of a
            # bare tensor; pull the pooled (chunk, 1152) image embedding out either way.
            features = out if isinstance(out, torch.Tensor) else out.pooler_output
            all_features.append(features)
        return torch.cat(all_features, dim=0)  # (N, 1152)

    def count_params(self) -> dict:
        """Return trainable vs total parameter counts."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return {
            "trainable": trainable,
            "total": total,
            "frozen": total - trainable,
        }

#   GroundZeroModel
#   ├── VisualEncoder       ← visual_encoder.py  (this file)
#   ├── TextEncoder         ← text_encoder.py    (Chunk E)
#   ├── TemporalContext     ← temporal_context.py (Chunk C, done)
#   ├── CrossModalTransformer
#   └── SpanExtractionHead


#  After training, Phase 7 loading will look exactly like your LLM experience — two options:

#   Option A — keep LoRA separate (small checkpoint, 0.147M):
#   encoder = VisualEncoder()                          # empty LoRA slots
#   encoder.load_adapter("our_trained_lora_weights")  # fill slots

#   Option B — merge and drop LoRA (what you did with merge_and_unload()):
#   vision_model = encoder.model.vision_model.merge_and_unload()
#   # LoRA gone, weights baked in, faster inference

#   We'll do Option B in Phase 8. So yes — the final production model will have no LoRA visible at inference, same as your experience.

#   The only reason LoRA config lives in visual_encoder.py is because we're assembling a custom model, not using a Trainer that manages everything for us.


# ● Exactly.

#   HuggingFace Trainer approach (what you did):
#   Trainer owns everything — model loading, LoRA, optimizer, training loop, saving
#   You just configure it and call trainer.train()
#   LoRA config naturally lives in the training script because Trainer handles it

#   Pure PyTorch approach (what we're doing):
#   You own everything — you build the model, write the training loop yourself
#   LoRA config lives in the model file because YOU are the Trainer
#   Phase 5 training loop = what SFTTrainer was doing internally

#   That's the whole difference. SFTTrainer is just someone else's Phase 5 training loop written for you. When you use it, you hand it a model and it handles
#   the rest. When you write your own, you need the model to be fully defined before your training loop touches it — so LoRA config goes in the model file.

#   We're doing pure PyTorch because our model has custom components (TemporalContext, CrossModalTransformer, SpanExtractionHead) that HuggingFace Trainer
#   doesn't know how to handle. SFTTrainer is built for standard LLMs — give it a model and text data, it knows what to do. Our grounding model takes video
#   frames + text query → timestamps, which is a completely custom pipeline.