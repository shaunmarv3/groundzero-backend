"""
visual_encoder.py — SigLIP 2 So400m Visual Encoder with LoRA adapters.
Phase 3.2 (Chunk D).

Architecture:
  Backbone : SigLIP 2 So400m (google/siglip2-so400m-patch14-384)
             400M params, 1152-d embeddings, 27 transformer blocks
             Loaded in float16 to fit in GPU memory — stays frozen.
  Adapters : LoRA on last 4 blocks (layers 23-26), q_proj + v_proj only.
             get_peft_model automatically freezes the base model.
  Output   : (N, 1152) float16 tensor — one embedding per frame.
"""

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModel
from peft import LoraConfig, get_peft_model


class VisualEncoder(nn.Module):
    """
    SigLIP 2 So400m vision tower with LoRA adapters on the last 4 blocks.

    Frozen:     entire SigLIP 2 backbone (~400M params, float16)
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

        # Full SigLIP 2 model in float16 (saves ~800MB vs float32)
        full_model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)

        # Freeze text tower — visual_encoder.py only handles the vision side
        for param in full_model.text_model.parameters():
            param.requires_grad = False

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

        self.model = full_model.to(device)

    def encode_frames(self, frames: list) -> torch.Tensor:
        """
        Encode a list of PIL Images into frame embeddings.

        Args:
            frames : list of PIL.Image — any size, processor handles resize+normalise

        Returns:
            Tensor (N, 1152) — one 1152-d vector per frame
        """
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)  # (N, 1152)
        return features

    def count_params(self) -> dict:
        """Return trainable vs total parameter counts."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return {
            "trainable": trainable,
            "total": total,
            "frozen": total - trainable,
        }
