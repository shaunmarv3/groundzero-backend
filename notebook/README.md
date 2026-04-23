# GroundZero — Training Notebooks

> **Colab only.** These notebooks are NOT deployed. Run them on Google Colab free tier (T4 GPU).
> Run them **in order** — each notebook depends on the previous one's outputs.

| #   | Notebook                        | Purpose                                                                                       |
| --- | ------------------------------- | --------------------------------------------------------------------------------------------- |
| 01  | `01_data_preparation.ipynb`     | Download QVHighlights / Charades-STA / DiDeMo, extract frames, build frame index              |
| 02  | `02_baseline_measurement.ipynb` | Measure SigLIP 2 zero-shot baseline on QVHighlights val — **run first, record floor numbers** |
| 03  | `03_train_groundzero.ipynb`     | Full training loop: all custom modules, W&B logging, checkpoint saving                        |
| 04  | `04_ablation_study.ipynb`       | Train ablation variants (no temporal conv / no cross-attn / no contrastive loss)              |
| 05  | `05_evaluation.ipynb`           | Run full eval suite (R@K, mAP, latency, calibration, attention heatmaps)                      |
| 06  | `06_export_to_hub.ipynb`        | Push trained weights + LoRA adapters to HuggingFace Hub                                       |

## Requirements (Colab)

```
!pip install torch transformers peft einops wandb huggingface-hub datasets
!pip install ffmpeg-python scikit-learn matplotlib
```

## Backbone

**SigLIP 2 So400m** — `google/siglip2-so400m-patch14-384`  
400M params, 1152-d embeddings, LoRA on last 4 of 27 transformer blocks.
