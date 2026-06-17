# GroundZero — Backend

PyTorch model + training + evaluation + **FastAPI inference server** for temporal video grounding:
a video + a natural-language query → predicted **start/end timestamps** of the described moment.

> This README covers the backend only. For the full design rationale see [`../project.md`](../project.md);
> for the chunk-by-chunk build log see [`../progress.md`](../progress.md); for a component-by-component
> walkthrough with tensor shapes see [`WORKFLOW.md`](WORKFLOW.md).

---

## What this is

A frozen **SigLIP 2 So400m** vision-language backbone (1152-d, 27 blocks) adapted with **LoRA**, plus
custom trainable modules and a span-extraction head:

```
video ─► frame extraction (1 fps) ─► SigLIP 2 (frozen + LoRA) ─┐
                                                               ├─► per-frame embeddings (N, 1152)
query ─► SigLIP 2 text tower (frozen) ─────────────────────────┘
                          │
        Temporal Context (dilated 1D conv, RF ≈ 60 s)
                          │
        Cross-Modal Transformer (4× cross→self attention)
                          │
        Span Extraction Head (per-frame start/end + confidence)
                          │
              decode best span ─► start_sec / end_sec
```

- **LoRA:** r=8, α=16 on `q_proj`/`v_proj` of layers 23–26 only → **0.147 M** adapter params.
- **Trainable totals:** ~154.9 M trainable / ~2426.9 M total.

---

## Real results (QVHighlights val, n = 1550)

Measured with `scripts/evaluate.py` on `best.pt` (epoch 182). **Trained and evaluated on QVHighlights only.**

| Metric | Result |
|---|---|
| R@1 IoU=0.5 | **0.5413** |
| R@1 IoU=0.7 | **0.3858** |
| R@5 IoU=0.5 | **0.7974** |
| Median center displacement | 10.0 s (mean 21.2 s) |
| Confidence ECE | 0.3955 — **uncalibrated, do not gate on it** |

**The confidence head is broken** (ECE 0.40, non-discriminative — it returned 0.012 on a verified
IoU-0.91 prediction). The pipeline therefore sets `found` from whether a real span was produced over
real frames, **never** from the confidence score. Confidence is reported raw, for transparency only.

**Not done** (see `../project.md` for the honest gap list): cross-dataset eval (Charades-STA /
ActivityNet / DiDeMo / TACoS), mAP, the ablation table, latency profiling, and the regression suite.

---

## Run it

Python 3.11, CUDA GPU recommended (~3 GB VRAM; fits a 4 GB RTX 3050, tight).

```bash
pip install -r requirements.txt
python scripts/download_model.py        # pulls best.pt (~590 MB) from HuggingFace → models/
uvicorn main:app --port 8000            # add --reload for active dev
```

Wait ~30–60 s for SigLIP 2 + `best.pt` to load, then confirm `http://localhost:8000/health` shows
`model_loaded: true`. Interactive docs: `http://localhost:8000/docs`.

> The orchestrator owns its own forward path (mirrors `scripts/evaluate.py` exactly: `eval()` mode,
> fp16 autocast compute, fp32 head inputs). It deliberately does **not** call `GroundZeroModel.predict()`,
> which lacks `eval()` and defaults to a 4 fps fine pass the model never trained on. Default is a single
> 1 fps pass (`use_coarse_to_fine = False`) — the correctness-safe path.

---

## API

| Endpoint | Body | Returns |
|---|---|---|
| `POST /api/predict` | multipart: `video` (file), `query` (str), optional `fps`, `coarse_to_fine` | `GroundingResult{start_sec, end_sec, confidence, found, low_confidence, duration_sec, query, fps, n_frames, took_ms, coarse}` |
| `POST /api/attention` | same multipart | `{timestamps, weights, start_sec, end_sec, query}` — per-frame relevance curve |
| `GET /health` | — | `{status, model_loaded, device, checkpoint}` |

503 if the model isn't loaded; 415 on a non-video upload. Uploads are streamed to a temp file (no
in-memory read) and GPU work runs in a threadpool.

---

## Layout

```
groundzero-backend/
├── app/
│   ├── pipeline/        # frame_extractor, visual_encoder, text_encoder, temporal_context,
│   │                    # cross_modal_transformer, span_extraction, groundzero_model, orchestrator, baseline
│   ├── training/        # losses.py, augmentation.py, dataset.py
│   ├── routes/          # predict.py, attention.py, health.py
│   └── schema/          # schemas.py (Pydantic models)
├── scripts/             # train.py, evaluate.py, download_model.py, preprocess_qvhighlights.py, tests, helpers
├── config.py            # pydantic-settings (model path, fps, device, CORS, upload cap)
├── main.py              # FastAPI app — lifespan loads orchestrator + warmup
├── requirements.txt
└── WORKFLOW.md
```

---

## Training

`scripts/train.py` (standalone, runs on any SSH GPU box / Lightning.ai). AdamW, cosine LR + 500-step
warmup, fp16 autocast, combined loss `= span + 0.5·iou + 0.1·contrastive` + confidence BCE. Checkpoints
store only the ~310 MB of trainable params; frozen SigLIP 2 is reloaded from HuggingFace each run. Best
checkpoint is pushed to `shaunmarvell/qvhighlights-model`. Dataset: `shaunmarvell/qvhighlights-1fps`
(7445 train + 1550 val, ~35 GB of 384×384 JPEG frames at 1 fps).
