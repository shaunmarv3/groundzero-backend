# GroundZero — Complete System Workflow

> Given a video and a text query, return the exact start and end timestamps of the described event.

---

## Table of Contents

1. [Bird's Eye View](#1-birds-eye-view)
2. [Full Architecture Diagram](#2-full-architecture-diagram)
3. [Phase A — Video Pre-Processing (done once per video)](#3-phase-a--video-pre-processing)
4. [Phase B — Query Processing (done per user query)](#4-phase-b--query-processing)
5. [Every Component Explained](#5-every-component-explained)
6. [Data Shapes at Every Step](#6-data-shapes-at-every-step)
7. [Training Flow](#7-training-flow)
8. [Inference Flow (Coarse-to-Fine)](#8-inference-flow-coarse-to-fine)
9. [File Map](#9-file-map)

---

## 1. Bird's Eye View

```
VIDEO FILE  ──────────────────────────────────────────────────────────┐
                                                                       │
  [Frame Extraction]  →  [Visual Encoder]  →  [Temporal Context]      │
       1fps                SigLIP 2 So400m      Dilated Conv1D         │
  frames as PIL imgs    (N, 1152) vectors     (N, 1152) enriched       │
                                                                       │
                              ↓  STOP HERE (can cache these vectors)   │
                                                                       │
USER TYPES QUERY ─────────────────────────────────────────────────────┘
                                                                       │
  [Text Encoder]  →  [Cross-Modal Transformer]  →  [Span Head]        │
   SigLIP 2 text       Query ↔ Frames fuse          per-frame         │
  (1, 1152) vector      (N, 1152) enriched          start/end scores  │
                                                          ↓            │
                                              (start_sec, end_sec)    │
```

**Key insight:** video processing and query processing are two separate phases.
The video only needs to be processed once. Every query after that is fast.

---

## 2. Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        GroundZeroModel                          │
│                                                                 │
│  VIDEO ──► FrameExtractor ──► VisualEncoder ──► TemporalCtx    │
│              (ffmpeg)          (SigLIP 2)       (Conv1D x4)    │
│              N frames          (N,1152)          (N,1152)       │
│                                                     │           │
│                                              frame_embeddings   │
│                                                     │           │
│  QUERY ─────────────────► TextEncoder               │           │
│  (string)                  (SigLIP 2)               │           │
│                            (1,1152)                 │           │
│                               │                     │           │
│                       query_embedding               │           │
│                               │                     │           │
│                               └──────► CrossModal ◄─┘          │
│                                        Transformer              │
│                                        (N,1152)                 │
│                                            │                    │
│                                       SpanHead                  │
│                                       start_scores (N,)         │
│                                       end_scores   (N,)         │
│                                            │                    │
│                                    best (start_idx, end_idx)    │
│                                            │                    │
│                                    × (duration / N)             │
│                                            │                    │
│                                    (start_sec, end_sec) ◄───────┘
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase A — Video Pre-Processing

> This phase runs **once per video**. Results can be cached. No query needed yet.

### Step 1 — Frame Extraction (`frame_extractor.py`)

```
Input:  video file path  (any format ffmpeg supports)
Output: List of (timestamp_sec, PIL.Image)

What happens:
  - ffmpeg decodes the video
  - 1 frame extracted per second (1fps) for coarse pass
  - Each frame resized to 384×384 RGB (SigLIP 2's expected input)
  - Returns (timestamp, image) pairs so we always know when each frame is

Example: 3-minute video → 180 frames
```

**Why 1fps?** Coarse pass only. In inference, a second 4fps pass refines around the predicted region. During training, 1fps gives enough coverage for the model to learn boundaries.

---

### Step 2 — Visual Encoding (`visual_encoder.py`)

```
Input:  List[PIL.Image]  — N frames
Output: Tensor (N, 1152) — one 1152-d vector per frame

What happens:
  - SigLIP 2 So400m processes each frame through 27 transformer blocks
  - Patch embeddings → pooled → 1152-d vector
  - LoRA adapters (last 4 blocks) adjust the representation for grounding
  - Output: each frame is now a 1152-d semantic fingerprint

Params: 1136.2M total / 0.147M trainable (LoRA only)
```

**What the 1152-d vector contains:** a compressed representation of visual content — objects, actions, scene type, motion blur. Frames that look similar produce similar vectors.

**Why LoRA only on last 4 blocks?**
```
Blocks  0–10:  low-level (edges, textures, patches)     → universal, freeze
Blocks 11–22:  mid-level (objects, shapes)              → mostly universal, freeze
Blocks 23–26:  high-level (semantic meaning, context)   → adapt for grounding
```

---

### Step 3 — Temporal Context (`temporal_context.py`)

```
Input:  (N, 1152) frame embeddings  +  timestamps (N,) fractional [0,1]
Output: (N, 1152) temporally-enriched frame embeddings

What happens:
  1. Sinusoidal positional encoding injected:
       each frame gets a unique t/T position vector added
       (so downstream layers know where in the video each frame is)

  2. 4-layer dilated Conv1D:
       dilation=[1,2,4,8], kernel_size=5 → receptive field = 61 frames

       each output frame now contains a blend of:
         - its own visual content
         - what happened ±30 frames (seconds) around it

Params: fully trainable (random init, learned during Phase 5)
```

**Why fractional timestamps (t/T) not raw frame indices?**
Frame index 50 means different things in a 100-frame video vs a 500-frame video.
t/T = 0.5 always means "halfway through" regardless of video length.

**⛔ PROCESSING STOPS HERE FOR PHASE A.**
The (N, 1152) tensor can be cached to disk.
Every query for this video reuses these cached vectors — no need to re-run
FrameExtractor, VisualEncoder, or TemporalContext.

---

## 4. Phase B — Query Processing

> This phase runs **per user query**. Takes the cached frame embeddings + fresh query.

### Step 4 — Text Encoding (`text_encoder.py`)

```
Input:  query string  e.g. "person opens the refrigerator"
Output: Tensor (1, 1152) — one 1152-d vector for the query

What happens:
  - SigLIP 2 text tokenizer converts string → token IDs
  - Text tower (transformer) encodes tokens → pooled 1152-d vector
  - Output lives in the SAME embedding space as the frame vectors
    (this is why SigLIP 2 works — it was trained to align vision + language)

Params: 0 trainable — fully frozen
```

**Why frozen?** The text tower already understands language perfectly from SigLIP 2 pretraining. We're not training a new language model — we're reusing the one embedded in SigLIP 2.

**Why same 1152-d space as frames?** SigLIP 2 was trained with a contrastive objective — "dog on a beach" and a photo of a dog on a beach should have similar vectors. This alignment is exactly what we exploit.

---

### Step 5 — Cross-Modal Fusion (`cross_modal_transformer.py`)

```
Input:  frames (B, N, 1152)  +  query (B, 1, 1152)
Output: (B, N, 1152) — each frame now query-aware

What happens (4 layers, each layer has 2 steps):

  CROSS-ATTENTION:
    Q = query, K = frames, V = frames
    Query attends over all N frames → attention weights (B, 1, N) = relevance map
    Enriched query (B, 1, d) broadcast back to each frame, scaled by its attention weight
    → relevant frames (high weight) absorb strong query signal
    → irrelevant frames (weight ≈ 0) get almost no update
    NOTE: Q=frames direction is wrong — with 1 key (query), softmax=1.0 always → degenerate projection

  SELF-ATTENTION:
    Q = K = V = frames  (after cross-attention)
    Frames talk to ALL other frames globally
    → relevance signal spreads: neighbours of relevant frames become relevant
    → this is how the model learns that events have duration, not just 1 frame

  Repeat 4× — each layer refines the previous layer's output:
    Layer 1: crude alignment (query finds roughly relevant region)
    Layer 2: neighbours pull in, region sharpens
    Layer 3: query re-examines sharpened region, more precise
    Layer 4: clean start/end signal ready for span extraction

Params: fully trainable (randomly initialized)
```

**Difference from TemporalContext:**
```
TemporalContext:       local, time-based, query-unaware
                       "what happened around me in time?"
                       runs BEFORE query is involved

CrossModalTransformer: global, query-conditioned
                       "which frames share my query-relevance?"
                       runs AFTER query is joined
```

---

### Step 6 — Span Extraction (`span_extraction.py`)

```
Input:  (B, N, 1152) query-aware frame embeddings
Output: start_idx, end_idx (frame indices), confidence ∈ [0,1]

What happens:
  - start_scorer: Linear(1152 → 1) applied to every frame → N start scores
  - end_scorer:   Linear(1152 → 1) applied to every frame → N end scores
  - confidence_head: Linear(1152 → 1) on mean pooled → single confidence score

  Decode best span:
    find argmax of start_scores → candidate start
    find argmax of end_scores where end >= start → candidate end
    (enforces that end timestamp always comes after start)

  Convert to seconds:
    start_sec = start_idx / N * video_duration
    end_sec   = end_idx   / N * video_duration

Params: fully trainable (3 small linear layers, ~3500 params total)
```

**Why per-frame scoring instead of regression?**
Regression (predict one number) is a hard optimization problem — small gradient signal.
Per-frame scoring is like reading comprehension: "which frame is the start?" — much stronger signal, same approach as QA models (BERT SQuAD).

---

## 5. Every Component Explained

| Component | File | Trainable params | Purpose |
|---|---|---|---|
| FrameExtractor | `frame_extractor.py` | 0 (no params) | decode video → frames |
| VisualEncoder | `visual_encoder.py` | 0.147M (LoRA) | frames → semantic vectors |
| TemporalContext | `temporal_context.py` | ~10M (conv weights) | inject temporal context |
| TextEncoder | `text_encoder.py` | 0 (frozen) | query → semantic vector |
| CrossModalTransformer | `cross_modal_transformer.py` | ~100M (attention + FFN) | fuse query + frames |
| SpanExtractionHead | `span_extraction.py` | ~3.5K (linear layers) | predict start/end frame |

**Total trainable: ~110M**
**Total model: ~1136M (SigLIP 2 backbone frozen, ~110M custom layers trained)**

---

## 6. Data Shapes at Every Step

```
Video (180 frames, 3-min video)
│
├─ After FrameExtractor:       List[180 × PIL.Image(384,384)]
│                              + timestamps [0.0, 1.0, 2.0, ..., 179.0]
│
├─ After VisualEncoder:        Tensor (180, 1152)   float16
│
├─ After TemporalContext:      Tensor (180, 1152)   float32
│                              (enriched, same shape)
│
│    ← CACHE POINT: save (180, 1152) tensor to disk
│
Query: "person opens the refrigerator"
│
├─ After TextEncoder:          Tensor (1, 1152)     float16
│
├─ Batched for transformer:
│     frames: (1, 180, 1152)   ← added batch dim B=1
│     query:  (1, 1,   1152)
│
├─ After CrossModalTransformer: Tensor (1, 180, 1152)
│
├─ After SpanHead:
│     start_scores:  (180,)
│     end_scores:    (180,)
│     confidence:    scalar
│     best_start:    frame index, e.g. 42
│     best_end:      frame index, e.g. 45
│
└─ Final output:
      start_sec = 42 / 180 * 180.0 = 42.0s
      end_sec   = 45 / 180 * 180.0 = 45.0s
      confidence = 0.87
```

---

## 7. Training Flow

> Phase 5 — not implemented yet. This is what will happen.

```
for each batch (video, query, gt_start_sec, gt_end_sec):

  1. FORWARD PASS
     frames    = VisualEncoder.encode_frames(video_frames)        (N, 1152)
     frames    = TemporalContext(frames, timestamps)              (N, 1152)
     query_emb = TextEncoder.encode_query(query)                  (1, 1152)
     fused     = CrossModalTransformer(frames, query_emb)         (N, 1152)
     start_logits, end_logits, conf = SpanHead(fused)             (N,), (N,), scalar

  2. CONVERT GT TO FRAME INDICES
     gt_start_idx = round(gt_start_sec / duration * N)
     gt_end_idx   = round(gt_end_sec   / duration * N)

  3. COMPUTE LOSS (three components)
     span_loss       = CrossEntropy(start_logits, gt_start_idx)
                     + CrossEntropy(end_logits,   gt_end_idx)
     iou_loss        = 1 - IoU(pred_span, gt_span)
     contrastive_loss = push non-event frames away from query

     total = span_loss + 0.5*iou_loss + 0.1*contrastive_loss

  4. BACKWARD PASS
     total.backward()
     → gradients flow to: LoRA adapters + TemporalContext + CrossModal + SpanHead
     → SigLIP 2 backbone stays frozen (no gradients reach it past LoRA)

  5. OPTIMIZER STEP
     AdamW updates all trainable params

  Repeat for 20 epochs on QVHighlights train set (~10k videos)
```

**What each component learns during training:**
```
LoRA adapters:          produce frame embeddings that are better for grounding
                        (not just generic visual similarity)

TemporalContext conv:   which temporal patterns signal event start/end
                        e.g. "motion spike followed by stable" = action event

CrossModalTransformer:  how to align "opens refrigerator" with the frame where
                        a hand reaches for a door handle

SpanHead:               how to read the fused embeddings to produce clean
                        start/end scores
```

---

## 8. Inference Flow (Coarse-to-Fine)

> Two-pass strategy. Coarse finds the region, fine pass refines within it.

### Pass 1 — Coarse (1fps, full video)

```
extract frames at 1fps → encode → temporal context → text encode → fuse → span head
→ predicted region: e.g. [38s, 50s]
```

Fast. Processes whole video but at low frame rate. Good enough to find which 10-20 second window the event is in.

### Pass 2 — Fine (4fps, ±5s around predicted region)

```
re-extract frames at 4fps between [33s, 55s]  (predicted ± 5s buffer)
→ encode → temporal context → text encode → fuse → span head
→ refined prediction: e.g. [41.25s, 44.5s]
```

4× denser sampling in the relevant region only. Gives sub-second precision without processing the entire video at high fps.

```
Timeline:
|────────────────────────[===========]────────────────────|
0s                       38s       50s                   180s
                    ↑ coarse prediction

         |──────[████████████████████████]──────|
         33s    41.25s        44.5s          55s
              ↑ fine prediction (4fps window)
```

**Latency targets:**
- 5-min video → Pass 1: ~1s, Pass 2: ~0.3s → total ~1.3s
- 30-min video → Pass 1: ~5s, Pass 2: ~0.3s → total ~5.3s
- 120-min video → Pass 1: ~20s, Pass 2: ~0.3s → total ~20.3s

Pass 2 is always fast because it only covers a ~20s window regardless of video length.

---

## 9. File Map

```
groundzero-backend/
│
├── app/
│   ├── pipeline/
│   │   ├── frame_extractor.py       Phase 2.2  ✅ done
│   │   ├── baseline.py              Phase 3.1  ✅ done
│   │   ├── visual_encoder.py        Phase 3.2  ✅ done
│   │   ├── text_encoder.py          Phase 3.3  ✅ done
│   │   ├── temporal_context.py      Phase 3.4  ✅ done
│   │   ├── cross_modal_transformer.py Phase 3.5  🔨 next
│   │   ├── span_extraction.py       Phase 3.6  ⬜ not started
│   │   └── groundzero_model.py      Phase 3.7  ⬜ not started (assembles all above)
│   │
│   └── routes/
│       ├── health.py                Phase 7.3  ⬜
│       ├── predict.py               Phase 7.3  ⬜
│       └── attention.py             Phase 7.3  ⬜
│
├── scripts/
│   ├── run_baseline.py              ✅ done
│   ├── test_temporal_context.py     ✅ done  all 5 tests passed
│   ├── test_visual_encoder.py       ✅ done  all 4 tests passed
│   ├── test_text_encoder.py         ✅ done  all 5 tests passed
│   ├── test_cross_modal_transformer.py  🔨 next
│   └── test_span_extraction.py      ⬜ not started
│
├── main.py                          Phase 1.1  ✅ done
├── config.py                        Phase 1.1  ✅ done
├── requirements.txt                 Phase 1.1  ✅ done
└── WORKFLOW.md                      ← you are here
```

---

## Quick Reference — The One-Sentence Summary of Each Component

| Component | One sentence |
|---|---|
| FrameExtractor | Pulls N frames out of a video file at a given fps using ffmpeg |
| VisualEncoder | Turns each frame into a 1152-d vector using SigLIP 2 with LoRA fine-tuning |
| TemporalContext | Makes each frame vector aware of what happened ±30 seconds around it |
| TextEncoder | Turns the user's query string into a 1152-d vector in the same space as the frames |
| CrossModalTransformer | Makes each frame vector aware of what the query is asking for |
| SpanExtractionHead | Reads the fused frame vectors and picks which frame is the start and which is the end |
| GroundZeroModel | Wires all of the above into a single forward pass |
