# GroundZero — Temporal Video Grounding from Natural Language

## Overview

GroundZero is a deep learning system that finds the exact moment inside a video that corresponds to a natural language description. Given a long, unstructured video and a text query — _"the moment the speaker slams the table"_ or _"when the engineer explains backpropagation"_ — GroundZero predicts the precise start and end timestamps of that event.

This is not video classification (labeling the whole video) and not video captioning (describing what happens). It is **temporal grounding** — pinpointing where inside a continuous video a described event occurs, measured in seconds.

The system is designed for unstructured, untrimmed, chapter-free video: security footage, lecture archives, legal depositions, sports recordings, medical training videos. These are the cases where existing chapter-based or metadata-based solutions do not work.

---

## Problem Statement

Most long-form video is unstructured. A two-hour university lecture has no chapters. A court deposition recording has no index. A security camera feed has no labels. A sports match archive has no event timestamps. If you want to find a specific moment inside any of these, you either scrub manually or you need a system that understands both language and video simultaneously.

Existing approaches fail in different ways:

- **Video search (retrieval):** Finds which video in a database matches a query. Does not localize _where_ inside the video the event is.
- **Video captioning:** Describes what happens in a clip. Does not answer "find me the moment where X happens."
- **Chapter-based navigation:** Only works for videos where creators manually added chapters. Most video has none.
- **Keyword search on transcripts:** Only works when the event is spoken aloud. Visual events — a gesture, a reaction, an action — are invisible to text search.

**The gap GroundZero fills:** Given any untrimmed video and a natural language description of an event, output `(start_time, end_time)` in seconds with high temporal precision.

---

## System Architecture — Full Pipeline

```
Input: Video file + Text Query
    │                    │
    ▼                    ▼
┌──────────────┐   ┌──────────────────┐
│ Frame        │   │ Text Encoder     │
│ Sampler      │   │ CLIP text tower  │
│ 1fps → N     │   │ (frozen)         │
│ frames       │   │ Query → 512-d    │
│              │   │ → project to     │
│              │   │   768-d          │
└──────┬───────┘   └────────┬─────────┘
       │                    │
       ▼                    │
┌──────────────┐            │
│ Visual       │            │
│ Encoder      │            │
│ CLIP ViT-L   │            │
│ + LoRA       │            │
│ adapters     │            │
│ Frame → 768-d│            │
└──────┬───────┘            │
       │                    │
       ▼                    ▼
┌────────────────────────────────┐
│   Temporal Context Module      │
│   Dilated 1D Conv [1,2,4,8]   │
│   + Positional Encoding       │
│   → time-aware frame features │
│   (~60s receptive field)       │
└──────────────┬─────────────────┘
               │
               ▼
┌────────────────────────────────┐
│   Cross-Modal Attention        │
│   Transformer (4L, 8H)        │
│   Query attends over frames   │
│   → grounded frame features   │
└──────────────┬─────────────────┘
               │
               ▼
┌────────────────────────────────┐
│   Span Extraction Head         │
│   Per-frame start/end scoring  │
│   → best (start, end) span    │
│   Confidence head              │
│   → event presence score       │
└──────────────┬─────────────────┘
               │
               ▼
  Coarse: { start: 310s, end: 330s }
               │
               ▼ (re-sample region at 4fps)
               │
  Fine:   { start: 312.4s, end: 328.1s, confidence: 0.91 }
```

---

## Model Architecture — Detailed

### Stage 1: Frame Sampling and Visual Encoding

**Frame Sampler:**
Videos are sampled at 1 frame per second (fps) for the initial coarse pass. A 30-minute video produces 1,800 frames. Sampling rate is configurable: 0.5fps for very long videos (>60 min), 2fps for short clips (<10 min) where fine-grained localization matters.

For inference, a **coarse-to-fine strategy** is used: the initial 1fps pass identifies the approximate region, then a second pass re-samples that region at 4fps for precise boundary localization (see Workflow section).

**Visual Encoder: CLIP ViT-L/14 with LoRA Adapters**
Each sampled frame is passed through the CLIP visual encoder (ViT-L/14). CLIP is used because its visual representations are already aligned with natural language — it understands that a frame showing a person raising their hand corresponds semantically to "someone raises their hand." This eliminates the need to train a visual encoder from scratch.

Output: sequence of frame embeddings `F = [f_1, f_2, ..., f_N]` where each `f_i ∈ R^768` and N = number of sampled frames.

**CLIP is mostly frozen, with lightweight LoRA adapters on the last 4 transformer blocks.** Fully fine-tuning CLIP requires significantly more GPU memory and data than available on Colab free tier. Instead, low-rank adaptation (LoRA) is applied to the query and value projection layers of CLIP's last 4 blocks — adding ~0.5M trainable parameters while keeping CLIP's 300M parameters frozen. This gives the visual encoder limited temporal adaptability without the memory cost of full fine-tuning.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=list(range(20, 24)),  # last 4 of 24 ViT blocks
)
clip_visual = get_peft_model(clip_model.visual, lora_config)
# ~0.5M trainable params — fits comfortably in Colab free tier
```

---

### Stage 2: Temporal Context Module

Raw CLIP frame embeddings have no temporal awareness — each frame is encoded independently with no knowledge of what came before or after. The Temporal Context Module fixes this.

**Architecture:**

- **Positional Encoding:** Sinusoidal temporal position encoding added to each frame embedding, encoding the frame's timestamp relative to total video duration (not just frame index)
- **Dilated 1D Temporal Convolution:** 4-layer 1D conv with kernel size 5 and exponentially increasing dilation rates `[1, 2, 4, 8]`, applied across the frame sequence. Dilated convolutions exponentially expand the receptive field without adding parameters — the effective receptive field covers ~60 seconds at 1fps (vs ~13 seconds with standard convolutions), enabling the model to capture long-range temporal patterns like scene transitions and multi-step events.
- **Output:** Time-aware frame embeddings `F' = [f'_1, f'_2, ..., f'_N]` with the same dimensionality but now encoding temporal context

```python
class TemporalContextModule(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=5, padding='same', dilation=1),   # ±2 frames
            nn.Conv1d(d_model, d_model, kernel_size=5, padding='same', dilation=2),   # ±4 frames
            nn.Conv1d(d_model, d_model, kernel_size=5, padding='same', dilation=4),   # ±8 frames
            nn.Conv1d(d_model, d_model, kernel_size=5, padding='same', dilation=8),   # ±16 frames
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])

    def forward(self, frame_embeddings, timestamps):
        x = frame_embeddings + self.pos_encoding(timestamps)
        x = x.transpose(1, 2)  # (batch, d_model, N) for Conv1d
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = F.gelu(conv(x)) + residual  # residual connection
            x = norm(x.transpose(1, 2)).transpose(1, 2)
        return x.transpose(1, 2)  # back to (batch, N, d_model)
```

The positional encoding uses fractional timestamps (`t/T` where T is total video duration) rather than absolute frame indices. This makes the model generalise across videos of different lengths — a timestamp at 30% of a 10-minute video and 30% of a 2-hour video are encoded the same way, which is the right inductive bias.

---

### Stage 3: Text Encoding

**Text Encoder: CLIP text tower (frozen)**
The query string is passed through the frozen CLIP text encoder.

Output: query embedding `q ∈ R^512`, projected to `R^768` via a learned linear layer to match the visual embedding dimension.

**Frozen for the same reason as the visual encoder.** The cross-modal transformer learns to align the spaces, not modify the encoders.

---

### Stage 4: Cross-Modal Attention Transformer

This is the core learned component. It takes the time-aware frame sequence and the query embedding and learns to produce grounded frame representations — embeddings that are "lit up" near the event described by the query and suppressed everywhere else.

**Architecture:**

```python
class CrossModalTransformer(nn.Module):
    def __init__(self, d_model=768, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        # Query is the "question being asked"
        # Frame sequence is the "context being searched"
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.self_attention_layers = nn.ModuleList([
            SelfAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, frame_embeddings, query_embedding):
        # query_embedding shape: (batch, 1, 768)
        # frame_embeddings shape: (batch, N_frames, 768)

        x = frame_embeddings
        for cross_attn, self_attn in zip(self.cross_attention_layers, self.self_attention_layers):
            # Cross-attention: query attends over frames
            # Q = query (what we're looking for)
            # K, V = frames (the temporal context being searched)
            # This produces query-conditioned relevance scores for each frame
            attn_output = cross_attn(query=query_embedding, key=x, value=x)

            # Broadcast attention output back to frame dimension
            # Each frame is modulated by how relevant it is to the query
            x = x + attn_output.expand_as(x)

            # Self-attention: frames attend to each other
            # "given query relevance, refine temporal context"
            x = self_attn(x)

        return x  # grounded frame embeddings: (batch, N_frames, 768)
```

**Why query-over-frames attention:** The query is a single token — using it as K/V in cross-attention (the original design) reduces cross-attention to a learned projection, since there's only one key to attend to. Flipping the direction so the query attends over the full frame sequence produces a meaningful attention distribution — a relevance map over the timeline that highlights which frames match the query. This relevance signal is then broadcast back and refined through self-attention.

**Why cross-attention before self-attention at each layer:** Cross-attention injects the query signal into the frame representations first, then self-attention refines the temporal context given that signal. Interleaving them at every layer propagates query relevance progressively through the temporal dimension.

**4 layers, 8 heads.** Deeper than 4 layers does not improve performance significantly on QVHighlights and adds memory overhead that hits Colab limits.

---

### Stage 5: Per-Frame Span Extraction Head

The boundary predictor uses a **span extraction** approach inspired by extractive question answering (e.g., BERT for SQuAD). Instead of pooling all frame features into a single vector and regressing two numbers, each frame in the sequence receives a start probability and an end probability. The predicted moment is the highest-scoring valid span where `end >= start`.

This preserves the full temporal structure — the model directly scores "is this frame the start of the event?" and "is this frame the end?" rather than compressing all temporal information into a single pooled vector.

**Architecture:**

```python
class SpanExtractionHead(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        # Per-frame start/end scoring
        self.start_scorer = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)    # per-frame start logit
        )
        self.end_scorer = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)    # per-frame end logit
        )
        # Confidence head: does this video contain the described event at all?
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()         # 0 = event absent, 1 = event present
        )

    def forward(self, grounded_features):
        # grounded_features: (batch, N_frames, 768) — temporal dimension preserved
        start_logits = self.start_scorer(grounded_features).squeeze(-1)  # (batch, N)
        end_logits = self.end_scorer(grounded_features).squeeze(-1)      # (batch, N)

        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)

        # Decode best valid span: argmax of start_probs[i] * end_probs[j] for j >= i
        # Same decoding algorithm as extractive QA (BERT SQuAD)
        best_start, best_end = decode_best_span(start_probs, end_probs)

        # Confidence from pooled features
        pooled = grounded_features.mean(dim=1)
        confidence = self.confidence_head(pooled)

        return best_start, best_end, confidence, start_probs, end_probs
```

**Output is frame indices, converted to fractional timestamps (0–1) by dividing by N_frames.** Multiply by video duration to get seconds. This makes the model length-agnostic.

**Why span extraction over regression:** A pooled regression head discards all temporal structure — it must predict timestamps from a single 768-d vector with no spatial information. Span extraction keeps the full frame sequence and directly identifies start/end boundaries in the temporal dimension, giving the model access to local features at decision time. This is the same insight that made extractive QA (BERT SQuAD) outperform generative approaches.

---

### Loss Function

Training uses three losses jointly:

**Loss 1: Span Extraction Loss (primary)**

With the per-frame span extraction head, boundary prediction is trained as a classification problem — the ground truth start and end frames are target indices.

```python
def span_extraction_loss(start_logits, end_logits, gt_start_idx, gt_end_idx):
    # Cross-entropy over frame positions — same as extractive QA training
    start_loss = F.cross_entropy(start_logits, gt_start_idx)
    end_loss = F.cross_entropy(end_logits, gt_end_idx)
    return start_loss + end_loss
```

**Loss 2: Temporal IoU Loss (auxiliary)**

```python
def temporal_iou_loss(pred_start, pred_end, gt_start, gt_end):
    intersection = torch.clamp(
        torch.min(pred_end, gt_end) - torch.max(pred_start, gt_start), min=0
    )
    union = torch.max(pred_end, gt_end) - torch.min(pred_start, gt_start)
    iou = intersection / (union + 1e-8)
    return 1 - iou.mean()  # maximize IoU = minimize 1 - IoU
```

Provides a complementary gradient signal that directly optimizes the evaluation metric (IoU) in addition to the span classification objective.

**Loss 3: Intra-Video Contrastive Loss**

Instead of relying on in-batch negatives (which requires large batch sizes that exceed Colab memory limits), negatives are mined from **non-overlapping segments of the same video**. These are harder negatives (same visual domain, different temporal content) and don't require large batches to be effective.

```python
def contrastive_loss_intra_video(query_emb, grounded_features, gt_start_idx, gt_end_idx, temperature=0.07):
    # Positive: pooled features from ground truth segment
    pos_emb = grounded_features[:, gt_start_idx:gt_end_idx+1, :].mean(dim=1)

    # Negatives: 8 random non-overlapping segments from the same video
    neg_segments = sample_non_overlapping_segments(
        n_frames=grounded_features.shape[1],
        gt_start=gt_start_idx, gt_end=gt_end_idx, n_negatives=8
    )
    neg_embs = torch.stack([
        grounded_features[:, s:e+1, :].mean(dim=1) for s, e in neg_segments
    ])

    pos_sim = F.cosine_similarity(query_emb, pos_emb) / temperature
    neg_sims = F.cosine_similarity(query_emb.unsqueeze(1), neg_embs, dim=-1) / temperature

    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sims.squeeze(0)], dim=-1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
```

**Why intra-video negatives:** With Colab batch sizes of 2-4, in-batch negatives provide only 1-3 negatives — too few for meaningful contrastive learning (random chance is 25-50%). Intra-video mining provides 8 hard negatives per sample regardless of batch size, and they're semantically harder — "this is the wrong moment in the right video" is a more useful training signal than "this is a completely different video."

**Total loss:**

```python
total_loss = span_loss + 0.5 * iou_loss + 0.1 * contrastive_loss
```

Weights are hyperparameters tuned during training.

---

### Data Augmentation

With ~10K training queries from QVHighlights, overfitting is a significant risk — especially for a transformer with 4 layers and 8 heads. Four augmentation strategies are applied during training:

**1. Temporal Jitter:** Randomly shift ground truth boundaries by ±10% of the event duration. Prevents the model from memorizing exact boundary positions and improves robustness to annotation noise.

**2. Random Temporal Crop:** Randomly crop 60-100% of the video while ensuring the ground truth segment remains fully included (with ±5 second margins). This changes the relative position and proportion of the event within the video, forcing the model to generalize rather than memorize absolute positions.

**3. Speed Perturbation:** Randomly drop or duplicate 10% of frames (applied 30% of the time). Simulates variable playback speed and makes the model robust to frame rate variations.

**4. Query Paraphrasing:** Pre-generate 3-5 alternate phrasings per training query using an LLM (offline, stored in a JSON lookup). During training, randomly sample one paraphrase per epoch. This expands the effective training vocabulary and reduces overfitting to specific query phrasings.

```python
# Example paraphrase lookup (generated offline)
{
    "the moment the speaker slams the table": [
        "when someone hits the table",
        "the part where the table gets slammed",
        "speaker banging on the table"
    ]
}
```

Additionally, **negative query injection** is used to train the confidence head: 20% of training samples are paired with a query from a different video, with confidence target = 0. This teaches the model to output low confidence when the described event is genuinely absent.

---

## Datasets

| Dataset                  | Size            | What it contains                                                                                                      | Link                                |
| :----------------------- | :-------------- | :-------------------------------------------------------------------------------------------------------------------- | :---------------------------------- |
| **QVHighlights**         | 10,310 queries  | YouTube videos (avg 150s) with start/end timestamp annotations per query. Clean, well-labeled, the primary benchmark. | `huggingface: Davlan/qvhighlights`  |
| **Charades-STA**         | 16,128 queries  | Indoor activity videos. Each clip has multiple annotated temporal moments with text descriptions.                     | `huggingface: charades_sta`         |
| **ActivityNet Captions** | 100,000 queries | 20,000 YouTube videos with dense temporal annotations and captions. Longest videos in the benchmark suite.            | `huggingface: activitynet_captions` |
| **DiDeMo**               | 33,005 queries  | Flickr videos (avg 30s), moment descriptions by crowd workers. Good diversity of query styles.                        | `huggingface: didemo`               |
| **TACoS**                | 18,818 queries  | Kitchen activity videos. Dense annotations, narrow domain. Good for testing domain robustness.                        | `github: TACoS`                     |

**Training split:** QVHighlights (primary) + Charades-STA + DiDeMo
**Validation split:** QVHighlights val set (standard benchmark)
**Test split:** QVHighlights test + ActivityNet Captions (cross-dataset generalization)

---

## Workflow — Step by Step

**User submits:** Video file (mp4) + query: _"when the interviewer asks about salary expectations"_

**Pass 1 — Coarse Localization (1fps):**

1. **Frame extraction:** Video sampled at 1fps → N frames extracted
2. **Visual encoding:** Each frame passed through CLIP ViT-L/14 (with LoRA adapters) → N × 768 frame embeddings
3. **Temporal context:** Dilated 1D temporal conv + positional encoding → time-aware frame embeddings
4. **Text encoding:** Query passed through frozen CLIP text tower → 512-d embedding → projected to 768-d
5. **Cross-modal attention:** Query attends over frame sequence → grounded frame features with query-relevance scores
6. **Span extraction:** Per-frame start/end scoring identifies the best span, confidence head outputs event presence score
7. **Coarse output:** `{ start: 310s, end: 330s, confidence: 0.88 }`

**Pass 2 — Fine Refinement (4fps):**

8. **Region re-sampling:** The predicted region ±5 seconds is re-sampled at 4fps (e.g., 305s–335s → 120 frames at 4fps)
9. **Steps 2–6 repeat** on the high-resolution region
10. **Final output:** `{ start: 312.4s, end: 328.1s, confidence: 0.91 }` — 4x better temporal precision

The coarse-to-fine strategy keeps memory manageable (full video at 1fps) while achieving high temporal precision where it matters (4fps in the target region). Total inference cost is ~1.5x a single pass, not 4x.

If confidence < 0.4: _"The described event was not found in this video with sufficient confidence."_

---

## Production-Grade Evaluation Suite

Every component is evaluated independently and end-to-end. All results logged to W&B. Baselines measured before any training begins.

---

### Eval 0: Baseline Measurements (Run First)

Before training, measure what zero-shot CLIP cosine similarity retrieval achieves on QVHighlights. This is your floor — the simplest possible approach with no learned temporal reasoning.

```python
# Zero-shot CLIP baseline: for each query, find the frame with highest cosine
# similarity to the query embedding. Use that frame as the center of a
# fixed-width window (e.g., ±5 seconds) as the predicted moment.

def clip_zeroshot_baseline(video_frames, query, window_width=10):
    query_emb = clip_text_encoder(query)                     # (512,)
    frame_embs = clip_visual_encoder(video_frames)           # (N, 512)
    similarities = F.cosine_similarity(query_emb, frame_embs)
    best_frame_idx = similarities.argmax().item()
    center_time = best_frame_idx / fps
    return max(0, center_time - window_width/2), center_time + window_width/2

# Measure R@1 IoU=0.3 and R@1 IoU=0.5 of this baseline on QVHighlights val set
# Your trained model must beat this — if it doesn't, training has failed
```

**Document the baseline numbers before any training. Every improvement claim is relative to this.**

---

### Eval 1: Temporal Grounding Metrics (Primary)

Temporal video grounding is evaluated using **Recall@K at IoU threshold θ** — the standard metric across all benchmarks.

#### 1.1 R@1, IoU=0.5 and IoU=0.7 (Primary Metrics)

```python
def temporal_iou(pred_start, pred_end, gt_start, gt_end):
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / (union + 1e-8)

def recall_at_k_iou(predictions, ground_truths, k=1, iou_threshold=0.5):
    """
    For each query, check if the top-k predictions contain one with IoU >= threshold.
    R@1 IoU=0.5: does the best prediction overlap ground truth by at least 50%?
    """
    hits = 0
    for pred_list, gt in zip(predictions, ground_truths):
        top_k_preds = pred_list[:k]
        for pred in top_k_preds:
            if temporal_iou(pred['start'], pred['end'], gt['start'], gt['end']) >= iou_threshold:
                hits += 1
                break
    return hits / len(ground_truths)

# Primary evaluation
r1_iou05 = recall_at_k_iou(predictions, ground_truths, k=1, iou_threshold=0.5)
r1_iou07 = recall_at_k_iou(predictions, ground_truths, k=1, iou_threshold=0.7)
r5_iou05 = recall_at_k_iou(predictions, ground_truths, k=5, iou_threshold=0.5)

print(f"R@1 IoU=0.5: {r1_iou05:.4f}")  # Primary metric — report this first
print(f"R@1 IoU=0.7: {r1_iou07:.4f}")  # Stricter threshold
print(f"R@5 IoU=0.5: {r5_iou05:.4f}")  # How often correct in top-5 predictions
```

**Target benchmarks (QVHighlights val set):**

> **Note:** Baseline numbers below are **estimated**, not measured. The actual CLIP zero-shot baseline must be measured before training (see Eval 0). All targets are relative to baseline and will be adjusted after baseline measurement.

```
Metric         | CLIP baseline (est.) | Target (trained) | SOTA reference
---------------|----------------------|------------------|----------------
R@1 IoU=0.5    | ~0.35               | > 0.50           | ~0.65 (Moment-DETR)
R@1 IoU=0.7    | ~0.18               | > 0.32           | ~0.45 (Moment-DETR)
R@5 IoU=0.5    | ~0.55               | > 0.72           | ~0.85 (Moment-DETR)
```

Note: SOTA uses much larger models (DETR-based, pretrained on video-language pairs). Your target is meaningful improvement over baseline, not SOTA. Reaching 77% of SOTA performance with a simpler architecture is a legitimate research contribution.

#### 1.2 mean Average Precision (mAP) at IoU Thresholds

```python
def mean_average_precision(predictions, ground_truths, iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Compute mAP averaged across multiple IoU thresholds.
    Standard COCO-style evaluation adapted for temporal grounding.
    """
    aps = []
    for iou_thresh in iou_thresholds:
        ap = compute_ap_at_iou(predictions, ground_truths, iou_thresh)
        aps.append(ap)
    return np.mean(aps)

map_score = mean_average_precision(predictions, ground_truths)
print(f"mAP@[0.5:0.75]: {map_score:.4f}")
# Target: > 0.35 on QVHighlights
```

#### 1.3 Temporal Displacement Error

R@K at IoU threshold tells you if predictions are good enough. Temporal displacement tells you how far off they are when they're wrong.

```python
def temporal_displacement_analysis(predictions, ground_truths):
    start_errors = []
    end_errors = []
    center_errors = []
    duration_errors = []

    for pred, gt in zip(predictions, ground_truths):
        start_errors.append(abs(pred['start'] - gt['start']))
        end_errors.append(abs(pred['end'] - gt['end']))

        pred_center = (pred['start'] + pred['end']) / 2
        gt_center = (gt['start'] + gt['end']) / 2
        center_errors.append(abs(pred_center - gt_center))

        pred_dur = pred['end'] - pred['start']
        gt_dur = gt['end'] - gt['start']
        duration_errors.append(abs(pred_dur - gt_dur))

    for name, errors in [
        ("Start error (s)", start_errors),
        ("End error (s)", end_errors),
        ("Center error (s)", center_errors),
        ("Duration error (s)", duration_errors)
    ]:
        print(f"{name}: mean={np.mean(errors):.2f}, median={np.median(errors):.2f}, p95={np.percentile(errors, 95):.2f}")

# Target: median center error < 8 seconds on QVHighlights
```

---

### Eval 2: Cross-Dataset Generalization

A model that only works on QVHighlights is not a robust model — it has memorized dataset-specific patterns. Evaluate on datasets the model was not trained on.

#### 2.1 Zero-Shot Generalization to Charades-STA

```python
# Train ONLY on QVHighlights. Evaluate on Charades-STA with no fine-tuning.
# This tests whether the model learned generalizable temporal grounding
# or QVHighlights-specific patterns.

charades_results = evaluate_on_dataset(model, charades_sta_test, trained_on="qvhighlights")
print(f"Charades-STA R@1 IoU=0.5 (zero-shot): {charades_results['r1_iou05']:.4f}")
print(f"Charades-STA R@1 IoU=0.7 (zero-shot): {charades_results['r1_iou07']:.4f}")
# Target: > 0.35 R@1 IoU=0.5 zero-shot (significant drop from in-domain is expected)
# If < 0.20, the model has overfit to QVHighlights distribution
```

#### 2.2 ActivityNet Captions (Long-Video Robustness)

ActivityNet has much longer videos than QVHighlights. If your model degrades significantly on longer videos, your temporal positional encoding is not generalizing to length variation.

```python
# Bucket test set by video duration
short_videos = [v for v in activitynet_test if v['duration'] < 120]   # < 2 min
medium_videos = [v for v in activitynet_test if 120 <= v['duration'] < 600]
long_videos = [v for v in activitynet_test if v['duration'] >= 600]   # > 10 min

for bucket, name in [(short_videos, "short (<2min)"), (medium_videos, "medium (2-10min)"), (long_videos, "long (>10min)")]:
    results = evaluate_on_dataset(model, bucket)
    print(f"{name}: R@1 IoU=0.5 = {results['r1_iou05']:.4f}")

# Target: performance drop from short to long < 15 percentage points
# If drop > 25pp, your temporal encoding fails on long videos
```

---

### Eval 3: Query Type Analysis

Not all queries are equally hard. Understanding which query types the model handles well is as important as overall accuracy.

#### 3.1 Query Complexity Stratification

```python
query_categories = {
    "action":       ["slams", "runs", "picks up", "drops", "opens"],
    "speech":       ["explains", "mentions", "asks about", "says", "announces"],
    "visual_state": ["is wearing", "appears on screen", "stands near", "holds"],
    "abstract":     ["funniest moment", "most intense", "turning point"],
    "temporal_rel": ["after the introduction", "before the break", "at the end when"]
}

def categorize_query(query_text, categories):
    for cat, keywords in categories.items():
        if any(kw in query_text.lower() for kw in keywords):
            return cat
    return "other"

# Evaluate R@1 IoU=0.5 per category
for category in query_categories:
    cat_queries = [q for q in test_queries if categorize_query(q['text']) == category]
    results = evaluate_on_dataset(model, cat_queries)
    print(f"{category}: R@1 IoU=0.5 = {results['r1_iou05']:.4f}, n={len(cat_queries)}")

# Expected: action queries are easiest, abstract queries are hardest
# This is your model's capability profile — document it honestly
```

#### 3.2 Query Length Analysis

```python
# Bin queries by token length
bins = [(1, 5), (6, 10), (11, 15), (16, 25)]
for min_len, max_len in bins:
    bin_queries = [q for q in test_queries if min_len <= len(q['text'].split()) <= max_len]
    results = evaluate_on_dataset(model, bin_queries)
    print(f"Query length {min_len}-{max_len} tokens: R@1={results['r1_iou05']:.4f}, n={len(bin_queries)}")
```

---

### Eval 4: Model Component Ablation

Ablation studies prove that each component is actually contributing. Remove one component at a time and measure the performance drop.

```python
ablation_configs = {
    "Full model":                  {"temporal_conv": True,  "cross_attn": True,  "contrastive_loss": True},
    "No temporal conv":            {"temporal_conv": False, "cross_attn": True,  "contrastive_loss": True},
    "No cross-modal attention":    {"temporal_conv": True,  "cross_attn": False, "contrastive_loss": True},
    "No contrastive loss":         {"temporal_conv": True,  "cross_attn": True,  "contrastive_loss": False},
    "No temporal conv, no cross":  {"temporal_conv": False, "cross_attn": False, "contrastive_loss": True},
    "CLIP baseline (no training)": {"temporal_conv": False, "cross_attn": False, "contrastive_loss": False},
}

ablation_results = {}
for config_name, config in ablation_configs.items():
    model_variant = build_model(config)
    # Train each variant under identical conditions
    results = evaluate_on_dataset(model_variant, qvhighlights_val)
    ablation_results[config_name] = results['r1_iou05']
    print(f"{config_name}: R@1 IoU=0.5 = {results['r1_iou05']:.4f}")

# Expected result: each removed component reduces R@1
# If removing a component does NOT hurt performance, it is not contributing — diagnose why
```

**This table is your most important research artifact.** It proves your architectural choices are justified, not arbitrary.

---

### Eval 5: Attention Visualization (Qualitative Eval)

Quantitative metrics tell you if the model works. Attention visualization tells you _why_ it works (or why it fails).

#### 5.1 Cross-Modal Attention Heatmaps

```python
def visualize_attention(model, video_path, query, output_path):
    """
    Extract cross-modal attention weights from the transformer.
    Plot: x-axis = video timeline (seconds), y-axis = attention weight
    Expected: high attention near the ground truth moment
    """
    frames, timestamps = extract_frames(video_path)
    query_emb = encode_query(query)
    frame_embs = encode_frames(frames)

    # Hook into cross-attention layer to extract attention weights
    attention_weights = []
    def attention_hook(module, input, output):
        attention_weights.append(output[1])  # (batch, n_heads, seq_len, seq_len)

    hook = model.cross_attn_layers[0].register_forward_hook(attention_hook)
    _ = model(frame_embs, query_emb)
    hook.remove()

    # Average attention across heads
    avg_attention = attention_weights[0].mean(dim=1).squeeze()  # (seq_len,)

    # Plot
    plt.figure(figsize=(14, 4))
    plt.plot(timestamps, avg_attention.cpu().numpy())
    plt.axvspan(gt_start, gt_end, alpha=0.2, color='green', label='Ground Truth')
    plt.axvspan(pred_start, pred_end, alpha=0.2, color='red', label='Prediction')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Cross-modal attention weight")
    plt.title(f'Query: "{query}"')
    plt.legend()
    plt.savefig(output_path)
```

Generate attention heatmaps for at minimum:

- 5 correct predictions (high IoU) → verify attention peaks at the right moment
- 5 failure cases (low IoU) → diagnose what the model was attending to instead

#### 5.2 Failure Mode Taxonomy

Manually annotate 50 failure cases with a failure type:

```
Failure Type              | Definition                                    | Expected %
--------------------------|-----------------------------------------------|------------
Off-by-one scene         | Predicted adjacent scene, not target          | ~35%
Correct center, wrong dur | Center is near GT but span is too wide/narrow | ~25%
Semantically close miss  | Predicted a similar-but-wrong event           | ~20%
Event not in video       | Model predicted location but event is absent  | ~10%
Complete miss            | Prediction has near-zero IoU                  | ~10%
```

Report this taxonomy. It tells you specifically what to improve in the next iteration.

---

### Eval 6: Latency and Computational Cost

```python
import time
import torch

def profile_inference(model, video_path, query, n_runs=20):
    """Profile end-to-end inference time for a given video length"""
    video_duration = get_duration(video_path)
    frames = extract_frames(video_path, fps=1)

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            result = model.predict(frames, query)
        latencies.append(time.perf_counter() - start)

    return {
        "video_duration_s": video_duration,
        "n_frames": len(frames),
        "p50_ms": np.percentile(latencies, 50) * 1000,
        "p95_ms": np.percentile(latencies, 95) * 1000,
        "frames_per_second": len(frames) / np.mean(latencies)
    }

# Profile across video lengths
for test_video in [short_5min, medium_30min, long_60min, very_long_120min]:
    profile = profile_inference(model, test_video, "test query")
    print(f"Duration: {profile['video_duration_s']/60:.0f}min | "
          f"Frames: {profile['n_frames']} | "
          f"p50: {profile['p50_ms']:.0f}ms | "
          f"p95: {profile['p95_ms']:.0f}ms")
```

Expected latency profile:

```
Video Duration | N Frames | p50 Latency | p95 Latency
---------------|----------|-------------|-------------
5 min          | 300      | ~800ms      | ~1.2s
30 min         | 1,800    | ~4s         | ~6s
60 min         | 3,600    | ~8s         | ~12s
120 min        | 7,200    | ~16s        | ~22s
```

Note: 120-minute video at 1fps = 7,200 frames through CLIP ViT-L/14. This is the practical upper limit on Colab free tier without frame subsampling.

#### 6.1 Throughput vs Accuracy at Different Sample Rates

```python
# Test whether reducing frame sampling rate saves time at acceptable accuracy cost
for fps in [0.25, 0.5, 1.0, 2.0]:
    results = evaluate_at_fps(model, qvhighlights_val, fps=fps)
    latency = profile_at_fps(model, sample_video, fps=fps)
    print(f"fps={fps}: R@1 IoU=0.5={results['r1_iou05']:.4f}, "
          f"latency_p50={latency['p50_ms']:.0f}ms")

# Expected: accuracy drops below 0.5fps, minimal gain above 2fps
# Use this to set the recommended fps for the deployed API
```

---

### Eval 7: Confidence Calibration

The model outputs a confidence score for event presence. Calibrate it.

```python
# Calibration: does confidence=0.9 mean the model is correct 90% of the time?
from sklearn.calibration import calibration_curve

confidences = [result['confidence'] for result in all_predictions]
# "Correct" = IoU with ground truth > 0.5
correct = [
    1 if temporal_iou(pred['start'], pred['end'], gt['start'], gt['end']) >= 0.5 else 0
    for pred, gt in zip(all_predictions, ground_truths)
]

fraction_correct, mean_confidence = calibration_curve(correct, confidences, n_bins=10)
ece = compute_ece(fraction_correct, mean_confidence)

print(f"Confidence ECE: {ece:.4f}")
# Target: ECE < 0.10 — confidence scores should be meaningful, not arbitrary
# Plot reliability diagram for the demo

# Critical threshold: find confidence below which the model should say
# "I cannot find this event" rather than return a low-quality prediction
low_confidence_recall = recall_at_k_iou(
    [p for p in predictions if p['confidence'] < 0.4],
    [g for g, p in zip(ground_truths, predictions) if p['confidence'] < 0.4],
    k=1, iou_threshold=0.5
)
print(f"R@1 IoU=0.5 for confidence < 0.4: {low_confidence_recall:.4f}")
# If this is < 0.2, setting confidence threshold at 0.4 is the right call
```

---

### Eval 8: Regression and Stability

#### 8.1 Regression Test Suite

Fixed set of 30 video-query pairs with known ground truth. Run before every deployment.

```python
REGRESSION_SUITE = [
    {
        "video": "tests/videos/lecture_30min.mp4",
        "query": "when the professor writes the equation on the board",
        "gt_start": 845.0,
        "gt_end": 872.0,
        "expected_min_iou": 0.4
    },
    {
        "video": "tests/videos/interview_15min.mp4",
        "query": "when the interviewer laughs",
        "gt_start": 312.0,
        "gt_end": 319.0,
        "expected_min_iou": 0.3
    },
    # ... 28 more
]

def run_regression(model, suite=REGRESSION_SUITE):
    failures = []
    for case in suite:
        result = model.predict(case['video'], case['query'])
        iou = temporal_iou(result['start'], result['end'], case['gt_start'], case['gt_end'])
        if iou < case['expected_min_iou']:
            failures.append({
                "query": case['query'],
                "predicted": (result['start'], result['end']),
                "ground_truth": (case['gt_start'], case['gt_end']),
                "iou": iou
            })

    print(f"Regression: {len(suite)-len(failures)}/{len(suite)} passed")
    return len(failures) == 0
```

#### 8.2 W&B Metric Logging

```python
import wandb
wandb.init(project="groundzero", name=f"eval_{version}")
wandb.log({
    "grounding/r1_iou05":           r1_iou05,
    "grounding/r1_iou07":           r1_iou07,
    "grounding/r5_iou05":           r5_iou05,
    "grounding/map_50_75":          map_score,
    "grounding/median_center_error": median_center_error,
    "generalization/charades_r1":   charades_r1,
    "generalization/actnet_r1":     actnet_r1,
    "latency/p50_30min_video_ms":   p50_30min,
    "latency/p95_30min_video_ms":   p95_30min,
    "calibration/confidence_ece":   confidence_ece,
    "ablation/no_temporal_conv":    ablation_no_temporal_conv,
    "ablation/no_cross_attn":       ablation_no_cross_attn,
    "ablation/no_contrastive":      ablation_no_contrastive,
})
```

---

## Evaluation Summary — Targets at a Glance

**Measure CLIP zero-shot baseline first. All targets below are relative to baseline.**

```
Component              | Metric                           | Target
-----------------------|----------------------------------|-------------------------------
Primary Grounding      | R@1 IoU=0.5 (QVHighlights)      | Baseline + 15pp minimum
                       | R@1 IoU=0.7 (QVHighlights)      | Baseline + 14pp minimum
                       | R@5 IoU=0.5 (QVHighlights)      | Baseline + 17pp minimum
                       | mAP@[0.5:0.75]                  | > 0.35
                       | Median center error              | < 8 seconds
                       |                                  |
Generalization         | R@1 IoU=0.5 (Charades-STA)      | > 0.35 zero-shot
                       | Long video perf drop             | < 15pp vs short video
                       |                                  |
Query Type             | Action queries R@1              | Highest category
                       | Abstract queries R@1            | Lowest (document honestly)
                       |                                  |
Ablation               | Each component contributes       | > 3pp drop when removed
                       |                                  |
Latency                | p50 for 30-min video            | < 5s
                       | p95 for 30-min video            | < 8s
                       |                                  |
Calibration            | Confidence ECE                  | < 0.10
                       |                                  |
Regression             | Fixed test suite                | 30/30 pass before deploy
```

---

## File Structure

```
groundzero-backend/
│
├── app/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py              # End-to-end prediction pipeline
│   │   ├── frame_extractor.py          # Video → sampled frames via ffmpeg
│   │   ├── visual_encoder.py           # CLIP ViT-L/14 + LoRA adapter encoding (batched)
│   │   ├── text_encoder.py             # CLIP text encoding + projection layer
│   │   ├── temporal_context.py         # Dilated 1D temporal conv + positional encoding
│   │   ├── cross_modal_transformer.py  # Cross-modal attention transformer (Q→frames)
│   │   ├── span_extraction.py          # Per-frame start/end scoring + confidence head
│   │   └── augmentation.py             # Temporal jitter, crop, speed perturbation
│   │
│   ├── models/
│   │   ├── groundzero_model.py         # Full model class (assembles all components)
│   │   └── checkpoints/                # Trained weights (pulled from HuggingFace Hub)
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── predict.py                  # POST /predict — video + query → timestamps
│   │   ├── attention.py               # POST /attention — return attention heatmap data
│   │   └── health.py                  # GET /health
│   │
│   ├── schema/
│   │   └── schemas.py                  # Pydantic models (PredictRequest, GroundingResult)
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── run_all_evals.py            # Master eval runner → W&B
│       ├── grounding/
│       │   ├── eval_recall_iou.py      # R@K at multiple IoU thresholds
│       │   ├── eval_map.py             # mAP@[0.5:0.75]
│       │   └── eval_displacement.py    # Center/boundary error analysis
│       ├── generalization/
│       │   ├── eval_charades.py        # Zero-shot on Charades-STA
│       │   └── eval_activitynet.py     # Long-video generalization
│       ├── query_analysis/
│       │   ├── eval_query_type.py      # Performance per query category
│       │   └── eval_query_length.py    # Performance by query token length
│       ├── ablation/
│       │   └── eval_ablation.py        # Component ablation study
│       ├── latency/
│       │   ├── eval_inference_time.py  # Latency by video length
│       │   └── eval_fps_tradeoff.py    # Accuracy vs fps sampling rate
│       ├── calibration/
│       │   └── eval_confidence_ece.py  # Confidence calibration
│       ├── visualization/
│       │   └── attention_heatmap.py    # Cross-modal attention visualizer
│       ├── regression/
│       │   ├── regression_suite.json   # 30 fixed test cases
│       │   └── run_regression.py       # CI regression runner
│       └── data/
│           ├── failure_taxonomy.json   # Annotated failure cases
│           └── query_categories.json   # Query type labels
│
├── config.py                           # Pydantic-settings (model path, fps, thresholds)
├── main.py                             # FastAPI app init, CORS, router registration
├── requirements.txt
├── .env.example
└── README.md
```

```
groundzero-training/   (this is actually notebook folder in backend repo)
│
├── 01_data_preparation.ipynb           # Download datasets, extract frames, build index
├── 02_baseline_measurement.ipynb       # Measure CLIP zero-shot baseline (run first)
├── 03_train_groundzero.ipynb           # Full training loop with W&B logging
├── 04_ablation_study.ipynb            # Train all ablation variants
├── 05_evaluation.ipynb                 # Run full eval suite, generate report
└── 06_export_to_hub.ipynb             # Push trained weights to HuggingFace Hub
```

---

## Tech Stack

```
# Frontend
Next.js 14 (App Router), TypeScript, Tailwind CSS
HTML5 video player for timestamp-seeking demo
Custom timeline visualisation component

# Backend
FastAPI, Uvicorn, Python 3.11
ffmpeg-python for video frame extraction
torch, torchvision for model inference

# ML Core
torch, transformers (CLIP from openai/clip-vit-large-patch14)
peft (LoRA adapters for CLIP fine-tuning)
einops (tensor manipulation for attention layers)

# Training (Colab only)
torch, transformers, datasets (HuggingFace)
peft (LoRA — same as ML Core)
wandb (experiment tracking)

# Evaluation
wandb
scikit-learn (calibration curves, metrics)
matplotlib (attention heatmaps, ablation plots)

# Deployment
Docker + docker-compose
HuggingFace Hub (model storage)
HuggingFace Spaces (deployment)
```

---
