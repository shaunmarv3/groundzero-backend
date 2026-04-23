"""
scripts/siglip_demo.py — SigLIP 2 So400m Hands-On Demo
=======================================================
Demonstrates what SigLIP 2 actually does:
  - Load the model (downloads ~1.6GB on first run, cached after)
  - Encode sample images with the visual tower → 1152-d vectors
  - Encode text queries with the text tower → 1152-d vectors
  - Compute cosine similarity between every image-query pair
  - Print a table showing which image matched which query

Run from groundzero-backend/:
    python scripts/siglip_demo.py

On first run: downloads SigLIP 2 So400m from HuggingFace (~1.6GB).
Subsequent runs: loads from local cache instantly.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModel

# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #
MODEL_ID = "google/siglip2-so400m-patch14-384"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ #
#  Sample queries to test                                              #
# ------------------------------------------------------------------ #
QUERIES = [
    "a person writing on a whiteboard",
    "someone laughing",
    "a dog running in a park",
    "a car on a highway",
    "a person sitting at a desk with a computer",
]

# ------------------------------------------------------------------ #
#  Load model                                                          #
# ------------------------------------------------------------------ #
def load_model():
    print(f"\n{'='*60}")
    print(f"  Loading SigLIP 2 So400m")
    print(f"  Model: {MODEL_ID}")
    print(f"  Device: {DEVICE.upper()}")
    print(f"  (First run downloads ~1.6GB — this is cached after)")
    print(f"{'='*60}\n")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model     = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model     = model.to(DEVICE).eval()

    # Count parameters
    total  = sum(p.numel() for p in model.parameters())
    visual = sum(p.numel() for p in model.vision_model.parameters())
    text   = sum(p.numel() for p in model.text_model.parameters())
    print(f"  Parameters:")
    print(f"    Total:         {total/1e6:.1f}M")
    print(f"    Visual tower:  {visual/1e6:.1f}M")
    print(f"    Text tower:    {text/1e6:.1f}M\n")

    return processor, model


# ------------------------------------------------------------------ #
#  Encode images                                                       #
# ------------------------------------------------------------------ #
def encode_images(processor, model, images: list[Image.Image]) -> torch.Tensor:
    """
    Pass images through the visual tower.
    Returns: Tensor of shape (N_images, 1152) — L2-normalised embeddings.
    """
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # L2 normalise → unit vectors (so cosine sim = dot product)
    image_features = F.normalize(image_features, dim=-1)

    print(f"  Visual tower output shape: {image_features.shape}")
    print(f"  (batch_size={image_features.shape[0]}, embedding_dim={image_features.shape[1]})\n")
    return image_features


# ------------------------------------------------------------------ #
#  Encode text queries                                                  #
# ------------------------------------------------------------------ #
def encode_queries(processor, model, queries: list[str]) -> torch.Tensor:
    """
    Pass text queries through the text tower.
    Returns: Tensor of shape (N_queries, 1152) — L2-normalised embeddings.
    """
    inputs = processor(text=queries, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    text_features = F.normalize(text_features, dim=-1)

    print(f"  Text tower output shape:   {text_features.shape}")
    print(f"  (batch_size={text_features.shape[0]}, embedding_dim={text_features.shape[1]})\n")
    return text_features


# ------------------------------------------------------------------ #
#  Similarity table                                                    #
# ------------------------------------------------------------------ #
def print_similarity_table(
    image_features: torch.Tensor,
    text_features:  torch.Tensor,
    image_names:    list[str],
    queries:        list[str],
):
    """
    Print cosine similarity between every image-query pair.
    Cosine similarity: 1.0 = identical direction, 0.0 = unrelated, -1.0 = opposite.
    For SigLIP 2: >0.25 is a good match, <0.15 is a poor match.
    """
    # (N_images, N_queries)
    sim_matrix = (image_features @ text_features.T).cpu().float()

    print(f"{'='*60}")
    print("  COSINE SIMILARITY TABLE")
    print(f"  (higher = better match, range: -1 to 1)")
    print(f"{'='*60}")

    # Column headers
    col_w = 36
    print(f"\n  {'Image':<20}", end="")
    for q in queries:
        short = q[:col_w] + "…" if len(q) > col_w else q
        print(f"  {short:<{col_w}}", end="")
    print()
    print(f"  {'-'*20}", end="")
    for _ in queries:
        print(f"  {'-'*col_w}", end="")
    print()

    # Rows
    for i, name in enumerate(image_names):
        print(f"  {name:<20}", end="")
        for j in range(len(queries)):
            score = sim_matrix[i, j].item()
            marker = " ✅" if score == sim_matrix[i].max().item() else "   "
            print(f"  {score:>+.4f}{marker:<{col_w-9}}", end="")
        print()

    print()

    # Best match per image
    print(f"{'='*60}")
    print("  BEST MATCH PER IMAGE")
    print(f"{'='*60}")
    for i, name in enumerate(image_names):
        best_j    = sim_matrix[i].argmax().item()
        best_score = sim_matrix[i, best_j].item()
        print(f"  {name:<20} → \"{queries[best_j]}\"")
        print(f"  {'':20}   score: {best_score:+.4f}")
    print()

    # Best match per query
    print(f"{'='*60}")
    print("  BEST MATCHING IMAGE PER QUERY")
    print(f"{'='*60}")
    for j, q in enumerate(queries):
        best_i    = sim_matrix[:, j].argmax().item()
        best_score = sim_matrix[best_i, j].item()
        print(f"  \"{q}\"")
        print(f"  → Best image: {image_names[best_i]:<20}  score: {best_score:+.4f}\n")


# ------------------------------------------------------------------ #
#  Generate or load sample images                                      #
# ------------------------------------------------------------------ #
def get_sample_images() -> tuple[list[Image.Image], list[str]]:
    """
    Try to find any images in the project directory.
    If none found, generate simple colored placeholder images.
    """
    # Search for any jpg/png in the project
    roots = [
        Path("d:/groundzero"),
        Path("d:/groundzero/groundzero-frontend/public"),
    ]
    found = []
    for root in roots:
        if root.exists():
            found += list(root.glob("**/*.jpg"))[:3]
            found += list(root.glob("**/*.png"))[:3]
            if found:
                break

    if found:
        found = found[:5]  # max 5
        images = [Image.open(p).convert("RGB") for p in found]
        names  = [p.name for p in found]
        print(f"  Found {len(images)} image(s) in project directory.")
        for n in names:
            print(f"    - {n}")
        print()
        return images, names

    # No images found — generate simple colored placeholders
    print("  No images found in project. Generating solid-color placeholders.")
    print("  (Replace with real images by putting them in groundzero-frontend/public/)\n")

    colors = [
        ("red_bg",    (220, 50,  50)),
        ("green_bg",  (50,  180, 80)),
        ("blue_bg",   (50,  100, 220)),
        ("gray_bg",   (180, 180, 180)),
        ("dark_bg",   (30,  30,  30)),
    ]
    images = [Image.new("RGB", (384, 384), color=c) for _, c in colors]
    names  = [n for n, _ in colors]
    return images, names


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    print("\n" + "="*60)
    print("  GroundZero — SigLIP 2 So400m Demo")
    print("  Phase 0.4 Output: see embeddings + similarity in action")
    print("="*60)

    # 1. Load model
    processor, model = load_model()

    # 2. Get images
    print("─"*60)
    print("  LOADING IMAGES")
    print("─"*60)
    images, image_names = get_sample_images()

    # 3. Encode images
    print("─"*60)
    print("  ENCODING IMAGES (visual tower)")
    print("─"*60)
    image_features = encode_images(processor, model, images)

    # 4. Encode queries
    print("─"*60)
    print("  ENCODING QUERIES (text tower)")
    print("─"*60)
    text_features = encode_queries(processor, model, QUERIES)

    # 5. Similarity table
    print("─"*60)
    print()
    print_similarity_table(image_features, text_features, image_names, QUERIES)

    # 6. Key takeaway
    print("="*60)
    print("  WHAT YOU JUST SAW:")
    print()
    print("  1. Visual tower:  image  → 1152 numbers (a point in space)")
    print("  2. Text tower:    text   → 1152 numbers (a point in space)")
    print("  3. Cosine sim:    how close those two points are")
    print()
    print("  This is the ZERO-SHOT BASELINE — no training, just SigLIP 2.")
    print("  For temporal grounding: do this for every video frame,")
    print("  pick the frame with highest similarity → our starting point.")
    print()
    print("  Problem: only ~35% accurate. That's what we fix by building")
    print("  the Temporal Context Module + Cross-Modal Transformer + Span Head.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
