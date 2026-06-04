"""
download_model.py — Fetch the trained GroundZero head weights from HuggingFace.
==============================================================================
The model was trained with the feature-caching recipe (--use_cache), so the
checkpoint contains ONLY the trainable head (TemporalContext + CrossModalTransformer
+ SpanExtractionHead) under the key "trainable_state". The frozen SigLIP 2 backbone
is NOT in here — it is downloaded separately from `google/siglip2-so400m-patch14-384`
the first time the model loads.

Usage:
    python scripts/download_model.py                 # downloads best.pt -> models/
    python scripts/download_model.py --file latest.pt  # also grab optimizer state
    python scripts/download_model.py --inspect       # just print what's inside best.pt

Run from the backend root (groundzero-backend/).
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "shaunmarvell/qvhighlights-model"
BACKEND_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_ROOT / "models"


def download(repo: str, filename: str, token: str | None) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {filename} from {repo} -> {MODELS_DIR} ...")
    local = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="model",
        local_dir=str(MODELS_DIR),
        token=token or None,
    )
    p = Path(local)
    print(f"Done: {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    return p


def inspect(path: Path) -> None:
    """Print the top-level keys + the names/shapes inside trainable_state."""
    import torch

    print(f"\nInspecting {path} ...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        print("Top-level keys:", list(ckpt.keys()))
        for k, v in ckpt.items():
            if k == "trainable_state" and isinstance(v, dict):
                print(f"\ntrainable_state: {len(v)} tensors")
                # group by top-level module prefix so we can see what loaded
                prefixes: dict[str, int] = {}
                for name in v:
                    head = name.split(".")[0]
                    prefixes[head] = prefixes.get(head, 0) + 1
                for head, n in sorted(prefixes.items()):
                    print(f"  {head:<20} {n} tensors")
            elif not hasattr(v, "shape"):
                print(f"  {k}: {v!r}")
    else:
        print("Checkpoint is not a dict:", type(ckpt))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF model repo id")
    ap.add_argument("--file", default="best.pt", help="checkpoint filename to download")
    ap.add_argument("--token", default="", help="HF token (only if repo is private)")
    ap.add_argument("--inspect", action="store_true",
                    help="inspect the (already-downloaded) checkpoint instead of downloading")
    args = ap.parse_args()

    target = MODELS_DIR / args.file
    if args.inspect:
        if not target.exists():
            raise SystemExit(f"{target} not found — download it first.")
        inspect(target)
        return

    path = download(args.repo, args.file, args.token)
    inspect(path)


if __name__ == "__main__":
    main()
