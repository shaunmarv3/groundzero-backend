"""
test_fixes.py — checks for the Session-35 fixes (CPU, tiny model, no SigLIP download).

    python scripts/test_fixes.py

Uses GroundZeroModel(use_cache=True, load_text_encoder=False) with d_model=32, so
nothing is downloaded and it runs in seconds. Each test prints PASS/FAIL.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.groundzero_model import GroundZeroModel, resolve_model_cfg
from app.pipeline.span_extraction import decode_best_span
from app.training.losses import combined_loss

D, HEADS, LAYERS = 32, 4, 2
torch.manual_seed(0)


def tiny(**kw):
    m = GroundZeroModel(d_model=D, n_heads=HEADS, n_layers=LAYERS, device="cpu",
                        use_cache=True, load_text_encoder=False, **kw)
    return m.eval()


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def test_padding_invariance(word_level):
    """
    Fix 4: a video's real-frame outputs must NOT change when it is batched next to a
    longer video (i.e. zero-padded). Without the mask they do.
    """
    print(f"\n=== padding mask: real frames unaffected by padding (word_level={word_level}) ===")
    m = tiny(word_level=word_level)
    n_short, n_long = 10, 16
    short = torch.randn(1, n_short, D)
    ts_short = torch.linspace(0, 1, n_short)
    if word_level:
        q = torch.randn(1, 6, D); qm = torch.tensor([[True] * 4 + [False] * 2])
    else:
        q = torch.randn(1, 1, D); qm = None

    with torch.no_grad():
        s_alone, e_alone, _ = m(short, ts_short.unsqueeze(0), q,
                                frame_mask=torch.ones(1, n_short, dtype=torch.bool), query_mask=qm)

        # same video, zero-padded to n_long next to a longer one
        frames = torch.zeros(2, n_long, D); frames[0, :n_short] = short[0]; frames[1] = torch.randn(n_long, D)
        ts = torch.ones(2, n_long); ts[0, :n_short] = ts_short; ts[1] = torch.linspace(0, 1, n_long)
        mask = torch.zeros(2, n_long, dtype=torch.bool); mask[0, :n_short] = True; mask[1] = True
        qb = q.expand(2, -1, -1)
        qmb = qm.expand(2, -1) if qm is not None else None

        s_m, e_m, _ = m(frames, ts, qb, frame_mask=mask, query_mask=qmb)
        s_nm, _, _ = m(frames, ts, qb, frame_mask=None, query_mask=qmb)

    ok = check("masked: real-frame start/end logits identical to the unpadded run",
               torch.allclose(s_alone[0], s_m[0, :n_short], atol=1e-4)
               and torch.allclose(e_alone[0], e_m[0, :n_short], atol=1e-4))
    ok &= check("unmasked: padding DOES change them (the bug)",
                not torch.allclose(s_alone[0], s_nm[0, :n_short], atol=1e-4))
    ok &= check("pad positions get logit -1e4", bool((s_m[0, n_short:] <= -1e4 + 1).all()))
    s_idx, e_idx = decode_best_span(s_m[0], e_m[0])
    ok &= check(f"decoded span ({s_idx}, {e_idx}) lies inside the real frames",
                s_idx < n_short and e_idx < n_short)
    return ok


def test_gelu():
    print("\n=== GELU: changes the function, adds no parameters ===")
    torch.manual_seed(1); a = tiny(use_gelu=False)
    torch.manual_seed(1); b = tiny(use_gelu=True)
    x, ts, q = torch.randn(1, 12, D), torch.linspace(0, 1, 12), torch.randn(1, 1, D)
    with torch.no_grad():
        oa, ob = a.temporal_context(x, ts), b.temporal_context(x, ts)
    ok = check("same state_dict keys (old best.pt still loads)",
               set(a.state_dict()) == set(b.state_dict()))
    ok &= check("different outputs with identical weights", not torch.allclose(oa, ob))
    return ok


def test_word_level_shapes():
    print("\n=== word-level cross-attention: shapes + per-frame content ===")
    m = tiny(word_level=True)
    B, N, L = 2, 12, 8
    frames, ts = torch.randn(B, N, D), torch.linspace(0, 1, N)
    words = torch.randn(B, L, D)
    wmask = torch.ones(B, L, dtype=torch.bool); wmask[:, 5:] = False
    with torch.no_grad():
        s, e, c, x = m(frames, ts, words, query_mask=wmask, return_features=True)
        # changing only a PAD token must not change anything
        words2 = words.clone(); words2[:, 6] += 100.0
        s2, _, _ = m(frames, ts, words2, query_mask=wmask)
    ok = check(f"outputs {tuple(s.shape)} {tuple(e.shape)} {tuple(c.shape)} {tuple(x.shape)}",
               s.shape == (B, N) and e.shape == (B, N) and c.shape == (B,) and x.shape == (B, N, D))
    ok &= check("masked (pad) word tokens have no effect", torch.allclose(s, s2, atol=1e-4))
    return ok


def test_contrastive_gradient():
    """
    Fix 1: the original call fed the cached frame embeddings (no grad path) to the
    contrastive term → it is a constant. The fixed call feeds the model's output.
    """
    print("\n=== contrastive: dead on cached inputs, alive on grounded features ===")
    m = tiny().train()
    B, N = 3, 30
    frame_embs = torch.randn(B, N, D)            # like the cache: requires_grad=False
    query = torch.randn(B, 1, D)                 # frozen text: requires_grad=False
    ts = torch.linspace(0, 1, N)
    gs, ge = torch.tensor([3, 10, 20]), torch.tensor([8, 15, 25])

    s, e, c, grounded = m(frame_embs, ts, query, return_features=True)
    _, _, _, cont_old = combined_loss(s, e, gs, ge, query, frame_embs)
    _, _, _, cont_new = combined_loss(s, e, gs, ge, query, grounded)

    ok = check(f"original: cont={cont_old.item():.3f} has requires_grad={cont_old.requires_grad} "
               "(constant → trains nothing)", not cont_old.requires_grad)
    cont_new.backward()
    g = sum(p.grad.abs().sum().item() for p in m.cross_modal.parameters() if p.grad is not None)
    ok &= check(f"fixed: cont={cont_new.item():.3f} sends gradient into cross_modal (|grad| sum {g:.3e})",
                cont_new.requires_grad and g > 0)
    return ok


def test_cfg_defaults():
    print("\n=== old checkpoints: no model_cfg → original design ===")
    return check(f"resolve_model_cfg({{}}) = {resolve_model_cfg({})}",
                 not any(resolve_model_cfg({}).values()))


if __name__ == "__main__":
    results = [
        test_padding_invariance(word_level=False),
        test_padding_invariance(word_level=True),
        test_gelu(),
        test_word_level_shapes(),
        test_contrastive_gradient(),
        test_cfg_defaults(),
    ]
    print(f"\n{sum(results)}/{len(results)} test groups passed")
