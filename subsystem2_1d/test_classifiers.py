# subsystem2_1d/test_classifiers.py
# =========================================================
# Subsystem 2 -- Architecture Verification Test Suite
# Person 2 | feature/1d-model-p2
#
# Run from the repo root:
#   python subsystem2_1d/test_classifiers.py
#
# Verifies:
#   1. Instantiation of both model architectures
#   2. Forward pass output shape contract [B, num_classes]
#   3. Intermediate tensor shapes along each forward path
#   4. Trainable parameter counts (CNN1D > Hybrid expected)
#   5. __init__.py public API import test
#   6. build_model() factory function
#   7. PositionalEncoding properties (norm preservation, shape)
# =========================================================

import sys
import os

# Force UTF-8 on Windows cp1252 terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure repo root on sys.path for `from subsystem2_1d import ...`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import math
import torch
import torch.nn as nn

# =========================================================
# Terminal colour helpers
# =========================================================
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
INFO = f"{CYAN}[INFO]{RESET}"

_results = {"passed": 0, "failed": 0}


def _report(label: str, ok: bool, detail: str = "") -> None:
    if ok:
        _results["passed"] += 1
        print(f"  {PASS}  {label}")
    else:
        _results["failed"] += 1
        print(f"  {FAIL}  {label}")
    if detail:
        print(f"         {detail}")


# =========================================================
# Helpers
# =========================================================

def _make_batch(batch_size: int = 32) -> torch.Tensor:
    """Create a synthetic IQ batch: (B, 2, 128) float32."""
    return torch.randn(batch_size, 2, 128)


def _hook_shapes(model: nn.Module, x: torch.Tensor) -> dict:
    """
    Run a forward pass with register_forward_hook to capture the
    output shape of every named module. Returns {name: shape}.
    """
    shapes = {}
    hooks  = []

    for name, module in model.named_modules():
        if name == "":         # skip the root module itself
            continue
        def _make_hook(n):
            def _hook(mod, inp, out):
                if isinstance(out, torch.Tensor):
                    shapes[n] = tuple(out.shape)
            return _hook
        hooks.append(module.register_forward_hook(_make_hook(name)))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return shapes


# =========================================================
# Test Suite
# =========================================================

def run_tests():
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  Subsystem 2 -- 1D Classifier Architecture Verification{RESET}")
    print(f"{BOLD}  feature/1d-model-p2 | Person 2{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")

    BATCH       = 32
    NUM_CLASSES = 11
    x           = _make_batch(BATCH)

    # ----------------------------------------------------------
    # Test 1: Import from classifiers_1d directly
    # ----------------------------------------------------------
    print(f"\n{CYAN}[T-1]  Direct module import{RESET}")
    try:
        from subsystem2_1d.classifiers_1d import (
            CNN1DClassifier,
            CNNTransformerHybrid,
            PositionalEncoding,
            build_model,
        )
        _report("from subsystem2_1d.classifiers_1d import CNN1DClassifier",     True)
        _report("from subsystem2_1d.classifiers_1d import CNNTransformerHybrid", True)
        _report("from subsystem2_1d.classifiers_1d import PositionalEncoding",   True)
        _report("from subsystem2_1d.classifiers_1d import build_model",          True)
    except ImportError as e:
        _report("Direct module import succeeded", False, str(e))
        print(f"\n{RED}Cannot continue -- import failed.{RESET}")
        return

    # ----------------------------------------------------------
    # Test 2: __init__.py public API imports
    # ----------------------------------------------------------
    print(f"\n{CYAN}[T-2]  Public API via subsystem2_1d package __init__.py{RESET}")
    try:
        from subsystem2_1d import CNN1DClassifier      as _c1
        from subsystem2_1d import CNNTransformerHybrid as _c2
        _report("from subsystem2_1d import CNN1DClassifier",       True)
        _report("from subsystem2_1d import CNNTransformerHybrid",  True)
    except ImportError as e:
        _report("__init__.py re-exports both model classes", False, str(e))

    # ----------------------------------------------------------
    # Test 3: Model instantiation
    # ----------------------------------------------------------
    print(f"\n{CYAN}[T-3]  Model instantiation smoke test{RESET}")
    try:
        cnn1d   = CNN1DClassifier(num_classes=NUM_CLASSES)
        _report(f"CNN1DClassifier(num_classes={NUM_CLASSES}) instantiated", True)
    except Exception as e:
        _report("CNN1DClassifier instantiation", False, str(e))
        cnn1d = None

    try:
        hybrid  = CNNTransformerHybrid(num_classes=NUM_CLASSES)
        _report(f"CNNTransformerHybrid(num_classes={NUM_CLASSES}) instantiated", True)
    except Exception as e:
        _report("CNNTransformerHybrid instantiation", False, str(e))
        hybrid = None

    if cnn1d is None or hybrid is None:
        print(f"\n{RED}Cannot continue -- one or more models failed to instantiate.{RESET}")
        return

    # ----------------------------------------------------------
    # Test 4: Forward pass output shape
    # ----------------------------------------------------------
    print(f"\n{CYAN}[T-4]  Forward pass output shapes  (batch={BATCH}, input=[{BATCH}, 2, 128]){RESET}")

    cnn1d.eval()
    hybrid.eval()

    with torch.no_grad():
        out_cnn = cnn1d(x)
        out_hyb = hybrid(x)

    expected_shape = torch.Size([BATCH, NUM_CLASSES])

    _report(
        f"CNN1DClassifier       output shape == [{BATCH}, {NUM_CLASSES}]"
        f"  (got {list(out_cnn.shape)})",
        out_cnn.shape == expected_shape,
    )
    _report(
        f"CNNTransformerHybrid  output shape == [{BATCH}, {NUM_CLASSES}]"
        f"  (got {list(out_hyb.shape)})",
        out_hyb.shape == expected_shape,
    )
    _report(
        "CNN1DClassifier output dtype is float32",
        out_cnn.dtype == torch.float32,
        f"got {out_cnn.dtype}",
    )
    _report(
        "CNNTransformerHybrid output dtype is float32",
        out_hyb.dtype == torch.float32,
        f"got {out_hyb.dtype}",
    )

    # ----------------------------------------------------------
    # Test 5: Variable batch sizes
    # ----------------------------------------------------------
    print(f"\n{CYAN}[T-5]  Variable batch size robustness{RESET}")

    for bs in [1, 16, 64]:
        x_var = _make_batch(bs)
        with torch.no_grad():
            o_c = cnn1d(x_var)
            o_h = hybrid(x_var)
        exp = torch.Size([bs, NUM_CLASSES])
        _report(
            f"batch={bs:<3}  CNN1D:  {list(o_c.shape)}",
            o_c.shape == exp,
        )
        _report(
            f"batch={bs:<3}  Hybrid: {list(o_h.shape)}",
            o_h.shape == exp,
        )

    # ----------------------------------------------------------
    # Test 6: Intermediate shapes -- CNN1DClassifier
    # ----------------------------------------------------------
    print(f"\n{CYAN}[T-6]  CNN1DClassifier -- intermediate tensor shapes{RESET}")

    shapes_cnn = _hook_shapes(cnn1d, x)

    # Expected shapes after each MaxPool block
    _expected_cnn = {
        "conv_blocks.0.block": (BATCH, 64,  64),
        "conv_blocks.1.block": (BATCH, 128, 32),
        "conv_blocks.2.block": (BATCH, 256, 16),
        "conv_blocks.3.block": (BATCH, 256,  8),
    }
    for key, exp_shape in _expected_cnn.items():
        got = shapes_cnn.get(key, "NOT_CAPTURED")
        _report(
            f"After {key.split('.')[-2]+key.split('.')[-1]}: {list(exp_shape)}  (got {list(got) if isinstance(got, tuple) else got})",
            got == exp_shape,
        )

    flat_shape = shapes_cnn.get("classifier.0")  # nn.Flatten
    _report(
        f"After Flatten: [{BATCH}, 2048]  (got {list(flat_shape) if flat_shape else 'N/A'})",
        flat_shape == (BATCH, 2048),
    )

    # ----------------------------------------------------------
    # Test 7: Intermediate shapes -- CNNTransformerHybrid
    # ----------------------------------------------------------
    print(f"\n{CYAN}[T-7]  CNNTransformerHybrid -- intermediate tensor shapes{RESET}")

    # Manually trace key shapes
    with torch.no_grad():
        feat    = hybrid.cnn_frontend(x)
        tokens  = feat.transpose(1, 2)
        tokens_pe = hybrid.pos_encoder(tokens)
        encoded = hybrid.transformer_encoder(tokens_pe)
        context = encoded.mean(dim=1)

    _report(f"CNN frontend output: [{BATCH}, 128, 32]  (got {list(feat.shape)})",
            feat.shape    == torch.Size([BATCH, 128, 32]))
    _report(f"After transpose:     [{BATCH}, 32,  128] (got {list(tokens.shape)})",
            tokens.shape  == torch.Size([BATCH, 32, 128]))
    _report(f"After pos encoding:  [{BATCH}, 32,  128] (got {list(tokens_pe.shape)})",
            tokens_pe.shape == torch.Size([BATCH, 32, 128]))
    _report(f"After Transformer:   [{BATCH}, 32,  128] (got {list(encoded.shape)})",
            encoded.shape == torch.Size([BATCH, 32, 128]))
    _report(f"After global avg pool:[{BATCH}, 128]     (got {list(context.shape)})",
            context.shape == torch.Size([BATCH, 128]))

    # ----------------------------------------------------------
    # Test 8: PositionalEncoding properties
    # ----------------------------------------------------------
    print(f"\n{CYAN}[T-8]  PositionalEncoding -- properties{RESET}")

    pe  = PositionalEncoding(d_model=128, max_len=512, dropout=0.0)
    tok = torch.zeros(1, 32, 128)    # zero input so output IS the PE itself

    pe.eval()
    with torch.no_grad():
        out_pe = pe(tok)

    _report("PE output shape (1, 32, 128)",
            out_pe.shape == torch.Size([1, 32, 128]))

    # PE should change at least some values (non-trivial encoding)
    _report("PE modifies the input (non-zero encoding injected)",
            not torch.allclose(out_pe, tok))

    # PE values should be bounded in [-1, 1] since they are sin/cos
    pe_vals = out_pe[0]   # (32, 128)
    _report(
        f"PE values bounded in [-1, 1]  (min={pe_vals.min():.4f}, max={pe_vals.max():.4f})",
        pe_vals.min() >= -1.01 and pe_vals.max() <= 1.01,
    )

    # ----------------------------------------------------------
    # Test 9: Trainable parameter counts & capacity comparison
    # ----------------------------------------------------------
    print(f"\n{CYAN}[T-9]  Trainable parameter footprint{RESET}")

    print(f"\n  {INFO} Parameter counts:")
    print(f"  {'':>4}", end="")
    n_cnn   = cnn1d.get_param_count()
    print(f"  {'':>4}", end="")
    n_hyb   = hybrid.get_param_count()
    print()

    _report(
        f"CNN1DClassifier has more params than CNNTransformerHybrid"
        f"  ({n_cnn:,} > {n_hyb:,})",
        n_cnn > n_hyb,
    )
    _report("CNN1DClassifier param count > 0",       n_cnn  > 0)
    _report("CNNTransformerHybrid param count > 0",  n_hyb  > 0)

    # Sanity range checks (ballpark expected sizes)
    _report(
        f"CNN1D param count in expected range [500k, 3M]  (got {n_cnn:,})",
        500_000 < n_cnn < 3_000_000,
    )
    _report(
        f"Hybrid param count in expected range [100k, 1M]  (got {n_hyb:,})",
        100_000 < n_hyb < 1_000_000,
    )

    # ----------------------------------------------------------
    # Test 10: build_model() factory
    # ----------------------------------------------------------
    print(f"\n{CYAN}[T-10] build_model() factory function{RESET}")

    m1 = build_model("cnn1d",           num_classes=11)
    m2 = build_model("cnn_transformer", num_classes=11)
    _report("build_model('cnn1d') returns CNN1DClassifier",
            isinstance(m1, CNN1DClassifier))
    _report("build_model('cnn_transformer') returns CNNTransformerHybrid",
            isinstance(m2, CNNTransformerHybrid))

    try:
        build_model("nonexistent_model")
        _report("build_model raises KeyError for unknown model", False,
                "Expected KeyError but none raised!")
    except KeyError:
        _report("build_model raises KeyError for unknown model", True)

    # ----------------------------------------------------------
    # Test 11: No gradient leak into input (in eval mode)
    # ----------------------------------------------------------
    print(f"\n{CYAN}[T-11] Gradient isolation check{RESET}")

    x_grad = _make_batch(4).requires_grad_(True)
    cnn1d.train()
    out = cnn1d(x_grad).sum()
    out.backward()
    _report("Gradients flow back to input (model is differentiable)",
            x_grad.grad is not None)
    _report("Input grad shape == input shape",
            x_grad.grad.shape == x_grad.shape if x_grad.grad is not None else False)

    # ----------------------------------------------------------
    # Final Summary
    # ----------------------------------------------------------
    p = _results["passed"]
    f = _results["failed"]

    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"  {GREEN}Passed : {p}{RESET}")
    print(f"  {RED}Failed : {f}{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")

    if f > 0:
        print(f"\n{RED}{BOLD}  [!!] {f} test(s) FAILED.{RESET}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}  [OK] All {p} architecture tests PASSED!{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    run_tests()
