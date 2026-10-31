"""Exercise the core computations with synthetic inputs and random model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .adaptation import adapt, chunks
from .model import AdaREF, BinnedRegressionLoss, fit_transition, forward_prior


def run(*, config=None, steps=2, windows=96, seed=0, device="cpu"):
    settings = {
        "steps": 100, "kl_weight": 3.0, "learning_rate": 1e-3,
        "block_length": 64, "max_blocks": 8, "seed": 0, "objective": "temporal",
    }
    if config is not None:
        settings.update(json.loads(Path(config).read_text())["adaptation"])
    if settings["objective"] != "temporal":
        raise ValueError("The review package implements the paper's temporal objective")
    if steps < 1 or windows < settings["block_length"]:
        raise ValueError("Use at least one update and enough windows for one complete block")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    model = AdaREF().to(device).eval()
    record = SimpleNamespace(
        spec=rng.normal(size=(windows, 7, 64, 2)).astype(np.float32),
        time=rng.normal(size=(windows, 1280, 1)).astype(np.float32),
        segment=np.zeros(windows, dtype=np.int64),
    )
    waveform = torch.as_tensor(record.time, device=device)
    spectra = torch.as_tensor(record.spec, device=device)
    labels = torch.as_tensor(rng.uniform(50, 150, size=windows), dtype=torch.float32, device=device)
    source_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    probabilities = model(spectra, waveform, lstm_dropout=False)
    source_loss = BinnedRegressionLoss().to(device)(probabilities, labels)
    source_loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError("Source-loss gradient check failed")
    editor_gradient = sum(float(p.grad.abs().sum()) for p in model.ref.parameters() if p.grad is not None)
    if editor_gradient <= 0:
        raise AssertionError("The reference editor received no source-loss gradient")
    model.zero_grad(set_to_none=True)

    ordered_hr = 90 + 10 * np.sin(np.arange(256) / 15) + rng.normal(0, .3, size=256)
    transition = fit_transition([ordered_hr.astype(np.float32)])
    transition_before = transition.copy()
    adapted = adapt(
        model, record, transition, steps=steps, kl_weight=settings["kl_weight"],
        lr=settings["learning_rate"], block_length=settings["block_length"],
        device=device, seed=settings["seed"], max_blocks=settings["max_blocks"],
    )
    changed = []
    for name, tensor in adapted.state_dict().items():
        if not torch.equal(tensor, source_state[name]):
            changed.append(name)
            if not name.startswith("ref."):
                raise AssertionError(f"EDTA changed a frozen parameter or buffer: {name}")
        if not torch.equal(model.state_dict()[name], source_state[name]):
            raise AssertionError("EDTA changed the source model")
    if not changed or not np.array_equal(transition, transition_before):
        raise AssertionError("EDTA update or fixed-transition check failed")
    with torch.no_grad():
        final = adapted(spectra, waveform, lstm_dropout=False).cpu().numpy()
    decoded, _ = forward_prior(final, transition, segment=record.segment)
    if not np.isfinite(decoded).all() or not np.allclose(decoded.sum(axis=1), 1, atol=1e-5):
        raise FloatingPointError("Decoded-posterior check failed")
    return {
        "scope": "Synthetic core-method verification; these are not paper performance results.",
        "pretrained_weights": False,
        "windows": windows,
        "editor_parameters": sum(p.numel() for p in model.ref.parameters()),
        "source_loss": float(source_loss.detach()),
        "source_editor_gradient_l1": editor_gradient,
        "edta_demo_steps": steps,
        "edta_paper_steps": settings["steps"],
        "available_blocks": len(chunks(record.segment, settings["block_length"])),
        "changed_editor_tensors": len(changed),
        "source_and_backbone_unchanged": True,
        "transition_unchanged": True,
        "posterior_normalized": True,
        "device": str(device),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="AdaREF core-method verification on synthetic inputs")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--steps", type=int, default=2, help="Demo updates; the paper uses 100")
    parser.add_argument("--windows", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = run(config=args.config, steps=args.steps, windows=args.windows,
                 seed=args.seed, device=args.device)
    print(json.dumps(result, indent=2, allow_nan=False))
