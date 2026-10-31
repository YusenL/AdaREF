"""Per-recording adaptation of the reference editor with a frozen HR backbone."""

import copy

import numpy as np
import torch


def chunks(segment, block_length):
    """Half-overlapping blocks; each segment identifier denotes one contiguous run."""
    starts = []
    for value in np.unique(segment):
        indices = np.flatnonzero(segment == value)
        if len(indices) >= block_length:
            starts += list(range(indices[0], indices[-1] - block_length + 2,
                                 block_length // 2))
    return np.array(starts)


def adaptation_objective(model, spec, time, transition, source_probabilities,
                         block_length=64):
    """Return negative log predictive agreement and KL(current || source outputs).

    The first agreement term is omitted; its state update remains differentiable.
    """
    probabilities = model(spec, time, lstm_dropout=False).reshape(-1, block_length, 64)
    source_probabilities = source_probabilities.reshape(probabilities.shape)
    state = torch.full((probabilities.shape[0], 64), 1 / 64,
                       device=probabilities.device)
    log_agreement = []
    for step in range(block_length):
        prediction = state @ transition.T
        mass = (prediction * probabilities[:, step]).sum(1)
        log_agreement.append(torch.log(mass + 1e-12))
        state = prediction * probabilities[:, step] / mass[:, None]
    native = -torch.stack(log_agreement[1:], 1).mean()
    anchor = (probabilities * (torch.log(probabilities + 1e-12)
              - torch.log(source_probabilities + 1e-12))).sum(-1).mean()
    return native, anchor


def _source_predictions(model, spec, time, batch_size=2048):
    with torch.no_grad():
        return torch.cat([
            model(spec[start:start + batch_size], time[start:start + batch_size],
                  lstm_dropout=False)
            for start in range(0, len(spec), batch_size)
        ])


def adapt(model, record, transition, steps=100, kl_weight=3.0, lr=0.001,
          block_length=64, device="cpu", seed=0, max_blocks=8):
    """Copy the source model, adapt only its reference editor, and return the copy.

    The transition weights and source outputs stay fixed. Record arrays are float32;
    adaptation never reads HR labels. Nonfinite losses restore the preceding state.
    """
    if block_length < 2 or max_blocks < 1 or steps < 0:
        raise ValueError("block_length >= 2, max_blocks >= 1, and steps >= 0 are required")

    adapted = copy.deepcopy(model).to(device).eval()
    for parameter in adapted.parameters():
        parameter.requires_grad_(False)
    parameters = list(adapted.ref.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr)

    starts = chunks(record.segment, block_length)
    if len(starts) == 0:
        return adapted
    spec = torch.as_tensor(record.spec, device=device)
    time = torch.as_tensor(record.time, device=device)
    source_probabilities = _source_predictions(adapted, spec, time)
    transition = torch.as_tensor(transition, device=device)
    rng = np.random.default_rng(seed)
    previous_state = copy.deepcopy(adapted.state_dict())

    for _ in range(steps):
        selected = rng.choice(starts, min(max_blocks, len(starts)), replace=False)
        indices = torch.as_tensor(
            (selected[:, None] + np.arange(block_length)).ravel(), device=device)
        native, anchor = adaptation_objective(
            adapted, spec[indices], time[indices], transition,
            source_probabilities[indices], block_length)
        loss = native + kl_weight * anchor
        if not torch.isfinite(loss):
            adapted.load_state_dict(previous_state)
            break
        previous_state = copy.deepcopy(adapted.state_dict())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return adapted
