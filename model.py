"""AdaREF: reference-editing fusion and fixed heart-rate dynamics.

BeliefPPG architecture, source-loss and HR-dynamics routines are adapted from
https://github.com/eth-siplab/BeliefPPG
commit 751cf56f30244d715fdda424cd94f91817beac44.
The upstream MIT notice follows.

MIT License

Copyright (c) 2023 ETH Sensing, Interaction & Perception Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def bpm_bins(dim=64, min_hz=0.5, max_hz=3.5):
    """Uniformly spaced HR-bin left edges in BPM."""
    return min_hz * 60 + np.arange(dim) * (max_hz - min_hz) * 60 / dim


class PositionalEncoding(nn.Module):
    def __init__(self, length, channels):
        super().__init__()
        depth = channels / 2
        angles = np.arange(length)[:, None] / 10000 ** (np.arange(depth)[None] / depth)
        encoding = np.concatenate([np.sin(angles), np.cos(angles)], -1).astype(np.float32)
        self.register_buffer('encoding', torch.from_numpy(encoding), persistent=False)
        self.scale = math.sqrt(channels)

    def forward(self, x):
        return x * self.scale + self.encoding


class KerasBatchNorm(nn.Module):
    """Keras batch normalization: epsilon .001 and biased moving variance."""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer('running_mean', torch.zeros(channels))
        self.register_buffer('running_var', torch.ones(channels))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                (var, mean) = torch.var_mean(x, dim=(0, 2), unbiased=False)
                self.running_mean.lerp_(mean, 0.01)
                self.running_var.lerp_(var, 0.01)
            return F.batch_norm(x, None, None, self.weight, self.bias, True, 0.0, 0.001)
        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, 0.001
        )


class KerasLSTM(nn.LSTM):
    """Input dropout uses one mask across timesteps and all four gates."""

    def __init__(self, input_size):
        super().__init__(input_size, 64, batch_first=True)
        self.bias_hh_l0.requires_grad_(False)
        self.reset_keras_parameters()

    def reset_keras_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih_l0)
        nn.init.orthogonal_(self.weight_hh_l0)
        nn.init.zeros_(self.bias_ih_l0)
        nn.init.zeros_(self.bias_hh_l0)
        with torch.no_grad():
            self.bias_ih_l0[64:128].fill_(1.0)

    def forward(self, x, dropout=True):
        if dropout:
            x = x * F.dropout(torch.ones_like(x[:, 0]), p=0.1, training=True)[:, None, :]
        return super().forward(x)[0]


class AttentionGate(nn.Module):
    def __init__(self, skip_channels, up_channels):
        super().__init__()
        inter = up_channels // 4
        self.theta = nn.Conv1d(skip_channels, inter, 1)
        self.phi = nn.Conv1d(up_channels, inter, 1)
        self.psi = nn.Conv1d(inter, 1, 1)

    def forward(self, skip, up):
        rate = torch.sigmoid(self.psi(F.relu(self.theta(skip) + self.phi(up))))
        return skip * rate


class BeliefPPG(nn.Module):
    """Inputs: spectra [B, 7, 64, 2] and waveform [B, 1280, 1]."""

    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2d_1 = nn.Conv2d(32, 32, 3, padding=1)
        self.freq_pe = PositionalEncoding(64, 32)
        self.time_pe = PositionalEncoding(7, 32)
        self.dense = nn.Linear(32, 32)
        self.dense_1 = nn.Linear(32, 32)
        self.dense_2 = nn.Linear(32, 32)
        self.dense_3 = nn.Linear(32, 32)
        self.dense_4 = nn.Linear(2, 32)
        self.down = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(a, b, 3, padding=1),
                nn.Conv1d(b, b, 3, padding=1),
            ])
            for (a, b) in [(32, 12), (12, 24), (24, 48)]
        ])
        self.time_conv1 = nn.Conv1d(1, 16, 10, dilation=2)
        self.time_bn1 = KerasBatchNorm(16)
        self.time_conv2 = nn.Conv1d(16, 16, 10, dilation=2)
        self.time_bn2 = KerasBatchNorm(16)
        self.lstm = KerasLSTM(16)
        self.lstm_1 = KerasLSTM(64)
        self.dense_5 = nn.Linear(64, 48)
        self.dense_6 = nn.Linear(64, 48)
        self.weight_conv = nn.Conv1d(48, 48, 2)
        self.feat_conv = nn.Conv1d(48, 48, 2)
        self.gates = nn.ModuleList([
            AttentionGate(48, 48), AttentionGate(24, 48), AttentionGate(12, 24)
        ])
        self.up = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(a, b, 3, padding=1),
                nn.Conv1d(b, b, 3, padding=1),
            ])
            for (a, b) in [(96, 48), (72, 24), (36, 12)]
        ])
        self.output_conv = nn.Conv1d(12, 1, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.lstm.reset_keras_parameters()
        self.lstm_1.reset_keras_parameters()

    @staticmethod
    def attention(query, value):
        return torch.matmul(torch.softmax(torch.matmul(query, value.transpose(-1, -2)), -1), value)

    def forward(self, spec, time=None, *, lstm_dropout=True):
        if time is None:
            (spec, time) = spec
        x = spec.permute(0, 3, 1, 2)
        x = F.dropout(F.leaky_relu(self.conv2d(x), 0.2), 0.1, self.training)
        x = F.dropout(F.leaky_relu(self.conv2d_1(x), 0.2), 0.1, self.training)
        x = x.permute(0, 2, 3, 1)
        xf = self.freq_pe(x)
        freq = self.attention(self.dense(xf), self.dense_1(xf))
        xt = self.time_pe(x.transpose(1, 2))
        time_attn = self.attention(self.dense_2(xt), self.dense_3(xt)).transpose(1, 2)
        x = (self.dense_4(spec) + time_attn + freq).mean(dim=1).transpose(1, 2)
        skips = []
        for (a, b) in self.down:
            x = F.dropout(F.relu(a(x)), 0.2, self.training)
            x = F.relu(b(x))
            skips.append(x)
            x = F.max_pool1d(x, 4, 4)
        t = time.transpose(1, 2)
        t = self.time_bn1(self.time_conv1(F.pad(t, (18, 0))))
        t = F.max_pool1d(F.leaky_relu(F.dropout(t, 0.1, self.training), 0.2), 4, 4)
        t = self.time_bn2(self.time_conv2(F.pad(t, (18, 0))))
        t = F.max_pool1d(F.leaky_relu(F.dropout(t, 0.1, self.training), 0.2), 4, 4)
        t = self.lstm(t.transpose(1, 2), lstm_dropout)
        t = self.lstm_1(t, lstm_dropout)[:, -1]
        weight = F.leaky_relu(self.dense_5(t), 0.2).unsqueeze(-1)
        feat = F.leaky_relu(self.dense_6(t), 0.2).unsqueeze(-1)
        weight = F.dropout(
            torch.tanh(self.weight_conv(torch.cat([weight, x], -1))), 0.2, self.training
        )
        feat = F.dropout(F.relu(self.feat_conv(torch.cat([feat, x], -1))), 0.2, self.training)
        x = x + weight * feat
        for (gate, (a, b), skip) in zip(self.gates, self.up, reversed(skips)):
            x = x.repeat_interleave(4, dim=-1)
            x = torch.cat([x, gate(skip, x)], dim=1)
            x = F.dropout(F.relu(a(x)), 0.2, self.training)
            x = F.relu(b(x))
        logits = self.output_conv(x).flatten(1)
        prob = torch.softmax(logits, -1)
        return prob


class BinnedRegressionLoss(nn.Module):
    """Gaussian target weights normalize over the entire minibatch."""

    def __init__(self, dim=64, min_hz=0.5, max_hz=3.5, sigma_y=1.5):
        super().__init__()
        self.sigma_y = sigma_y
        self.register_buffer(
            'bin_edges', torch.tensor(bpm_bins(dim, min_hz, max_hz), dtype=torch.float32)
        )

    def y_to_bins(self, y):
        z = (self.bin_edges - y.reshape(-1, 1)) / self.sigma_y
        density = torch.exp(-0.5 * z.square()) / (self.sigma_y * math.sqrt(2 * math.pi))
        return density / density.sum()

    def forward(self, probabilities, y):
        p = probabilities / probabilities.sum(-1, keepdim=True)
        return -(self.y_to_bins(y) * p.clamp(1e-07, 1 - 1e-07).log()).sum()


def fit_transition(ys: Sequence[np.ndarray], dim=64, min_hz=0.5, max_hz=3.5):
    """Fit unnormalized source transition weights; rows index the current HR bin."""
    from scipy.stats import norm
    diffs = np.concatenate([np.log(np.asarray(y)[1:]) - np.log(np.asarray(y)[:-1]) for y in ys])
    (mu, sigma) = norm.fit(diffs)
    b = bpm_bins(dim, min_hz, max_hz)
    half = (max_hz - min_hz) * 60 / dim / 2
    lower = np.log(b[:, None] - half) - np.log(b[None, :] + half)
    upper = np.log(b[:, None] + half) - np.log(b[None, :] - half)
    return (norm.cdf(upper, mu, sigma) - norm.cdf(lower, mu, sigma)).astype(np.float32)


def forward_prior(prob, transition, state=None, segment=None):
    """Float32 forward recursion; segment changes reset the state. Returns posterior and final state."""
    is_torch = isinstance(prob, torch.Tensor)
    dim = prob.shape[-1]
    if is_torch:
        transition = torch.as_tensor(transition, dtype=torch.float32, device=prob.device)
        uniform = torch.full((dim,), 1 / dim, dtype=torch.float32, device=prob.device)
        p = (
            uniform.clone() if state is None
            else torch.as_tensor(state, device=prob.device, dtype=torch.float32).clone()
        )
        rows = []
    else:
        prob = np.asarray(prob, dtype=np.float32)
        transition = np.asarray(transition, dtype=np.float32)
        uniform = np.full(dim, 1 / dim, dtype=np.float32)
        p = uniform.copy() if state is None else np.asarray(state, dtype=np.float32).copy()
        rows = np.empty_like(prob)
    for (i, likelihood) in enumerate(prob):
        if segment is not None and i and (segment[i] != segment[i - 1]):
            p = uniform.clone() if is_torch else uniform.copy()
        p = transition @ p * likelihood
        total = p.sum()
        if not is_torch and (not np.isfinite(total) or total <= 0):
            raise ValueError('Invalid prior probability mass')
        p = p / total
        if is_torch:
            rows.append(p)
        else:
            rows[i] = p
    if is_torch:
        rows = torch.stack(rows) if rows else prob.clone()
    return (rows, p)


BPM = 30 + torch.arange(64) * 180 / 64


def harmonic_index(factor):
    target = BPM * factor
    idx = torch.round((target - 30) / (180 / 64)).long()
    idx[(target < 30) | (target > 30 + 63 * 180 / 64)] = -1
    return idx


class ReferenceEditor(nn.Module):
    """A 5,569-parameter nonnegative correction of spectra [B, 7, 64, 2]."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            'pos',
            torch.stack([torch.sin(torch.arange(64) / 64 * 3.1416 * k) for k in (1, 2, 4)]),
            persistent=False,
        )
        self.register_buffer('idx_half', harmonic_index(0.5), persistent=False)
        self.register_buffer('idx_double', harmonic_index(2.0), persistent=False)
        self.net = nn.Sequential(
            nn.Conv2d(7, 16, (3, 5), padding=(1, 2)),
            nn.GELU(),
            nn.Conv2d(16, 16, (3, 5), padding=(1, 2)),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -4.0)

    def gather(self, a, idx):
        out = a[..., idx.clamp(min=0)]
        return out * (idx >= 0).to(a.dtype)

    def forward(self, spec):
        p, a = spec[..., 0], spec[..., 1]
        ah, ad = self.gather(a, self.idx_half), self.gather(a, self.idx_double)
        x = torch.stack([p, a, ah, ad], 1)
        x = torch.cat([x, self.pos[None, :, None, :].expand(len(p), 3, p.shape[1], 64)], 1)
        out = self.net(x)[:, 0]
        removal = F.softplus(out)
        return p - removal, removal


class AdaREF(nn.Module):
    """Edit the PPG spectrum using ACC; pass edited PPG and zero ACC to BeliefPPG."""

    def __init__(self):
        super().__init__()
        self.backbone = BeliefPPG()
        self.ref = ReferenceEditor()

    def forward(self, spec, time, lstm_dropout=True):
        clean, _ = self.ref(spec)
        edited = torch.stack([clean, torch.zeros_like(clean)], -1)
        return self.backbone(edited, time, lstm_dropout=lstm_dropout)
