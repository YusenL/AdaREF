"""Synthetic checks for the AdaREF review implementation."""

import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from adaref_review.adaptation import adapt, chunks
from adaref_review.model import (
    AdaREF, BinnedRegressionLoss, ReferenceEditor, bpm_bins, forward_prior,
)


class CaptureBackbone(nn.Module):
    def forward(self, spec, time, **kwargs):
        self.spec, self.time = spec, time
        return torch.full((len(spec), 64), 1 / 64)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reference_removal_and_input_routing(self):
        torch.manual_seed(3)
        model = AdaREF().eval()
        self.assertIsInstance(model.ref, ReferenceEditor)
        self.assertEqual(sum(p.numel() for p in model.ref.parameters()), 5569)
        model.backbone = CaptureBackbone()
        spec = torch.randn(3, 7, 64, 2)
        time = torch.randn(3, 1280, 1)
        clean, removal = model.ref(spec)
        self.assertTrue(torch.all(removal >= 0))
        self.assertTrue(torch.all(clean <= spec[..., 0]))
        model(spec, time, lstm_dropout=False)
        torch.testing.assert_close(model.backbone.spec[..., 0], clean)
        self.assertTrue(torch.equal(model.backbone.spec[..., 1], torch.zeros_like(clean)))
        self.assertIs(model.backbone.time, time)

    def test_source_loss_normalization_and_finite_gradient(self):
        labels = torch.tensor([30.0, 100.0])
        logits = (torch.arange(128).reshape(2, 64).float() / 40).requires_grad_()
        probabilities = torch.softmax(logits, -1)
        bins = torch.as_tensor(bpm_bins(), dtype=torch.float32)
        weights = torch.exp(-0.5 * ((bins - labels[:, None]) / 1.5).square())
        weights /= weights.sum()
        expected = -(weights * probabilities.clamp(1e-7, 1 - 1e-7).log()).sum()
        criterion = BinnedRegressionLoss()
        loss = criterion(probabilities, labels)
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(criterion.y_to_bins(labels).sum(), torch.tensor(1.0))
        self.assertGreater(abs(float(criterion.y_to_bins(labels)[0].sum()) - 0.5), 0.01)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_decoding_resets_segments_and_supports_streaming(self):
        rng = np.random.default_rng(4)
        probabilities = rng.uniform(0.01, 1, (10, 64)).astype(np.float32)
        probabilities /= probabilities.sum(-1, keepdims=True)
        transition = rng.uniform(0.01, 1, (64, 64)).astype(np.float32)
        segments = np.repeat([0, 1], 5)
        combined, _ = forward_prior(probabilities, transition, segment=segments)
        first, state = forward_prior(probabilities[:5], transition)
        second, _ = forward_prior(probabilities[5:], transition)
        np.testing.assert_array_equal(combined, np.concatenate([first, second]))
        streamed, _ = forward_prior(probabilities[5:], transition, state=state)
        unbroken, _ = forward_prior(probabilities, transition)
        np.testing.assert_array_equal(unbroken, np.concatenate([first, streamed]))

    def test_half_overlapping_blocks_respect_segments(self):
        segments = np.repeat([0, 1], 8)
        starts = chunks(segments, 4)
        np.testing.assert_array_equal(starts, [0, 2, 4, 8, 10, 12])
        for start in starts:
            self.assertEqual(len(np.unique(segments[start:start + 4])), 1)
        self.assertEqual(len(chunks(np.zeros(3, dtype=np.int64), 4)), 0)

    def test_unlabelled_edta_preserves_source_and_backbone(self):
        torch.manual_seed(9)
        rng = np.random.default_rng(9)
        model = AdaREF().eval()
        initial = copy.deepcopy(model.state_dict())
        record = SimpleNamespace(
            spec=rng.normal(size=(8, 7, 64, 2)).astype(np.float32),
            time=rng.normal(size=(8, 1280, 1)).astype(np.float32),
            segment=np.zeros(8, dtype=np.int64),
        )
        self.assertFalse(hasattr(record, 'y'))
        transition = np.eye(64, dtype=np.float32)
        adapted = adapt(model, record, transition, steps=2, block_length=4, max_blocks=1)
        changed = []
        for name, tensor in adapted.state_dict().items():
            if not torch.equal(tensor, initial[name]):
                changed.append(name)
                self.assertTrue(name.startswith('ref.'), name)
            self.assertTrue(torch.equal(model.state_dict()[name], initial[name]), name)
        self.assertTrue(changed)
        self.assertTrue(all(not p.requires_grad for p in adapted.backbone.parameters()))
        self.assertFalse(adapted.training)
        np.testing.assert_array_equal(transition, np.eye(64, dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
