import torch

from gl_gym.RL.agri_metarl.meta_advantage_head import (
    AdvantageResidualHead,
    TransitionSetEncoder,
)


def support_batch():
    torch.manual_seed(7)
    return {
        "observations": torch.randn(3, 6),
        "actions": torch.randn(3, 2),
        "rewards": torch.randn(3),
        "next_observations": torch.randn(3, 6),
        "dones": torch.tensor([False, False, True]),
    }


def test_set_encoder_is_permutation_invariant():
    encoder = TransitionSetEncoder(obs_dim=6, action_dim=2, context_dim=8)
    batch = support_batch()
    a = encoder(**batch)
    order = torch.tensor([2, 0, 1])
    b = encoder(**{key: value[order] for key, value in batch.items()})
    torch.testing.assert_close(a, b)


def test_residual_is_bounded_and_has_gradients():
    head = AdvantageResidualHead(obs_dim=6, context_dim=8, alpha=0.5)
    observations = torch.randn(4, 6)
    advantages = torch.randn(4)
    context = torch.randn(8)

    corrected, residual = head(observations, advantages, context)

    assert corrected.shape == advantages.shape
    assert torch.all(residual.abs() <= 0.5 + 1e-6)
    corrected.sum().backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_encoder_and_residual_outputs_are_finite():
    encoder = TransitionSetEncoder(obs_dim=6, action_dim=2, context_dim=8)
    head = AdvantageResidualHead(obs_dim=6, context_dim=8, alpha=0.25)
    context = encoder(**support_batch())
    corrected, residual = head(torch.randn(2, 6), torch.randn(2), context)
    assert torch.isfinite(context).all()
    assert torch.isfinite(corrected).all()
    assert torch.isfinite(residual).all()


def test_residual_head_starts_as_identity_correction():
    head = AdvantageResidualHead(obs_dim=6, context_dim=8, alpha=0.5)
    raw = torch.randn(4)
    corrected, residual = head(torch.randn(4, 6), raw, torch.randn(8))
    torch.testing.assert_close(residual, torch.zeros_like(residual))
    torch.testing.assert_close(corrected, raw)
