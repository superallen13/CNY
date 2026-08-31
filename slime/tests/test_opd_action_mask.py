"""Test that apply_opd_kl_to_advantages honors opd_action_mask."""

from __future__ import annotations

import sys
import types

import torch


def _stub_module(name: str) -> types.ModuleType:
    """Insert (or fetch) a stub module into sys.modules."""
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# Stub heavy training-only deps so loss.py imports cleanly in a CPU-only test
# env. apply_opd_kl_to_advantages itself is pure-torch and never touches
# megatron/ray, so the stubs only need to satisfy the module import chain.
_megatron = _stub_module("megatron")
_megatron_core = _stub_module("megatron.core")
_megatron_core_mpu = _stub_module("megatron.core.mpu")
_megatron_core.mpu = _megatron_core_mpu
_megatron.core = _megatron_core
_stub_module("ray")

from slime.backends.megatron_utils.loss import apply_opd_kl_to_advantages  # noqa: E402


class _Args:
    use_opd = True
    opd_type = "sglang"
    opd_kl_coef = 0.1


def test_mask_zeros_kl_at_false_positions():
    student = [torch.tensor([-0.5, -0.4, -0.3, -0.2], dtype=torch.float32)]
    teacher = [torch.tensor([-0.8, -0.7, -0.5, -0.1], dtype=torch.float32)]
    mask = [torch.tensor([False, True, True, False], dtype=torch.bool)]
    advantages = [torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)]

    rollout_data = {"teacher_log_probs": teacher, "opd_action_mask": mask}
    apply_opd_kl_to_advantages(
        args=_Args(),
        rollout_data=rollout_data,
        advantages=advantages,
        student_log_probs=student,
    )

    # reverse_kl = (student - teacher), but masked to zero at False positions:
    # position 0: False -> 0
    # position 1: True  -> -0.4 - (-0.7) = 0.3
    # position 2: True  -> -0.3 - (-0.5) = 0.2
    # position 3: False -> 0
    # advantages[i] = 1.0 - 0.1 * masked_reverse_kl
    expected = torch.tensor([1.0, 1.0 - 0.1 * 0.3, 1.0 - 0.1 * 0.2, 1.0], dtype=torch.float32)
    assert torch.allclose(advantages[0], expected, atol=1e-6)


def test_no_mask_falls_back_to_original_behavior():
    """When opd_action_mask is absent, behave exactly as before."""
    student = [torch.tensor([-0.5, -0.4], dtype=torch.float32)]
    teacher = [torch.tensor([-0.8, -0.7], dtype=torch.float32)]
    advantages = [torch.tensor([1.0, 1.0], dtype=torch.float32)]

    rollout_data = {"teacher_log_probs": teacher}  # no mask key
    apply_opd_kl_to_advantages(
        args=_Args(),
        rollout_data=rollout_data,
        advantages=advantages,
        student_log_probs=student,
    )

    # reverse_kl = [-0.5 - (-0.8), -0.4 - (-0.7)] = [0.3, 0.3]
    expected = torch.tensor([1.0 - 0.1 * 0.3, 1.0 - 0.1 * 0.3], dtype=torch.float32)
    assert torch.allclose(advantages[0], expected, atol=1e-6)
