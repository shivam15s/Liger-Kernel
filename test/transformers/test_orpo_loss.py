from test.utils import assert_verbose_allclose, set_seed

import pytest
import torch
import torch.nn.functional as F
from liger_kernel.ops.experimental.orpo_loss import (
    odds_ratio_loss,
    LigerFusedLinearORPOFunction,
)

# set random seed globally
set_seed()


def f(batch, weight, label, bias, ignore_index=-100):
    len_chosen = batch.shape[0] // 2
    unnorm_logits = batch @ weight.t()  # chunk_size x V
    if bias is not None:
        unnorm_logits = unnorm_logits + bias
    unnorm_logits = unnorm_logits.float()
    concatenated_logits = F.log_softmax(unnorm_logits, dim=-1)
    chosen_nll_loss = F.nll_loss(
        concatenated_logits[:len_chosen].view(-1, concatenated_logits.shape[-1]),
        label[:len_chosen].view(-1),
        reduction="sum",
        ignore_index=ignore_index
    )

    all_logps = concatenated_logits.gather(-1, label.unsqueeze(2)).squeeze(2)
    chosen_logps = all_logps[:len_chosen].mean(dim=1)
    rejected_logps = all_logps[len_chosen:].mean(dim=1)

    or_loss = odds_ratio_loss(chosen_logps, rejected_logps)

    chosen_nll_loss /= (label != ignore_index).sum().item()
    or_loss /= batch.shape[0]

    loss = chosen_nll_loss + or_loss
    return loss


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 2, 8, 8),
        # weird shapes
        (8, 8, 41, 41),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-3, 5e-2),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
def test_correctness_functional(B, T, H, V, scalar, dtype, bias, atol, rtol):
    device = "cuda"

    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    x1 = _input.detach().clone().requires_grad_(True)
    x2 = _input.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B, T,), device=device, dtype=torch.long)

    weight = torch.randn(V, H, device=device, dtype=dtype)
    bias = torch.randn(V, device=device, dtype=dtype) if bias else None

    y1 = f(x1, weight, target, bias)
    y2 = LigerFusedLinearORPOFunction.apply(x2, weight, target, bias)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(y1)

    y1.backward(grad_output)
    y2.backward(grad_output)

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)
