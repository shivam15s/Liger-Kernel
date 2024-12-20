from test.transformers.test_utils import (
    generate_random_labels,
    generate_random_tensor,
    run_loss_test,
    run_softcap_loss_test,
    run_z_loss_test,
)
from test.utils import set_seed, supports_bfloat16

import pytest
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from liger_kernel.ops.cross_entropy import (
    LigerCrossEntropyFunction,
    liger_cross_entropy_kernel,
)
from liger_kernel.ops.utils import is_hip
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.utils import infer_device

device = infer_device()
set_seed(42)


class CrossEntropyWithZLoss(torch.nn.Module):
    def __init__(
        self,
        lse_square_scale=0.0,
        reduction="mean",
        ignore_index=-100,
        label_smoothing=0.0,
        return_z_loss=False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.lse_square_scale = lse_square_scale
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.return_z_loss = return_z_loss
        self.label_smoothing = label_smoothing
        self.dtype = dtype

    def forward(self, logits, targets):
        # Loss calculations are all in float32
        logits = logits.to(torch.float32)
        # Standard cross entropy loss
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )

        # Compute log-sum-exp term
        lse = torch.logsumexp(logits, dim=-1)

        # Z-loss term
        z_loss = torch.where(
            targets != self.ignore_index, self.lse_square_scale * (lse**2), 0.0
        )
        z_loss = z_loss.to(logits.dtype)
        if self.reduction == "mean":
            z_loss = z_loss.sum() / (targets != self.ignore_index).sum()
        elif self.reduction == "sum":
            z_loss = z_loss.sum()
        else:
            z_loss = z_loss
        ce_loss = ce_loss.to(self.dtype)
        z_loss = z_loss.to(self.dtype)

        # Final loss: cross-entropy loss + Z-loss
        total_loss = ce_loss + z_loss
        if self.return_z_loss:
            return total_loss, z_loss
        else:
            return total_loss


def _test_correctness_once(target_ce, B, T, V, reduction, scalar, dtype, atol, rtol):
    """Test basic cross entropy functionality."""
    torch.manual_seed(0)
    torch_ce = CrossEntropyLoss(reduction=reduction)

    input_tensor = (
        generate_random_tensor(
            batch_size=B,
            seq_length=T,
            hidden_size=V,
            dtype=dtype,
            device=device,
        )
        * scalar
    )

    target = generate_random_labels(
        batch_size=B,
        seq_length=T,
        vocab_size=V,
        device=device,
    )

    # Compare outputs and gradients
    run_loss_test(
        target_ce,
        input_tensor,
        target,
        torch_ce(input_tensor, target),
        atol=atol,
        rtol=rtol,
    )


def _test_correctness_with_ignore_index_once(
    target_ce, B, T, V, ignore_index, reduction, scalar, dtype, atol, rtol
):
    """Test cross entropy with ignore_index parameter."""
    torch_ce = CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    input_tensor = (
        generate_random_tensor(
            batch_size=B,
            seq_length=T,
            hidden_size=V,
            dtype=dtype,
            device=device,
        )
        * scalar
    )

    target = generate_random_labels(
        batch_size=B,
        seq_length=T,
        vocab_size=V,
        ignore_index=ignore_index,
        device=device,
    )

    # Compare outputs and gradients
    run_loss_test(
        target_ce,
        input_tensor,
        target,
        torch_ce(input_tensor, target),
        atol=atol,
        rtol=rtol,
    )


def _test_correctness_with_label_smoothing_once(
    target_ce, B, T, V, label_smoothing, scalar, dtype, atol, rtol
):
    """Test cross entropy with label smoothing."""
    torch_ce = CrossEntropyLoss(label_smoothing=label_smoothing)

    input_tensor = (
        generate_random_tensor(
            batch_size=B,
            seq_length=T,
            hidden_size=V,
            dtype=dtype,
            device=device,
        )
        * scalar
    )

    target = generate_random_labels(
        batch_size=B,
        seq_length=T,
        vocab_size=V,
        device=device,
    )

    # Compare outputs and gradients
    run_loss_test(
        target_ce,
        input_tensor,
        target,
        torch_ce(input_tensor, target),
        atol=atol,
        rtol=rtol,
    )


def _test_correctness_with_label_smoothing_with_ignore_index_once(
    target_ce, B, T, V, ignore_index, label_smoothing, scalar, dtype, atol, rtol
):
    """Test cross entropy with both label smoothing and ignore_index."""
    torch_ce = CrossEntropyLoss(
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )

    input_tensor = (
        generate_random_tensor(
            batch_size=B,
            seq_length=T,
            hidden_size=V,
            dtype=dtype,
            device=device,
        )
        * scalar
    )

    target = generate_random_labels(
        batch_size=B,
        seq_length=T,
        vocab_size=V,
        ignore_index=ignore_index,
        device=device,
    )

    # Compare outputs and gradients
    run_loss_test(
        target_ce,
        input_tensor,
        target,
        torch_ce(input_tensor, target),
        atol=atol,
        rtol=rtol,
    )


def _test_correctness_with_softcap_once(
    target_ce, B, T, V, softcap, reduction, scalar, dtype, atol, rtol
):
    """Test cross entropy with softcap transformation."""
    input_tensor = (
        generate_random_tensor(
            batch_size=B,
            seq_length=T,
            hidden_size=V,
            dtype=dtype,
            device=device,
        )
        * scalar
    )

    target = generate_random_labels(
        batch_size=B,
        seq_length=T,
        vocab_size=V,
        device=device,
    )

    run_softcap_loss_test(
        target_ce,
        input_tensor,
        target,
        softcap=softcap,
        reduction=reduction,
        dtype=dtype,
        atol=atol,
        rtol=rtol,
    )


def _test_correctness_with_z_loss_once(
    target_ce, B, T, V, scalar, dtype, atol, rtol, lse_square_scale, return_z_loss
):
    """Test cross entropy with z-loss computation."""
    torch.manual_seed(0)

    input_tensor = (
        generate_random_tensor(
            batch_size=B,
            seq_length=T,
            hidden_size=V,
            dtype=dtype,
            device=device,
        )
        * scalar
    )

    target = generate_random_labels(
        batch_size=B,
        seq_length=T,
        vocab_size=V,
        device=device,
    )

    run_z_loss_test(
        target_ce,
        input_tensor,
        target,
        lse_square_scale=lse_square_scale,
        return_z_loss=return_z_loss,
        dtype=dtype,
        atol=atol,
        rtol=rtol,
    )


def _test_correctness_with_z_loss_with_other_params_once(
    target_ce,
    B,
    T,
    V,
    scalar,
    dtype,
    atol,
    rtol,
    lse_square_scale,
    return_z_loss,
    label_smoothing,
    ignore_index,
    reduction,
):
    """Test cross entropy with z-loss and additional parameters."""
    torch.manual_seed(0)

    input_tensor = (
        generate_random_tensor(
            batch_size=B,
            seq_length=T,
            hidden_size=V,
            dtype=dtype,
            device=device,
        )
        * scalar
    )

    target = generate_random_labels(
        batch_size=B,
        seq_length=T,
        vocab_size=V,
        ignore_index=ignore_index,
        device=device,
    )

    run_z_loss_test(
        target_ce,
        input_tensor,
        target,
        lse_square_scale=lse_square_scale,
        return_z_loss=return_z_loss,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
        reduction=reduction,
        dtype=dtype,
        atol=atol,
        rtol=rtol,
    )


def _test_correctness_not_last_layer_once(
    target_ce, B, T, V, reduction, scalar, dtype, atol, rtol
):
    """Test cross entropy when not in the last layer."""
    torch.manual_seed(0)
    from torch.nn import CrossEntropyLoss

    input_tensor = (
        generate_random_tensor(
            batch_size=B,
            seq_length=T,
            hidden_size=V,
            dtype=dtype,
            device=device,
        )
        * scalar
    )

    target = generate_random_labels(
        batch_size=B,
        seq_length=T,
        vocab_size=V,
        device=device,
    )

    # Use standard CrossEntropyLoss for comparison
    torch_ce = CrossEntropyLoss(reduction=reduction)

    # Clone inputs for gradient computation
    input1 = input_tensor.detach().clone().requires_grad_(True)
    input2 = input_tensor.detach().clone().requires_grad_(True)

    # Compute losses
    output1 = torch_ce(input1, target)
    output2 = target_ce(input2, target)

    # Compare outputs
    assert torch.allclose(output1, output2, atol=atol, rtol=rtol)

    # Apply additional operation (scaling by 3)
    loss1 = output1 * 3
    loss2 = output2 * 3

    # Backpropagate with ones
    loss1.backward(gradient=torch.ones_like(output1))
    loss2.backward(gradient=torch.ones_like(output2))
    assert torch.allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)


def _test_correctness_functional(
    B,
    T,
    V,
    scalar,
    dtype,
    atol,
    rtol,
):
    """Test functional interface implementations."""
    torch.manual_seed(0)

    input_tensor = (
        generate_random_tensor(
            batch_size=B,
            seq_length=T,
            hidden_size=V,
            dtype=dtype,
            device=device,
        )
        * scalar
    )

    target = generate_random_labels(
        batch_size=B,
        seq_length=T,
        vocab_size=V,
        device=device,
    )

    # Test with z-loss and all features enabled
    run_z_loss_test(
        LigerCrossEntropyFunction.apply,
        input_tensor,
        target,
        lse_square_scale=1e-4,
        return_z_loss=True,
        label_smoothing=0.1,
        ignore_index=0,
        reduction="mean",
        dtype=dtype,
        atol=atol,
        rtol=rtol,
    )


#############################################################################
# Test the correctness of the liger cross entropy loss
#############################################################################


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 4096, 32000),  # llama
        (3, 423, 32000),  # weird shapes
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness(B, T, V, scalar, dtype, reduction, atol, rtol):
    liger_ce = LigerCrossEntropyLoss(reduction=reduction)
    _test_correctness_once(liger_ce, B, T, V, reduction, scalar, dtype, atol, rtol)


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 2, 8),
        # weird shapes
        (9, 7, 41),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 1e-8, 5e-2),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_functional(B, T, V, scalar, dtype, atol, rtol):
    _test_correctness_functional(B, T, V, scalar, dtype, atol, rtol)


@pytest.mark.parametrize(
    "B, T, V, ignore_index",
    [
        (2, 4096, 32000, 2),
        # weird shapes
        (3, 423, 32000, -123),
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_with_ignore_index(
    B, T, V, ignore_index, reduction, scalar, dtype, atol, rtol
):
    liger_ce = LigerCrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
    _test_correctness_with_ignore_index_once(
        liger_ce, B, T, V, ignore_index, reduction, scalar, dtype, atol, rtol
    )


@pytest.mark.parametrize(
    "B, T, V, label_smoothing",
    [
        (2, 4096, 32000, 0.1),
        # weird shapes
        (3, 423, 32000, 0.1),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_with_label_smoothing_once(
    B, T, V, label_smoothing, scalar, dtype, atol, rtol
):
    liger_ce = LigerCrossEntropyLoss(label_smoothing=label_smoothing)
    _test_correctness_with_label_smoothing_once(
        liger_ce, B, T, V, label_smoothing, scalar, dtype, atol, rtol
    )


@pytest.mark.parametrize(
    "B, T, V, ignore_index, label_smoothing",
    [
        (2, 4096, 32000, 1, 0.1),
        # weird shapes
        (3, 423, 32000, -300, 0.2),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_with_label_smoothing_with_ignore_index_once(
    B, T, V, ignore_index, label_smoothing, scalar, dtype, atol, rtol
):
    liger_ce = LigerCrossEntropyLoss(
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    _test_correctness_with_label_smoothing_with_ignore_index_once(
        liger_ce, B, T, V, ignore_index, label_smoothing, scalar, dtype, atol, rtol
    )


@pytest.mark.parametrize(
    "B, T, V, softcap",
    [
        (2, 4096, 32000, 30.0),  # llama2, mistral
        # weird shapes
        (3, 423, 32000, 30.0),
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_with_softcap_once(
    B, T, V, softcap, reduction, scalar, dtype, atol, rtol
):
    liger_ce = LigerCrossEntropyLoss(softcap=softcap, reduction=reduction)
    _test_correctness_with_softcap_once(
        liger_ce, B, T, V, softcap, reduction, scalar, dtype, atol, rtol
    )


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 4096, 32000),  # llama2
        # weird shapes
        (3, 423, 32000),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
@pytest.mark.parametrize("return_z_loss", [True, False])
@pytest.mark.parametrize(
    "lse_square_scale",
    [
        1e-4,  # PaLM
        1e-5,  # Chameleon
    ],
)
def test_correctness_with_z_loss_once(
    B,
    T,
    V,
    scalar,
    dtype,
    atol,
    rtol,
    lse_square_scale,
    return_z_loss,
):
    test_ce = LigerCrossEntropyLoss(
        lse_square_scale=lse_square_scale,
        return_z_loss=return_z_loss,
    )
    _test_correctness_with_z_loss_once(
        test_ce,
        B,
        T,
        V,
        scalar,
        dtype,
        atol,
        rtol,
        lse_square_scale,
        return_z_loss,
    )


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 4096, 32000),  # llama2, mistral
        # weird shapes
        (3, 423, 32000),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
@pytest.mark.parametrize(
    "return_z_loss, lse_square_scale",
    [
        (True, 1e-4),
        (False, 1e-5),
    ],
)
@pytest.mark.parametrize(
    "label_smoothing, ignore_index, reduction",
    [
        (0.1, 42, "mean"),
        (0.2, -42, "sum"),
    ],
)
def test_correctness_with_z_loss_with_other_params_once(
    B,
    T,
    V,
    scalar,
    dtype,
    atol,
    rtol,
    lse_square_scale,
    return_z_loss,
    label_smoothing,
    ignore_index,
    reduction,
):
    test_ce = LigerCrossEntropyLoss(
        lse_square_scale=lse_square_scale,
        return_z_loss=return_z_loss,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    _test_correctness_with_z_loss_with_other_params_once(
        test_ce,
        B,
        T,
        V,
        scalar,
        dtype,
        atol,
        rtol,
        lse_square_scale,
        return_z_loss,
        label_smoothing,
        ignore_index,
        reduction,
    )


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 4096, 32000),  # llama2, mistral
        # # weird shapes
        (3, 423, 32000),
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_not_last_layer(B, T, V, reduction, scalar, dtype, atol, rtol):
    liger_ce = LigerCrossEntropyLoss(reduction=reduction)
    _test_correctness_not_last_layer_once(
        liger_ce, B, T, V, reduction, scalar, dtype, atol, rtol
    )


def test_float32_internal():
    """
    This test validates that the internal softmax calculations occur in float32,
    even if the input dtype is bfloat16.
    """
    # Set up test parameters
    batch_size = 4
    n_cols = 128256
    n_non_ignore = batch_size
    ignore_index = -100
    label_smoothing = 0.0
    lse_square_scale = 0.0
    softcap = 0.0
    BLOCK_SIZE = 32768
    reduction = "mean"

    # Initialize input tensors
    X_init = torch.randn(batch_size, n_cols, dtype=torch.bfloat16, device=device)
    Y = torch.randint(0, n_cols, (batch_size,), device=device)

    # Run kernel for bfloat16
    X_bf16 = X_init.clone()
    loss_bf16 = torch.zeros(batch_size, dtype=torch.float32, device=device)
    liger_cross_entropy_kernel[(batch_size,)](
        X_ptr=X_bf16,
        X_stride=X_bf16.stride(-2),
        Y_ptr=Y,
        Y_stride=Y.stride(-1),
        z_loss_ptr=loss_bf16,  # dummy ptr, not used
        loss_ptr=loss_bf16,
        loss_stride=loss_bf16.stride(-1),
        n_cols=n_cols,
        n_non_ignore=n_non_ignore,
        ignore_index=ignore_index,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
        softcap=softcap,
        RETURN_Z_LOSS=0,  # False
        HAS_SOFTCAPPING=False,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32 if not is_hip() else 16,
    )

    # Run kernel for float32
    X_fp32 = X_init.float()
    loss_fp32 = torch.zeros(batch_size, dtype=torch.float32, device=device)
    liger_cross_entropy_kernel[(batch_size,)](
        X_ptr=X_fp32,
        X_stride=X_fp32.stride(-2),
        Y_ptr=Y,
        Y_stride=Y.stride(-1),
        loss_ptr=loss_fp32,
        z_loss_ptr=loss_fp32,  # dummy ptr, not used
        loss_stride=loss_fp32.stride(-1),
        n_cols=n_cols,
        n_non_ignore=n_non_ignore,
        ignore_index=ignore_index,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
        softcap=softcap,
        RETURN_Z_LOSS=0,  # False
        HAS_SOFTCAPPING=False,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32 if not is_hip() else 16,
    )

    torch.allclose(X_bf16, X_fp32.bfloat16())
    torch.allclose(loss_bf16, loss_fp32)
