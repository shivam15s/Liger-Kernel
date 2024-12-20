"""Test utilities for Liger Kernel transformer tests.

This module provides helper functions to reduce code duplication
in transformer test files, particularly focusing on loss function testing.
"""

import torch


def generate_random_tensor(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate a random tensor for testing.

    Args:
        batch_size: Size of the batch dimension
        seq_length: Length of the sequence
        hidden_size: Size of the hidden dimension
        dtype: Tensor data type
        device: Device to place tensor on

    Returns:
        Random tensor of shape (batch_size, seq_length, hidden_size)
    """
    return torch.randn(
        (batch_size, seq_length, hidden_size),
        dtype=dtype,
        device=device,
    )


def generate_random_labels(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    ignore_index: int = -100,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate random labels for testing.

    Args:
        batch_size: Size of the batch dimension
        seq_length: Length of the sequence
        vocab_size: Size of the vocabulary
        ignore_index: Index to ignore in loss computation
        device: Device to place tensor on

    Returns:
        Random labels tensor of shape (batch_size, seq_length)
    """
    labels = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_length),
        device=device,
    )
    # Randomly mask some positions with ignore_index
    mask = torch.rand(batch_size, seq_length, device=device) > 0.8
    labels[mask] = ignore_index
    return labels


def run_loss_test(
    target_loss,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    expected_output: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """Run a loss function test with the given inputs and expected output.

    This helper function reduces code duplication in loss function tests by
    providing a standard way to compute and verify loss values.

    Args:
        target_loss: Loss function to test
        input_tensor: Input tensor for the loss function
        target_tensor: Target tensor for the loss function
        expected_output: Expected output tensor
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
    """
    output = target_loss(input_tensor, target_tensor)
    assert torch.allclose(
        output,
        expected_output,
        atol=atol,
        rtol=rtol,
    ), f"Expected {expected_output}, but got {output}"


def run_fused_loss_test(
    target_loss,
    weight: torch.Tensor,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    expected_output: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    bias: torch.Tensor = None,
) -> None:
    """Run a fused loss function test with the given inputs and expected output.

    Similar to run_loss_test but specifically for fused loss functions that
    combine linear transformation with loss computation.

    Args:
        target_loss: Fused loss function to test
        weight: Weight matrix for linear transformation
        input_tensor: Input tensor for the loss function
        target_tensor: Target tensor for the loss function
        expected_output: Expected output tensor
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        bias: Optional bias tensor for linear transformation
    """
    output = (
        target_loss(weight, input_tensor, target_tensor)
        if bias is None
        else target_loss(weight, input_tensor, target_tensor, bias)
    )
    assert torch.allclose(
        output,
        expected_output,
        atol=atol,
        rtol=rtol,
    ), f"Expected {expected_output}, but got {output}"


def run_softcap_loss_test(
    target_loss,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    softcap: float,
    reduction: str = "mean",
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    from torch.nn import CrossEntropyLoss

    torch_ce = CrossEntropyLoss(reduction=reduction)

    input1 = input_tensor.detach().clone().requires_grad_(True)
    input2 = input_tensor.detach().clone().requires_grad_(True)

    output = torch_ce(
        softcap * torch.tanh(input1.to(torch.float32) / softcap),
        target_tensor,
    ).to(dtype)
    output2 = target_loss(input2, target_tensor)

    assert torch.allclose(
        output,
        output2,
        atol=atol,
        rtol=rtol,
    ), f"Expected {output}, but got {output2}"

    output.backward(gradient=torch.ones_like(output))
    output2.backward(gradient=torch.ones_like(output))
    assert torch.allclose(
        input1.grad,
        input2.grad,
        atol=atol,
        rtol=rtol,
    ), f"Gradient mismatch: expected {input1.grad}, but got {input2.grad}"


def run_z_loss_test(
    target_loss,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    lse_square_scale: float,
    return_z_loss: bool = False,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
    reduction: str = "mean",
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    from test.transformers.test_cross_entropy import CrossEntropyWithZLoss

    torch_ce = CrossEntropyWithZLoss(
        lse_square_scale=lse_square_scale,
        return_z_loss=return_z_loss,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
        reduction=reduction,
        dtype=dtype,
    )

    input1 = input_tensor.detach().clone().requires_grad_(True)
    input2 = input_tensor.detach().clone().requires_grad_(True)

    if return_z_loss:
        output1, z_loss1 = torch_ce(input1, target_tensor)
        output2, z_loss2 = target_loss(input2, target_tensor)

        assert torch.allclose(
            z_loss1,
            z_loss2,
            atol=atol,
            rtol=rtol,
        ), f"Z-loss mismatch: expected {z_loss1}, but got {z_loss2}"
    else:
        output1 = torch_ce(input1, target_tensor)
        output2 = target_loss(input2, target_tensor)

    assert torch.allclose(
        output1,
        output2,
        atol=atol,
        rtol=rtol,
    ), f"Loss mismatch: expected {output1}, but got {output2}"

    output1.backward()
    output2.backward()
    assert torch.allclose(
        input1.grad,
        input2.grad,
        atol=atol,
        rtol=rtol,
    ), f"Gradient mismatch: expected {input1.grad}, but got {input2.grad}"
