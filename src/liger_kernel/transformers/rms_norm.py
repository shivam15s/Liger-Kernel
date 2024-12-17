import torch
import torch.nn as nn

from liger_kernel.ops.rms_norm import LigerRMSNormFunction


class LigerRMSNorm(nn.Module):
    """Optimized Root Mean Square Layer Normalization implementation.

    This module provides a highly efficient implementation of RMSNorm using Liger Kernel
    optimizations, resulting in improved training throughput and reduced memory usage
    compared to standard implementations.

    Args:
        hidden_size (int): Size of the hidden dimension to normalize
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
        offset (float, optional): Offset added to the normalized values. Defaults to 0.0.
        casting_mode (str, optional): Mode for type casting operations. Defaults to "llama".
        init_fn (str, optional): Weight initialization function, "ones" or "zeros". Defaults to "ones".
        in_place (bool, optional): Whether to perform operations in-place. Defaults to True.

    Note:
        The in-place operation mode can significantly reduce memory usage during training.

    Examples:
        >>> rms_norm = LigerRMSNorm(hidden_size=768)
        >>> output = rms_norm(hidden_states)  # Shape: (batch_size, seq_len, hidden_size)
    """

    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        offset=0.0,
        casting_mode="llama",
        init_fn="ones",
        in_place=True,
    ):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.weight = nn.Parameter(
            torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size)
        )
        self.variance_epsilon, self.offset, self.casting_mode, self.in_place = (
            eps,
            offset,
            casting_mode,
            in_place,
        )

    def forward(self, hidden_states):
        """Apply RMSNorm to the input hidden states.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input
        """
        return LigerRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
            self.in_place,
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}, in_place={self.in_place}"
