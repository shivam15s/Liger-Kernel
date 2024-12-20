"""Base class for Liger loss implementations.

This module provides a base class that implements common functionality
for various loss implementations in the Liger framework, particularly
focusing on parameter validation and initialization logic that is shared
across different loss types.
"""

from typing import Optional

import torch


class LigerBaseLoss(torch.nn.Module):
    """Base class for Liger loss implementations.

    This class provides common initialization logic and parameter validation
    used across different loss implementations in Liger.

    Args:
        ignore_index (int, optional): Specifies a target value that is ignored and does not
            contribute to the input gradient. Defaults to -100.
        lse_square_scale (float, optional): Scale factor for log-sum-exp squared term.
            Defaults to 0.0.
        label_smoothing (float, optional): Specifies the amount of label smoothing when
            computing the loss. Defaults to 0.0.
        reduction (str, optional): Specifies the reduction to apply to the output.
            Options: 'none' | 'mean' | 'sum'. Defaults to 'mean'.
        softcap (float, optional): If set, caps the maximum absolute value in the loss
            computation. Must be positive if provided. Defaults to None.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
    ):
        super().__init__()
        assert (label_smoothing >= 0) and (
            label_smoothing <= 1
        ), f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}"
        assert reduction in {
            "mean",
            "sum",
            "none",
        }, f"reduction must be one of 'mean', 'sum', or 'none'. Got: {reduction}"
        assert (
            softcap is None or softcap > 0
        ), f"softcap must greater than 0.0 or None. Got: {softcap}"
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap
