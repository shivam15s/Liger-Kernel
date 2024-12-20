from typing import Optional

import torch

from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.transformers.base_loss import LigerBaseLoss


class LigerCrossEntropyLoss(LigerBaseLoss):
    def __init__(
        self,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
    ):
        super().__init__(
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
        )
        self.return_z_loss = return_z_loss

    def forward(self, _input: torch.Tensor, target: torch.Tensor):
        loss, z_loss = LigerCrossEntropyFunction.apply(
            _input,
            target,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            self.softcap,
            self.return_z_loss,
        )
        if not self.return_z_loss:
            return loss
        return loss, z_loss
