"""Base implementation for model forward functions.

This module provides a base implementation of the forward pass used across
different model architectures in Liger, particularly focusing on the
fused linear cross entropy calculation during training.
"""

from typing import Optional, Tuple, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)


def base_lce_forward(
    self,
    hidden_states: torch.Tensor,
    labels: Optional[torch.LongTensor],
    lm_head_weight: torch.Tensor,
    vocab_size: int,
    config,
    pretraining_tp: int = 1,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Base implementation of fused linear cross entropy forward pass.

    This function implements the core logic for computing loss and logits
    during both training and inference, with special handling for
    pretraining tensor parallelism.

    Args:
        hidden_states: Output of the transformer model
        labels: Optional target labels for loss computation
        lm_head_weight: Weight matrix for the language model head
        vocab_size: Size of the vocabulary
        config: Model configuration object
        pretraining_tp: Pretraining tensor parallelism factor. Defaults to 1.

    Returns:
        Tuple containing:
            - loss: Computed loss if labels provided, None otherwise
            - logits: Computed logits if not in training or labels not provided,
                     None otherwise
    """
    loss = None
    logits = None

    if self.training and (labels is not None):
        # During training with labels, use fused implementation
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_hidden_states = shift_hidden_states.view(-1, config.hidden_size)
        shift_labels = shift_labels.view(-1)

        lce = LigerFusedLinearCrossEntropyLoss()
        loss = lce(lm_head_weight, shift_hidden_states, shift_labels)

    else:
        # During inference or when labels not provided
        if pretraining_tp > 1:
            # Handle tensor parallelism if enabled
            lm_head_slices = lm_head_weight.split(vocab_size // pretraining_tp, dim=0)
            logits = [
                torch.nn.functional.linear(hidden_states, lm_head_slices[i])
                for i in range(pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            # Standard linear projection
            logits = torch.nn.functional.linear(hidden_states, lm_head_weight)

    return loss, logits


def create_causal_lm_output(
    loss: Optional[torch.Tensor],
    logits: Optional[torch.Tensor],
    outputs,
    return_dict: bool = True,
) -> Union[Tuple, CausalLMOutputWithPast]:
    """Creates the appropriate output format for causal language models.

    Args:
        loss: Optional computed loss
        logits: Optional computed logits
        outputs: Base model outputs
        return_dict: Whether to return a dictionary or tuple

    Returns:
        Model outputs in the requested format
    """
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
