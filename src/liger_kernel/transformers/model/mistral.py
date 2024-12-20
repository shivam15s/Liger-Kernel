from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.mistral.modeling_mistral import (
    _CONFIG_FOR_DOC,
    MISTRAL_INPUTS_DOCSTRING,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

from liger_kernel.transformers.model.base_forward import (
    base_lce_forward,
    create_causal_lm_output,
)


@add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
@replace_return_docstrings(
    output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
)
def lce_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    loss, logits = base_lce_forward(
        self,
        hidden_states,
        labels,
        self.lm_head.weight,
        self.config.vocab_size,
        self.config,
    )

    return create_causal_lm_output(
        loss=loss,
        logits=logits,
        outputs=outputs,
        return_dict=return_dict,
    )


# Note: Grad Acc is not fixed in mistral at transformer 4.46.1
