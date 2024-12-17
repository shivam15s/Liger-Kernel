import torch.nn as nn

from liger_kernel.ops.swiglu import LigerSiLUMulFunction


class LigerSwiGLUMLP(nn.Module):
    """Optimized SwiGLU MLP implementation using Liger Kernel.

    This module provides an efficient implementation of the SwiGLU activation function
    combined with a feed-forward network, offering improved performance over standard
    implementations.

    Args:
        config: Model configuration object containing:
            - hidden_size (int): Size of the input and output dimensions
            - intermediate_size (int): Size of the intermediate (expanded) dimension
            - hidden_act (str): Activation function type, must be "silu" or "swish"

    Note:
        This implementation fuses the SiLU activation with the multiplication operation
        for better performance.

    Examples:
        >>> config = ModelConfig(hidden_size=768, intermediate_size=3072, hidden_act="silu")
        >>> swiglu = LigerSwiGLUMLP(config)
        >>> output = swiglu(hidden_states)  # Shape matches input hidden_states
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):

        return self.down_proj(
            LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x))
        )


class LigerBlockSparseTop2MLP(nn.Module):
    """Block-sparse MLP implementation with top-2 gating using Liger Kernel.

    This module implements a block-sparse MLP with top-2 gating mechanism, providing
    efficient computation by activating only the top 2 experts per token.

    Args:
        config: Model configuration object containing:
            - hidden_size (int): Size of the input and output dimensions
            - intermediate_size (int): Size of the intermediate (expanded) dimension
            - hidden_act (str): Activation function type, must be "silu" or "swish"

    Note:
        This implementation is particularly effective for mixture-of-experts models
        where sparse computation is desired.
    """

    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):

        return self.w2(LigerSiLUMulFunction.apply(self.w1(x), self.w3(x)))


class LigerPhi3SwiGLUMLP(nn.Module):
    """Optimized Phi-3 SwiGLU MLP implementation using Liger Kernel.

    This module provides a specialized implementation of the SwiGLU activation
    for Phi-3 models, with optimized memory usage and computation patterns.
    It patches the original Phi3MLP to use LigerSiLUMulFunction for better performance.

    Reference:
        https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/models/phi3/modeling_phi3.py#L241

    Args:
        config: Model configuration object containing:
            - hidden_size (int): Size of the input and output dimensions
            - intermediate_size (int): Size of the intermediate (expanded) dimension
            - hidden_act (str): Activation function type, must be "silu" or "swish"

    Note:
        This implementation uses a single linear layer for both gate and up projections,
        reducing memory bandwidth requirements.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(
            self.hidden_size, 2 * self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        up_states = self.gate_up_proj(x)
        gate, up_states = up_states.chunk(2, dim=-1)
        return self.down_proj(LigerSiLUMulFunction.apply(gate, up_states))
