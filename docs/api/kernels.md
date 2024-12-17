# Kernels API Reference

## Core Kernels

### Flash Attention
```python
from liger_kernel.ops import flash_attention_forward, flash_attention_backward

def flash_attention_forward(
    q: torch.Tensor,          # shape: [batch_size, num_heads, seq_len, head_dim]
    k: torch.Tensor,          # shape: [batch_size, num_heads, seq_len, head_dim]
    v: torch.Tensor,          # shape: [batch_size, num_heads, seq_len, head_dim]
    scale: float = None,      # attention scale factor
    causal: bool = False,     # whether to apply causal mask
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Efficient attention implementation using Flash Attention algorithm.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (output, attention_weights)
    """
```

### Fused Linear
```python
from liger_kernel.ops import fused_linear_forward

def fused_linear_forward(
    x: torch.Tensor,           # shape: [..., in_features]
    weight: torch.Tensor,      # shape: [out_features, in_features]
    bias: Optional[torch.Tensor] = None,  # shape: [out_features]
) -> torch.Tensor:
    """
    Fused linear layer implementation combining matrix multiplication and bias.
    """
```

### RMSNorm
```python
from liger_kernel.ops import rmsnorm

def rmsnorm(
    x: torch.Tensor,          # shape: [..., hidden_size]
    weight: torch.Tensor,     # shape: [hidden_size]
    eps: float = 1e-6,       # epsilon for numerical stability
) -> torch.Tensor:
    """
    Root Mean Square Layer Normalization.
    """
```

## Utility Kernels

### Memory Efficient Operations
```python
from liger_kernel.ops import (
    memory_efficient_attention,
    memory_efficient_linear
)

def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Memory-efficient attention implementation using chunked computation.
    """
```

### Fused Operations
```python
from liger_kernel.ops import (
    fused_add_relu,
    fused_bias_gelu,
    fused_bias_dropout
)

def fused_bias_gelu(
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused bias addition and GELU activation.
    """
```

## Custom Kernel Development

### Kernel Registration
```python
from liger_kernel import register_kernel

@register_kernel("custom_kernel_name")
def custom_kernel_implementation():
    """
    Custom kernel implementation.
    """
```

### Performance Optimization
```python
from liger_kernel import optimize_kernel

@optimize_kernel(
    num_warps=4,
    num_stages=3,
    debug=False
)
def optimized_kernel():
    """
    Optimized kernel with specific configuration.
    """
```

## Configuration Options

### Kernel Config
```python
from liger_kernel import KernelConfig

config = KernelConfig(
    max_sequence_length=2048,
    num_attention_heads=32,
    hidden_size=1024,
    intermediate_size=4096,
    attention_dropout_prob=0.1,
    hidden_dropout_prob=0.1,
)
```

### Memory Config
```python
from liger_kernel import MemoryConfig

memory_config = MemoryConfig(
    max_memory_gb=40,
    min_memory_gb=8,
    target_memory_gb=32,
)
```

## Best Practices

### Memory Management
1. Use memory-efficient kernels for large models
2. Enable gradient checkpointing when needed
3. Monitor memory usage during training
4. Use appropriate batch sizes

### Performance Optimization
1. Profile kernels before optimization
2. Use appropriate number of warps
3. Consider memory access patterns
4. Balance computation and memory usage

## Common Issues

### CUDA Errors
1. CUDA out of memory
   - Solution: Reduce batch size or use memory-efficient kernels
2. CUDA kernel launch failed
   - Solution: Check input shapes and memory alignment

### Performance Issues
1. Slow kernel execution
   - Solution: Profile and optimize kernel configuration
2. High memory usage
   - Solution: Use memory-efficient implementations

## See Also
- [Models API Reference](models.md)
- [Loss Functions API Reference](losses.md)
- [Performance Optimization Guide](../guides/performance-optimization.md)
