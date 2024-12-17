# Model Patching Guide

## Overview
This guide explains how to patch existing transformer models with Liger Kernel optimizations, covering both automatic and manual patching approaches.

## Automatic Patching

### Using AutoLigerKernelForCausalLM
```python
from liger_kernel import AutoLigerKernelForCausalLM

# Automatically apply appropriate optimizations
model = AutoLigerKernelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    torch_dtype=torch.bfloat16
)
```

### Model-Specific Patching
```python
from liger_kernel import (
    apply_liger_kernel_to_llama,
    apply_liger_kernel_to_gemma
)

# For Llama models
model = apply_liger_kernel_to_llama(
    model,
    use_fused_attention=True,
    use_fused_mlp=True
)

# For Gemma models
model = apply_liger_kernel_to_gemma(
    model,
    use_fused_layernorm=True
)
```

## Manual Patching

### Component-wise Patching
```python
from liger_kernel import (
    LigerAttention,
    LigerMLP,
    LigerLayerNorm
)

def patch_model_manually(model):
    # Replace attention layers
    for layer in model.layers:
        layer.attention = LigerAttention(
            config=model.config,
            layer_idx=layer.layer_idx
        )

    # Replace MLP layers
    for layer in model.layers:
        layer.mlp = LigerMLP(
            config=model.config
        )

    # Replace normalization layers
    for layer in model.layers:
        layer.input_layernorm = LigerLayerNorm(
            config=model.config
        )

    return model
```

### Custom Component Patching
```python
from liger_kernel import LigerLayer

class CustomAttention(LigerLayer):
    def __init__(self, config):
        super().__init__()
        # Custom initialization

    def forward(self, hidden_states, attention_mask=None):
        # Custom attention implementation
        pass

def patch_with_custom_attention(model):
    for layer in model.layers:
        layer.attention = CustomAttention(model.config)
    return model
```

## Verification and Testing

### Testing Patched Models
```python
from liger_kernel import verify_model_patch

# Verify patch correctness
verify_model_patch(
    original_model,
    patched_model,
    test_input,
    rtol=1e-3,
    atol=1e-3
)
```

### Performance Benchmarking
```python
from liger_kernel import benchmark_model

results = benchmark_model(
    model,
    batch_size=32,
    sequence_length=512,
    num_iterations=100
)
print(f"Throughput: {results['throughput']} tokens/sec")
print(f"Memory usage: {results['memory_used']} GB")
```

## Model-Specific Considerations

### Llama Models
```python
# Llama-specific optimizations
patch_config = {
    "use_flash_attention": True,
    "use_fused_mlp": True,
    "use_rmsnorm": True
}

model = apply_liger_kernel_to_llama(model, **patch_config)
```

### Gemma Models
```python
# Gemma-specific optimizations
patch_config = {
    "use_fused_layernorm": True,
    "layernorm_offset": 1.0,
    "casting_mode": "full_float32"
}

model = apply_liger_kernel_to_gemma(model, **patch_config)
```

## Common Pitfalls and Solutions

### Memory Issues
1. Gradient checkpointing conflicts
   ```python
   # Solution: Disable before patching
   model.gradient_checkpointing_disable()
   model = apply_liger_kernel(model)
   model.gradient_checkpointing_enable()
   ```

### Performance Regression
1. Incorrect dtype handling
   ```python
   # Solution: Ensure consistent dtypes
   model = model.to(torch.bfloat16)
   model = apply_liger_kernel(model)
   ```

### Compatibility Issues
1. Version mismatches
   ```python
   # Solution: Check compatibility
   from liger_kernel import check_compatibility
   check_compatibility(model)
   ```

## Best Practices

### Before Patching
1. Verify model compatibility
2. Check hardware requirements
3. Backup model weights
4. Document current performance

### During Patching
1. Apply patches systematically
2. Test each component
3. Monitor memory usage
4. Verify output correctness

### After Patching
1. Benchmark performance
2. Test edge cases
3. Validate training behavior
4. Document changes

## Troubleshooting

### Common Issues
1. Incorrect output shapes
2. Unexpected memory usage
3. Training instability
4. Performance degradation

### Debugging Steps
1. Verify patch configuration
2. Check hardware compatibility
3. Test with smaller inputs
4. Compare with baseline model

## Next Steps
- See [Performance Optimization Guide](performance-optimization.md)
- Check [Advanced Usage Guide](advanced-usage.md)
- Review [API Documentation](../api/kernels.md)
