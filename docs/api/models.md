# Models API Reference

## Model Classes


### AutoLigerKernelForCausalLM
```python
from liger_kernel import AutoLigerKernelForCausalLM

class AutoLigerKernelForCausalLM:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs
    ) -> PreTrainedModel:
        """
        Load a pretrained model with Liger Kernel optimizations.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or path
            torch_dtype: Model precision (float32, float16, bfloat16)
            trust_remote_code: Whether to allow loading remote code
            **kwargs: Additional arguments passed to from_pretrained
        """
```

### LigerModelForCausalLM
```python
from liger_kernel import LigerModelForCausalLM

class LigerModelForCausalLM(PreTrainedModel):
    def __init__(
        self,
        config: PretrainedConfig,
        *,
        use_flash_attention: bool = True,
        use_fused_mlp: bool = True,
    ):
        """
        Initialize a Liger-optimized causal language model.

        Args:
            config: Model configuration
            use_flash_attention: Enable Flash Attention optimization
            use_fused_mlp: Enable fused MLP operations
        """
```

## Model Components

### LigerAttention
```python
from liger_kernel import LigerAttention

class LigerAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: Optional[int] = None,
    ):
        """
        Optimized attention implementation.

        Args:
            config: Model configuration
            layer_idx: Layer index for position-specific optimizations
        """
```

### LigerMLP
```python
from liger_kernel import LigerMLP

class LigerMLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        *,
        use_fused_operations: bool = True,
    ):
        """
        Optimized MLP implementation.

        Args:
            config: Model configuration
            use_fused_operations: Enable fused operations
        """
```

## Model Patching

### apply_liger_kernel_to_llama
```python
from liger_kernel import apply_liger_kernel_to_llama

def apply_liger_kernel_to_llama(
    model: PreTrainedModel,
    *,
    use_flash_attention: bool = True,
    use_fused_mlp: bool = True,
    use_rmsnorm: bool = True,
) -> PreTrainedModel:
    """
    Apply Liger Kernel optimizations to a Llama model.

    Args:
        model: Llama model instance
        use_flash_attention: Enable Flash Attention
        use_fused_mlp: Enable fused MLP operations
        use_rmsnorm: Enable RMSNorm optimization
    """
```

### apply_liger_kernel_to_gemma
```python
from liger_kernel import apply_liger_kernel_to_gemma

def apply_liger_kernel_to_gemma(
    model: PreTrainedModel,
    *,
    use_fused_layernorm: bool = True,
    layernorm_offset: float = 1.0,
) -> PreTrainedModel:
    """
    Apply Liger Kernel optimizations to a Gemma model.

    Args:
        model: Gemma model instance
        use_fused_layernorm: Enable fused LayerNorm
        layernorm_offset: LayerNorm epsilon offset
    """
```

## Configuration

### LigerConfig
```python
from liger_kernel import LigerConfig

class LigerConfig:
    def __init__(
        self,
        *,
        attention_type: str = "flash_attention",
        mlp_type: str = "fused_mlp",
        norm_type: str = "rmsnorm",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Configuration for Liger Kernel optimizations.

        Args:
            attention_type: Type of attention implementation
            mlp_type: Type of MLP implementation
            norm_type: Type of normalization
            dtype: Model precision
        """
```

## Training Components

### LigerTrainer
```python
from liger_kernel import LigerTrainer

class LigerTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        **kwargs
    ):
        """
        Trainer class with Liger Kernel optimizations.

        Args:
            model: Model to train
            args: Training arguments
            **kwargs: Additional trainer arguments
        """
```

## Best Practices

### Model Loading
1. Use appropriate precision (fp16/bf16)
2. Enable optimizations selectively
3. Verify hardware compatibility
4. Monitor memory usage

### Training
1. Use appropriate batch sizes
2. Enable gradient checkpointing
3. Monitor training metrics
4. Use mixed precision training

## Common Issues

### Memory Issues
1. OOM during model loading
   - Solution: Use lower precision or enable memory optimizations
2. OOM during training
   - Solution: Reduce batch size or use gradient checkpointing

### Performance Issues
1. Slow inference
   - Solution: Enable appropriate optimizations
2. Training instability
   - Solution: Adjust learning rate or batch size

## See Also
- [Kernels API Reference](kernels.md)
- [Loss Functions API Reference](losses.md)
- [Model Patching Guide](../guides/model-patching.md)
