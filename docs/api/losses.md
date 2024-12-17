# Loss Functions API Reference

## Core Loss Functions

### LigerFusedLinearCrossEntropyLoss
```python
from liger_kernel import LigerFusedLinearCrossEntropyLoss

class LigerFusedLinearCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        Fused implementation of linear and cross entropy loss.

        Args:
            ignore_index: Target value to ignore
            reduction: Reduction method ('none', 'mean', 'sum')
            label_smoothing: Label smoothing factor
        """
```

### LigerJSDLoss
```python
from liger_kernel import LigerJSDLoss

class LigerJSDLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        temperature: float = 1.0,
    ):
        """
        Jensen-Shannon Divergence loss implementation.

        Args:
            reduction: Reduction method ('none', 'mean', 'sum')
            temperature: Temperature for softmax
        """
```

## Memory-Efficient Losses

### ChunkedCrossEntropyLoss
```python
from liger_kernel import ChunkedCrossEntropyLoss

class ChunkedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        chunk_size: int = 128,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        """
        Memory-efficient cross entropy using chunked computation.

        Args:
            chunk_size: Size of chunks for computation
            ignore_index: Target value to ignore
            reduction: Reduction method
        """
```

### GradientCheckpointedLoss
```python
from liger_kernel import GradientCheckpointedLoss

class GradientCheckpointedLoss(nn.Module):
    def __init__(
        self,
        base_loss: nn.Module,
        checkpoint_segments: int = 2,
    ):
        """
        Loss wrapper with gradient checkpointing.

        Args:
            base_loss: Base loss function
            checkpoint_segments: Number of segments for checkpointing
        """
```

## Custom Loss Development

### Creating Custom Losses
```python
from liger_kernel import LigerLoss

class CustomLoss(LigerLoss):
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Custom loss computation.

        Args:
            logits: Model predictions
            labels: Ground truth labels
        """
```

### Loss Registration
```python
from liger_kernel import register_loss

@register_loss("custom_loss")
class RegisteredLoss(LigerLoss):
    """
    Register custom loss for use with Liger Kernel.
    """
```

## Configuration Options

### Loss Config
```python
from liger_kernel import LossConfig

config = LossConfig(
    reduction="mean",
    ignore_index=-100,
    label_smoothing=0.1,
    use_fused_implementation=True,
)
```

### Memory Config
```python
from liger_kernel import LossMemoryConfig

memory_config = LossMemoryConfig(
    chunk_size=128,
    gradient_checkpointing=True,
    max_memory_gb=32,
)
```

## Best Practices

### Memory Management
1. Use chunked computation for large batches
2. Enable gradient checkpointing when needed
3. Monitor memory usage
4. Use appropriate chunk sizes

### Performance Optimization
1. Use fused implementations when possible
2. Profile loss computation
3. Balance accuracy and speed
4. Consider mixed precision

## Common Issues

### Memory Issues
1. OOM during loss computation
   - Solution: Use chunked computation or reduce batch size
2. High memory usage
   - Solution: Enable gradient checkpointing

### Numerical Issues
1. Loss instability
   - Solution: Adjust label smoothing or use mixed precision
2. NaN values
   - Solution: Check input ranges and normalization

## See Also
- [Kernels API Reference](kernels.md)
- [Models API Reference](models.md)
- [Performance Optimization Guide](../guides/performance-optimization.md)
