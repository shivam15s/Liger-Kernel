# Performance Optimization Guide

## Overview
This guide covers advanced techniques for optimizing Liger Kernel's performance in terms of memory usage, training speed, and multi-GPU scaling.

## Memory Optimization

### Gradient Checkpointing
Reduce memory usage by recomputing activations during backward pass:
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```

### Optimized Loss Functions
Use fused operations to reduce memory overhead:
```python
from liger_kernel import LigerFusedLinearCrossEntropyLoss

# Enable fused linear cross entropy
model.config.use_fused_linear_cross_entropy = True
```

### Memory-Efficient Attention
```python
# Enable optimized attention implementation
model.config.use_liger_attention = True
```

## Training Speed Optimization

### Batch Size Tuning
Find optimal batch size for your hardware:
```python
training_args = TrainingArguments(
    per_device_train_batch_size=32,  # Adjust based on GPU memory
    gradient_accumulation_steps=4,    # Increase for larger effective batch
)
```

### Compiler Optimization
Enable PyTorch 2.0 compiler:
```python
training_args = TrainingArguments(
    torch_compile=True,
    torch_compile_backend="inductor",  # Try different backends
)
```

### Data Loading Optimization
```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,        # Adjust based on CPU cores
    pin_memory=True,      # Faster CPU to GPU transfer
    prefetch_factor=2,    # Prefetch batches
)
```

## Multi-GPU Training

### FSDP Configuration
Optimize for distributed training:
```python
training_args = TrainingArguments(
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_backward_prefetch": "BACKWARD_PRE",
        "fsdp_state_dict_type": "FULL_STATE_DICT",
        "fsdp_offload_params": False,  # Enable for CPU offloading
    }
)
```

### Mixed Precision Training
```python
training_args = TrainingArguments(
    fp16=True,                # or bf16=True for newer GPUs
    fp16_opt_level="O2",      # Balance speed and stability
    fp16_backend="auto",
)
```

## Performance Monitoring

### Memory Profiling
```python
from liger_kernel.utils import profile_memory

@profile_memory
def training_step():
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
```

### Speed Benchmarking
```python
from liger_kernel import LigerCallback

trainer = LigerTrainer(
    model=model,
    args=training_args,
    callbacks=[LigerCallback()],  # Monitors throughput, memory
)
```

## Model-Specific Optimizations

### Llama Models
```python
from liger_kernel import apply_liger_kernel_to_llama

# Enable Llama-specific optimizations
model = apply_liger_kernel_to_llama(
    model,
    use_fused_attention=True,
    use_fused_mlp=True,
)
```

### Gemma Models
```python
from liger_kernel import apply_liger_kernel_to_gemma

model = apply_liger_kernel_to_gemma(
    model,
    use_fused_layernorm=True,
)
```

## Best Practices

### Memory Management
1. Start with small batch size, increase gradually
2. Monitor GPU memory usage with `nvidia-smi`
3. Use gradient checkpointing for large models
4. Enable mixed precision training

### Speed Optimization
1. Profile before optimizing
2. Find optimal batch size for your hardware
3. Enable compiler optimizations
4. Use appropriate number of workers for data loading

### Multi-GPU Training
1. Use FSDP for large models
2. Enable mixed precision
3. Monitor GPU utilization
4. Balance CPU offloading with speed

## Troubleshooting

### Common Issues
1. OOM errors: Reduce batch size or enable optimizations
2. Slow training: Check data loading and GPU utilization
3. Poor scaling: Verify FSDP configuration

## Next Steps
- See [Advanced Usage Guide](advanced-usage.md) for custom implementations
- Check [Model Patching Guide](model-patching.md) for model-specific details
- Review [API Documentation](../api/kernels.md) for detailed options
