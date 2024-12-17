# Quickstart Guide

## Overview
Liger Kernel is a high-performance machine learning framework that optimizes transformer model training. This guide will help you get started with basic usage.

## Basic Usage

### 1. Using with Hugging Face Transformers
```python
from transformers import AutoModelForCausalLM
from liger_kernel import apply_liger_kernel_to_llama

# Load your model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Apply Liger Kernel optimizations
model = apply_liger_kernel_to_llama(model)
```

### 2. Training with Optimized Kernels
```python
from transformers import Trainer, TrainingArguments
from liger_kernel import LigerTrainer

# Use LigerTrainer instead of standard Trainer
trainer = LigerTrainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
    ),
    train_dataset=dataset,
)

# Train as usual
trainer.train()
```

### 3. Multi-GPU Training
```python
from transformers import TrainingArguments
from liger_kernel import LigerTrainer

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_backward_prefetch": "BACKWARD_PRE",
        "fsdp_state_dict_type": "FULL_STATE_DICT",
    }
)
```

## Key Features

### Memory Optimization
Liger Kernel significantly reduces memory usage:
```python
# Enable memory optimization
training_args = TrainingArguments(
    output_dir="./results",
    gradient_checkpointing=True,
    torch_compile=True,
)
```

### Performance Monitoring
```python
from liger_kernel import LigerCallback

trainer = LigerTrainer(
    model=model,
    args=training_args,
    callbacks=[LigerCallback()],  # Monitors performance metrics
)
```

## Common Patterns

### Custom Loss Functions
```python
from liger_kernel import LigerFusedLinearCrossEntropyLoss

# Use optimized loss function
model.config.use_fused_linear_cross_entropy = True
```

### Model Patching
```python
from liger_kernel import AutoLigerKernelForCausalLM

# Automatically apply appropriate optimizations
model = AutoLigerKernelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
```

## Next Steps
- Check [Performance Optimization Guide](../guides/performance-optimization.md) for advanced tuning
- See [Model Patching Guide](../guides/model-patching.md) for custom model support
- Review [Troubleshooting Guide](troubleshooting.md) for common issues
- Explore [API Documentation](../api/kernels.md) for detailed reference
