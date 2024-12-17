# Troubleshooting Guide

## Common Issues

### Memory Issues

#### Out of Memory (OOM) Errors
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size:
   ```python
   training_args = TrainingArguments(
       per_device_train_batch_size=4,  # Reduce from default
       gradient_accumulation_steps=8,  # Increase to compensate
   )
   ```

2. Enable memory optimizations:
   ```python
   training_args = TrainingArguments(
       gradient_checkpointing=True,
       torch_compile=True,
   )
   ```

3. Use FSDP for multi-GPU training:
   ```python
   training_args = TrainingArguments(
       fsdp="full_shard auto_wrap",
       fsdp_config={"fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP"}
   )
   ```

### Performance Issues

#### Slow Training Speed
1. Check GPU utilization:
   ```bash
   nvidia-smi -l 1  # Monitor GPU usage
   ```

2. Optimize batch size and accumulation steps:
   ```python
   training_args = TrainingArguments(
       per_device_train_batch_size=16,
       gradient_accumulation_steps=4,
   )
   ```

3. Enable torch.compile:
   ```python
   training_args = TrainingArguments(
       torch_compile=True,
       torch_compile_backend="inductor",
   )
   ```

### Installation Issues

#### Import Errors
```python
ImportError: No module named 'triton'
```

**Solutions:**
1. Check CUDA/ROCm installation
2. Install correct Triton version:
   ```bash
   # For CUDA
   pip install triton>=2.3.0

   # For ROCm
   pip install triton>=3.0.0
   ```

#### Version Conflicts
```
ERROR: Cannot install package due to version conflict
```

**Solutions:**
1. Create fresh environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install in correct order:
   ```bash
   pip install torch
   pip install triton
   pip install -e .
   ```

### Model-Specific Issues

#### Llama Model Issues
```python
ValueError: Llama requires torch>=2.1.2
```

**Solutions:**
1. Upgrade PyTorch:
   ```bash
   pip install --upgrade torch>=2.1.2
   ```

2. Check model compatibility:
   ```python
   from liger_kernel import check_model_compatibility
   check_model_compatibility("meta-llama/Llama-2-7b")
   ```

#### Custom Model Integration
If patching fails:
1. Check model architecture compatibility
2. Use manual patching:
   ```python
   from liger_kernel import manually_patch_model
   model = manually_patch_model(model, patch_config={
       "attention": True,
       "mlp": True,
       "norm": True
   })
   ```

### Environment Issues

#### CUDA/ROCm Detection
If GPU not detected:
1. Check environment variables:
   ```bash
   echo $CUDA_HOME  # For CUDA
   echo $ROCM_PATH  # For ROCm
   ```

2. Verify GPU visibility:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.device_count())
   ```

## Best Practices

### Memory Management
1. Start with small batch size
2. Enable gradient checkpointing
3. Use FSDP for large models
4. Monitor memory usage with callbacks

### Performance Optimization
1. Profile before optimizing
2. Use appropriate batch size
3. Enable torch.compile
4. Monitor training metrics

### Development Setup
1. Use virtual environments
2. Install development dependencies
3. Run tests before changes
4. Check documentation

## Getting Help
- Check [GitHub Issues](https://github.com/linkedin/Liger-Kernel/issues)
- Review [API Documentation](../api/kernels.md)
- See [Performance Guide](../guides/performance-optimization.md)
- Read [Contributing Guide](../CONTRIBUTING.md)
