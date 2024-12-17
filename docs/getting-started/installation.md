# Installation Guide

## Prerequisites
- Python >= 3.8
- CUDA >= 11.8 (for CUDA support) or ROCm >= 5.6 (for AMD GPU support)
- Git

## Quick Installation
```bash
pip install liger-kernel
```

## Installation from Source
```bash
git clone https://github.com/linkedin/Liger-Kernel.git
cd Liger-Kernel
pip install -e .  # For basic installation
pip install -e .[dev]  # For development installation
```

If you encounter error `no matches found: .[dev]`, use:
```bash
pip install -e '.[dev]'
```

## Optional Dependencies
- For transformer models patching:
  ```bash
  pip install -e .[transformers]
  ```
- For development and testing:
  ```bash
  pip install -e .[dev]
  ```

## GPU Support Requirements

### CUDA Support
- CUDA >= 11.8
- torch >= 2.1.2
- triton >= 2.3.0

### ROCm Support
- ROCm >= 5.6
- torch >= 2.5.0
- triton >= 3.0.0

## Common Installation Issues

### CUDA/ROCm Not Found
Ensure CUDA or ROCm is properly installed and visible in your PATH. Check with:
```bash
# For CUDA
nvcc --version
echo $CUDA_HOME

# For ROCm
rocm-smi
echo $ROCM_PATH
```

### Import Errors
If you encounter import errors, ensure you have all required dependencies:
```bash
pip install -r requirements.txt
```

### Version Conflicts
If you encounter version conflicts:
1. Create a fresh virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```
2. Install dependencies in order:
   ```bash
   pip install torch
   pip install triton
   pip install liger-kernel
   ```

## Verifying Installation
Test your installation:
```python
import liger_kernel
print(liger_kernel.__version__)
```

## Next Steps
- Check out our [Quickstart Guide](quickstart.md) to begin using Liger Kernel
- Review [Troubleshooting](troubleshooting.md) if you encounter any issues
- For development setup, see our [Contributing Guide](../CONTRIBUTING.md)
