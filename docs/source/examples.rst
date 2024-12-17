Examples and Usage
================

This section provides comprehensive examples and usage patterns for Liger Kernel,
demonstrating how to integrate optimizations into your models and achieve better performance.

Getting Started
-------------

Basic Usage
~~~~~~~~~~

The simplest way to use Liger Kernel is through the AutoLigerKernelForCausalLM class:

.. code-block:: python

    from liger_kernel.transformers import AutoLigerKernelForCausalLM

    # Load and optimize a model automatically
    model = AutoLigerKernelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b",
        rope=True,              # Enable RoPE optimization
        rms_norm=True,          # Use optimized RMSNorm
        swiglu=True,            # Enable SwiGLU optimization
        cross_entropy=True      # Use optimized cross entropy
    )

Performance Improvements
---------------------

Liger Kernel provides significant performance improvements:

- Up to 20% throughput increase in training
- Up to 60% memory reduction
- Optimized kernel operations for better GPU utilization

Here's how to enable specific optimizations:

.. code-block:: python

    # Selective optimization for LLaMA models
    from liger_kernel.transformers import apply_liger_kernel_to_llama

    apply_liger_kernel_to_llama(
        model,
        rope=True,                    # Rotary Position Embedding optimization
        cross_entropy=False,          # Standard cross entropy
        fused_linear_cross_entropy=True,  # Fused linear + cross entropy
        rms_norm=True,               # RMSNorm optimization
        swiglu=True                  # SwiGLU activation optimization
    )

Migration Guide
-------------

From PyTorch
~~~~~~~~~~

If you're using standard PyTorch models:

.. code-block:: python

    # Before: Standard PyTorch
    from torch.nn import LayerNorm
    norm = LayerNorm(hidden_size)

    # After: Liger Kernel optimization
    from liger_kernel.transformers import LigerRMSNorm
    norm = LigerRMSNorm(hidden_size)

From Hugging Face
~~~~~~~~~~~~~~

For Hugging Face Transformers models:

.. code-block:: python

    # Before: Standard Hugging Face
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

    # After: Liger Kernel optimization
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    model = AutoLigerKernelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

Installation Requirements
----------------------

Core Dependencies:
~~~~~~~~~~~~~~~~

- PyTorch >= 2.1.2 (CUDA) or >= 2.5.0 (ROCm)
- Triton >= 2.3.0 (CUDA) or >= 3.0.0 (ROCm)

Optional Dependencies:
~~~~~~~~~~~~~~~~~~

- transformers >= 4.x (for transformers models patching APIs)

Installation:
~~~~~~~~~~~

.. code-block:: bash

    # Basic installation
    pip install -e .

    # With transformers support
    pip install -e .[transformers]

Performance Tips
-------------

1. Enable in-place operations where possible:

.. code-block:: python

    norm = LigerRMSNorm(hidden_size, in_place=True)

2. Use fused operations for better performance:

.. code-block:: python

    model = AutoLigerKernelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b",
        fused_linear_cross_entropy=True
    )

3. Consider model-specific optimizations:

.. code-block:: python

    # For Mixtral models
    from liger_kernel.transformers import apply_liger_kernel_to_mixtral
    apply_liger_kernel_to_mixtral(model, rope=True, rms_norm=True)

    # For Phi-3 models
    from liger_kernel.transformers import apply_liger_kernel_to_phi3
    apply_liger_kernel_to_phi3(model, rope=True, swiglu=True)


Advanced Optimizations
------------------

GEGLU Activation
~~~~~~~~~~~~~

.. code-block:: python

    from liger_kernel.transformers import LigerGEGLUMLP

    # Replace standard MLP with optimized GEGLU
    geglu = LigerGEGLUMLP(
        hidden_size=1024,
        intermediate_size=4096
    )

Jensen-Shannon Divergence
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from liger_kernel.transformers import (
        LigerJSD,
        LigerFusedLinearJSD
    )

    # Standard JSD loss
    loss_fn = LigerJSD()

    # Fused linear + JSD for better performance
    fused_loss = LigerFusedLinearJSD(
        hidden_size=1024,
        num_classes=32000
    )

Layer Normalization
~~~~~~~~~~~~~~~~

.. code-block:: python

    from liger_kernel.transformers import (
        LigerLayerNorm,
        LigerRMSNorm
    )

    # Standard layer normalization
    layer_norm = LigerLayerNorm(hidden_size=1024)

    # RMS normalization (recommended for most models)
    rms_norm = LigerRMSNorm(hidden_size=1024)
