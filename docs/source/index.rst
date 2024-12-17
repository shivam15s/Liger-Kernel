.. Liger-Kernel documentation master file, created by
   sphinx-quickstart on Tue Dec 17 05:54:19 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Liger-Kernel Documentation
===================================

Liger Kernel is a high-performance machine learning framework focused on optimizing GPU operations using Triton and PyTorch. It provides efficient implementations of transformer models, neural network layers, and loss functions.

Key Features
-----------

* 20% increase in multi-GPU training throughput
* 60% reduction in memory usage
* Support for multiple model architectures (LLaMA, Mistral, Mixtral, Gemma, Qwen2, Phi3)
* High-level and low-level APIs for kernel optimization
* Support for both CUDA and ROCm

Getting Started
-------------

Check out our :doc:`installation` guide to get started with Liger Kernel.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   api/high_level
   api/low_level
   api/model_kernels
   api/alignment_kernels
   api/distillation_kernels
   examples
   examples/huggingface
   examples/lightning
   examples/medusa

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

