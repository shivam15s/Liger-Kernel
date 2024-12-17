Low-level APIs
=============

This section covers the low-level APIs provided by Liger Kernel for fine-grained control over model optimizations.

Model Kernels
------------

RMSNorm
~~~~~~~
.. automodule:: liger_kernel.transformers.rms_norm
   :members:
   :undoc-members:
   :show-inheritance:

Rotary Position Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: liger_kernel.transformers.rope
   :members:
   :undoc-members:
   :show-inheritance:

SwiGLU Activation
~~~~~~~~~~~~~~~~
.. automodule:: liger_kernel.transformers.swiglu
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
------------

Here's how to use the low-level APIs directly:

.. code-block:: python

    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.swiglu import LigerSwiGLU

    # Use optimized RMSNorm
    rms_norm = LigerRMSNorm(hidden_size=768, eps=1e-6)

    # Use optimized SwiGLU activation
    swiglu = LigerSwiGLU(hidden_size=768, intermediate_size=3072)
