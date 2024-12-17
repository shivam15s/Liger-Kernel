High-level APIs
==============

This section covers the high-level APIs provided by Liger Kernel for easy integration with existing models.

AutoModel
---------
.. automodule:: liger_kernel.transformers.auto_model
   :members:
   :undoc-members:
   :show-inheritance:

Patching APIs
------------
.. automodule:: liger_kernel.transformers.monkey_patch
   :members:
   :undoc-members:
   :show-inheritance:

Model Optimization
----------------

The high-level APIs provide simple interfaces to optimize your models:

.. code-block:: python

    from liger_kernel.transformers import AutoLigerKernelForCausalLM

    # Load and optimize a model automatically
    model = AutoLigerKernelForCausalLM.from_pretrained("path/to/model")

    # Or use selective patching
    from liger_kernel.transformers import apply_liger_kernel_to_llama

    apply_liger_kernel_to_llama(
        rope=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True,
        swiglu=True
    )
