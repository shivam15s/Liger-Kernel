import inspect

from transformers import AutoConfig, AutoModelForCausalLM

from liger_kernel.transformers.monkey_patch import (
    MODEL_TYPE_TO_APPLY_LIGER_FN,
    _apply_liger_kernel,
)


def _get_model_config(model_dir, **model_init_kwargs):
    """Get the model configuration from a pretrained model directory.

    Args:
        model_dir (str): The directory containing the pretrained model.
        **model_init_kwargs: Additional keyword arguments to pass to AutoConfig.from_pretrained.

    Returns:
        transformers.PretrainedConfig: The model configuration.
    """
    config = AutoConfig.from_pretrained(model_dir, **model_init_kwargs)
    return config


class AutoLigerKernelForCausalLM(AutoModelForCausalLM):
    """A drop-in replacement for AutoModelForCausalLM that applies Liger Kernel optimizations.

    This class automatically applies Liger Kernel optimizations to supported models, providing:
    - 20% increase in multi-GPU training throughput
    - 60% reduction in memory usage
    - Support for multiple model architectures (LLaMA, Mistral, Mixtral, Gemma, Qwen2, Phi3)

    The optimizations include:
    - Efficient rotary position embeddings
    - Optimized RMSNorm implementation
    - Fused linear cross entropy loss
    - Optimized SwiGLU activation

    Examples:
        >>> from liger_kernel.transformers import AutoLigerKernelForCausalLM
        >>> # Load and optimize a model automatically
        >>> model = AutoLigerKernelForCausalLM.from_pretrained("path/to/model")
        >>> # The model will automatically use Liger Kernel optimizations
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load a pretrained model and apply Liger Kernel optimizations.

        This method extends the standard from_pretrained method to automatically apply
        Liger Kernel optimizations based on the model type.

        Args:
            pretrained_model_name_or_path (str): Path to pretrained model or model identifier from huggingface.co/models
            *model_args: Additional positional arguments passed to the underlying model
            **kwargs: Additional keyword arguments passed to the underlying model and optimization functions.
                     Common optimization kwargs include:
                     - rope (bool): Whether to apply optimized rotary position embedding
                     - rms_norm (bool): Whether to apply optimized RMSNorm
                     - swiglu (bool): Whether to apply optimized SwiGLU activation
                     - fused_linear_cross_entropy (bool): Whether to use fused linear cross entropy loss

        Returns:
            PreTrainedModel: The loaded and optimized model

        Examples:
            >>> model = AutoLigerKernelForCausalLM.from_pretrained(
            ...     "meta-llama/Llama-2-7b",
            ...     rope=True,
            ...     rms_norm=True,
            ...     swiglu=True,
            ...     fused_linear_cross_entropy=True
            ... )
        """
        model_config = _get_model_config(pretrained_model_name_or_path, **kwargs)

        # Determine the model type and apply the Liger Kernel if applicable
        # Note: _apply_liger_kernel will only pass relevant kwargs to the apply_liger_kernel_to_* function
        model_type = model_config.model_type

        _apply_liger_kernel(model_type, **kwargs)

        # Filter out kwargs that were passed to the apply_liger_* function, which will cause
        # model initialization errors otherwise
        apply_fn = MODEL_TYPE_TO_APPLY_LIGER_FN[model_type]
        apply_fn_signature = inspect.signature(apply_fn)

        applicable_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in apply_fn_signature.parameters
        }

        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **applicable_kwargs
        )
