import logging

import torch
import transformers

LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer


def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    raise ValueError(f'Unknown model type {model}')


def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization!
    pass


def get_model(model_name):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # Untie the embeddings (e.g. Llama-3.2-3B) so R1 can be folded into embed_tokens and lm_head separately.
    config = transformers.AutoConfig.from_pretrained(model_name)
    untie = config.tie_word_embeddings
    config.tie_word_embeddings = False

    model = transformers.LlamaForCausalLM.from_pretrained(
        model_name, config=config, torch_dtype='auto', low_cpu_mem_usage=True,
    )
    if untie:
        model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight.detach().clone())
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def replace_modules(root: torch.nn.Module, type_to_replace, new_module_factory) -> None:
    """Depth-first replacement of all modules of type `type_to_replace` with `new_module_factory(module)`."""
    for name, module in root.named_children():
        if isinstance(module, type_to_replace):
            setattr(root, name, new_module_factory(module))
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory)


class RMSN(torch.nn.Module):
    """
    RMSNorm without a scale (the scale is fused into the following linear layers).
    Implementation from LlamaRMSNorm:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)
