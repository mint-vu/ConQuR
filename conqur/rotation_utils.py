import functools
import math
import typing

import torch
import tqdm
import transformers
from fast_hadamard_transform import hadamard_transform

import model_utils
import monkeypatch
import quant_utils
import utils
from hadamard_utils import apply_exact_had_to_linear, is_pow2, random_hadamard_matrix


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """Fuse the linear operations in Layernorm into the adjacent linear blocks."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)


def fuse_layer_norms(model):
    # Embedding fusion
    W = model.model.embed_tokens
    W_ = W.weight.data.double()
    W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in model.model.layers:
        fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
        fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])

    fuse_ln_linear(model.model.norm, [model.lm_head])

    model_utils.replace_modules(
        model,
        transformers.models.llama.modeling_llama.LlamaRMSNorm,
        lambda _: model_utils.RMSN(model.config.hidden_size),
    )


def get_orthogonal_matrix(size, mode, device=utils.DEV):
    if mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    raise ValueError(f'Unknown mode {mode}')


def rotate_embeddings(model, Q: torch.Tensor) -> None:
    W = model.model.embed_tokens
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, Q) -> None:
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, Q) -> None:
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, Q):
    for W in [layer.mlp.up_proj, layer.mlp.gate_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, Q, use_r4):
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if use_r4:
        # apply exact (inverse) hadamard on the weights of mlp output
        apply_exact_had_to_linear(W, had_dim=-1, output=False)


def rotate_head(model, Q: torch.Tensor) -> None:
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_dim, kv_head, l_id, r2s):
    """Fold the per-KV-head rotation R2 into v_proj (output side) and o_proj (input side)."""
    Q = r2s[f"model.layers.{l_id}.self_attn.R2"].to(device=utils.DEV, dtype=torch.float64)
    apply_multi_head_rotate(layer.self_attn.v_proj, Q, head_dim, kv_head, output=True)
    apply_multi_head_rotate(layer.self_attn.o_proj, Q, head_dim, kv_head, output=False)


def apply_multi_head_rotate(module, Q, head_dim, kv_head, output=False):
    assert isinstance(module, torch.nn.Linear)

    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    num_head = init_shape[1] // head_dim
    n_rep = num_head // kv_head
    W_ = W_.to(device=utils.DEV, dtype=torch.float64)

    if output:
        W_ = W_.t()
        transposed_shape = W_.shape
        W_ = W_.reshape(-1, kv_head, head_dim).transpose(0, 1)
        W_ = torch.matmul(W_, Q)
        W_ = W_.transpose(0, 1).reshape(transposed_shape).t()
    else:
        W_ = W_.reshape(-1, init_shape[1] // head_dim, head_dim).transpose(0, 1)
        if len(Q.shape) == 3:
            Q = Q[:, None, :, :].expand(kv_head, n_rep, head_dim, head_dim)
            Q = Q.reshape(num_head, head_dim, head_dim)
        W_ = torch.matmul(W_, Q)
        W_ = W_.transpose(0, 1).reshape(init_shape)

    module.weight.data = W_.to(device=dev, dtype=dtype)


@torch.inference_mode()
def rotate_model(model, args):
    """Fold R1 (shared) and R2 (per layer) into the weights; optionally fuse the inverse of R4 into down_proj."""
    Q = torch.load(args.r1_path, map_location=torch.device(utils.DEV))['R1'].to(dtype=torch.float64)
    r2s = torch.load(args.r2_path, map_location='cpu')

    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    kv_head = config.num_key_value_heads

    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    utils.cleanup_memory()
    layers = model.model.layers
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layer, Q)
        rotate_attention_output(layer, Q)
        rotate_mlp_input(layer, Q)
        rotate_mlp_output(layer, Q, args.use_r4)
        rotate_ov_proj(layer, head_dim, kv_head, idx, r2s)


class QKRotationWrapper(torch.nn.Module):
    """Applies the online R3 Hadamard to Q/K after RoPE and quantizes K."""

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        head_dim = config.hidden_size // config.num_attention_heads
        assert is_pow2(head_dim), 'Only power of 2 head_dim is supported for K-cache Quantization!'
        assert kwargs['k_groupsize'] in [-1, head_dim], \
            f'Only token-wise/{head_dim}g quantization is supported for K-cache'
        self.func = func
        self.k_bits = kwargs['k_bits']
        self.k_groupsize = kwargs['k_groupsize']
        self.k_sym = kwargs['k_sym']
        self.k_clip_ratio = kwargs['k_clip_ratio']
        self.use_r3 = kwargs['use_r3']
        self.k_quantizer = quant_utils.ActQuantizer()
        # groupsize=-1 means token-wise quantization; head-wise quantization is handled in forward().
        self.k_quantizer.configure(bits=self.k_bits, groupsize=-1, sym=self.k_sym, clip_ratio=self.k_clip_ratio)

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        if self.use_r3:
            q = hadamard_transform(q.float(), scale=1 / math.sqrt(q.shape[-1])).to(dtype)
            k = hadamard_transform(k.float(), scale=1 / math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            self.k_quantizer.find_params(token_wise_k)
            k = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
        else:  # head-wise quantization
            per_head_k = k.reshape(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)

        self.k_quantizer.free()
        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module, "forward", function_name, functools.partial(QKRotationWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)
