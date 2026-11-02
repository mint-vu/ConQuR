"""Online hypercube-corner rotation calibration (ConQuR, Algorithm 1).

Learns R1 (shared, hidden_size x hidden_size) and R2 (per layer, block diagonal
over KV heads) by streaming calibration sequences through the model layer by
layer and applying closed-form orthogonal Procrustes updates (OPU) that pull
normalized activations toward the corners of the inscribed hypercube.

With --act_bits < 16 (and optionally --k_bits/--v_bits < 16) the activations
are fake-quantized during calibration ("quant. calib."); with 16 bits the
calibration runs in full precision ("f.p. calib.").
"""
import argparse
import gc
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import datasets
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

import hadamard_utils
import model_utils
import rotation_utils
import utils


ACCUM_DTYPE = torch.float32


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        trainloader.append(trainenc.input_ids[:, i:j].long())
    return trainloader


@torch.no_grad()
def compute_cross_cov(target: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Z^T X~ for row-normalized targets/activations. target, x: [N, d]."""
    target = target.to(ACCUM_DTYPE)
    x = x.to(ACCUM_DTYPE)
    t_norm = torch.linalg.norm(target, dim=1, keepdim=True).clamp_min(eps)
    x_norm = torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(eps)
    return (target / t_norm).T @ (x / x_norm)


@torch.no_grad()
def compute_cross_cov_per_head(target: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-head Z^T X~. target, x: [H, N, Dh] -> [H, Dh, Dh]."""
    target = target.to(ACCUM_DTYPE)
    x = x.to(ACCUM_DTYPE)
    t_norm = torch.linalg.norm(target, dim=2, keepdim=True).clamp_min(eps)
    x_norm = torch.linalg.norm(x, dim=2, keepdim=True).clamp_min(eps)
    return torch.matmul((target / t_norm).transpose(1, 2), (x / x_norm))


@torch.no_grad()
def procrustes(c: torch.Tensor) -> torch.Tensor:
    """Closed-form orthogonal Procrustes solution U V^T of C = U S V^T. Supports [d, d] or [H, d, d]."""
    u, _, vh = torch.linalg.svd(c, full_matrices=False, driver="gesvd")
    return (u @ vh).float()


def reshape_per_head(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    return x.reshape(*x.shape[:-1], n_heads, head_dim)


def apply_rotation_blocks(xh: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """xh: [..., H, Dh], r: [H, Dh, Dh] -> headwise xh @ r."""
    return torch.einsum("...hd,hdf->...hf", xh, r)


@torch.no_grad()
def fake_quant_token_asym(x: torch.Tensor, n_bits: int, clip_ratio: float, eps: float = 1e-8) -> torch.Tensor:
    """Asymmetric per-token fake quantization over the last dimension."""
    qmin = 0.0
    qmax = float((1 << n_bits) - 1)

    xmin = x.amin(dim=-1, keepdim=True)
    xmax = x.amax(dim=-1, keepdim=True)
    if clip_ratio < 1.0:
        xmin = xmin * clip_ratio
        xmax = xmax * clip_ratio

    degenerate = (xmax - xmin) < eps
    xmax = torch.where(degenerate, xmin + eps, xmax)

    scale = ((xmax - xmin) / (qmax - qmin)).clamp_min(eps)
    zero_point = (qmin - xmin / scale).round().clamp(qmin, qmax)

    q = (x / scale + zero_point).round().clamp(qmin, qmax)
    return (q - zero_point) * scale


@torch.no_grad()
def quant_shared(x: torch.Tensor, r: torch.Tensor, n_bits: int, clip_ratio: float) -> torch.Tensor:
    """Q(x R^T) R for a full-dimension rotation r: [D, D]."""
    x_q = fake_quant_token_asym(x.float() @ r.T, n_bits=n_bits, clip_ratio=clip_ratio)
    return (x_q @ r).to(x.dtype)


@torch.no_grad()
def quant_per_head(x: torch.Tensor, r: torch.Tensor, n_bits: int, clip_ratio: float) -> torch.Tensor:
    """Headwise Q(x_h R_h^T) R_h for block-diagonal r: [H, Dh, Dh]."""
    n_heads, head_dim = r.shape[0], r.shape[-1]
    xh = reshape_per_head(x.float(), n_heads, head_dim)
    xh_rot = apply_rotation_blocks(xh, r.transpose(1, 2))
    xh_q = fake_quant_token_asym(
        xh_rot.reshape(-1, head_dim), n_bits=n_bits, clip_ratio=clip_ratio
    ).reshape(xh_rot.shape)
    return apply_rotation_blocks(xh_q, r).reshape(x.shape).to(x.dtype)


@torch.no_grad()
def quant_hadamard(x: torch.Tensor, n_bits: int, clip_ratio: float) -> torch.Tensor:
    """Online Hadamard (R4) + quantization for down_proj; the inverse is fused into the weights."""
    had_k, k = hadamard_utils.get_hadK(x.shape[-1])
    x_rot = hadamard_utils.matmul_hadU_cuda(x.float(), had_k, k)
    return fake_quant_token_asym(x_rot, n_bits=n_bits, clip_ratio=clip_ratio).to(x.dtype)


class FirstLayerCatcher(nn.Module):
    def __init__(self, module: nn.Module, storage: Dict):
        super().__init__()
        self.module = module
        self.storage = storage
        self.self_attn = module.self_attn

    def forward(self, inp, **kwargs):
        self.storage["inp"] = inp.detach().cpu()
        self.storage["kwargs"] = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        raise ValueError


@dataclass
class CalibConfig:
    model: str = "meta-llama/Llama-2-7b-hf"
    nsamples: int = 128
    seqlen: int = 2048
    seed: int = 0
    batch_size: int = 1
    output_path: str = "rotations.pt"

    act_bits: int = 16
    clip_ratio: float = 1.0
    k_bits: int = 16
    v_bits: int = 16
    k_groupsize: int = -1
    k_clip_ratio: float = 1.0
    v_clip_ratio: float = 1.0


class OnlineRotationSolver:
    R1_CAPTURE_NAMES = ("self_attn.q_proj", "mlp.gate_proj")  # attention / MLP inputs used for R1
    O_PROJ_NAME = "self_attn.o_proj"                         # o_proj input used for R2
    SHARED_QUANT_NAMES = {
        "self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj", "mlp.up_proj",
    }

    def __init__(self, model, dataloader, cfg: CalibConfig):
        self.model = model
        self.dataloader = iter(dataloader)
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.eval()
        self.use_cache = model.config.use_cache
        model.config.use_cache = False

        if model_utils.model_type_extractor(model) != model_utils.LLAMA_MODEL:
            raise ValueError("Only Llama-style models are supported.")

        self.layers = model.model.layers
        self.embed_tokens = model.model.embed_tokens
        self.norm = model.model.norm
        self.rotary_emb = getattr(model.model, "rotary_emb", None)

        self.embed_tokens.to(self.device)
        self.norm.to(self.device)
        if self.rotary_emb is not None:
            self.rotary_emb.to(self.device)

        # Fuse the inverse of the online R4 Hadamard into down_proj (matches inference).
        if cfg.act_bits < 16:
            for layer in self.layers:
                hadamard_utils.apply_exact_had_to_linear(layer.mlp.down_proj, had_dim=-1, output=False)

        self.first_layer_storage: Dict = {}
        self.first_layer = self.layers[0]
        self.layers[0] = FirstLayerCatcher(self.first_layer.to(self.device), self.first_layer_storage)

        # Online R3 Hadamard + K-cache quantization after RoPE.
        if cfg.k_bits < 16:
            for layer in self.layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn, "apply_rotary_pos_emb", config=model.config,
                    k_bits=cfg.k_bits, k_groupsize=cfg.k_groupsize,
                    k_sym=False, k_clip_ratio=cfg.k_clip_ratio,
                    use_r3=True,
                )

        self.hidden_size = model.config.hidden_size
        self.n_layers = len(self.layers)
        self.n_heads = model.config.num_attention_heads
        self.kv_head = model.config.num_key_value_heads
        assert self.n_heads % self.kv_head == 0, "num_attention_heads must be divisible by num_key_value_heads"
        assert self.hidden_size % self.n_heads == 0, "hidden_size must be divisible by num_attention_heads"
        self.head_dim = self.hidden_size // self.n_heads

        # Hadamard initialization.
        self.r1 = rotation_utils.get_orthogonal_matrix(self.hidden_size, "hadamard", self.device).float()
        r2_base = rotation_utils.get_orthogonal_matrix(self.head_dim, "hadamard", self.device).float()
        self.r2 = r2_base.unsqueeze(0).repeat(self.n_layers, self.kv_head, 1, 1)  # [L, KV, Dh, Dh]

    def _expand_r2(self, r2_layer):
        """[KV, Dh, Dh] -> [H, Dh, Dh] (shared across the query heads of a GQA group)."""
        return r2_layer.repeat_interleave(self.n_heads // self.kv_head, dim=0)

    @torch.no_grad()
    def _capture_first_layer_input(self, input_ids: torch.Tensor):
        self.first_layer_storage.clear()
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        try:
            self.model(input_ids.to(self.device))
        except ValueError:
            pass
        else:
            raise RuntimeError("First layer catcher did not trigger.")
        return self.first_layer_storage["inp"][0], self.first_layer_storage["kwargs"]

    @torch.no_grad()
    def _next_batch(self, remaining: int) -> Tuple[List[torch.Tensor], List[Dict]]:
        batch_hidden, batch_kwargs = [], []
        for _ in range(min(self.cfg.batch_size, remaining)):
            hidden, kwargs = self._capture_first_layer_input(next(self.dataloader))
            batch_hidden.append(hidden)
            batch_kwargs.append(kwargs)
        return batch_hidden, batch_kwargs

    @torch.no_grad()
    def _accumulate_r1_stats(self, acts: List[torch.Tensor], c_shared: torch.Tensor):
        for x in acts:
            x = x.reshape(-1, x.shape[-1]).float().to(self.device)
            target = (x @ self.r1.T).sign()  # nearest hypercube corner (up to scale)
            c_shared += compute_cross_cov(target, x).to(torch.float64)

    @torch.no_grad()
    def _r2_stats(self, acts: List[torch.Tensor], layer_idx: int) -> torch.Tensor:
        x = torch.cat([a.reshape(-1, a.shape[-1]).float() for a in acts], dim=0).to(self.device)  # [N, D]
        xh = reshape_per_head(x, self.n_heads, self.head_dim)                                     # [N, H, Dh]
        y = apply_rotation_blocks(xh, self._expand_r2(self.r2[layer_idx]).transpose(1, 2))
        target = y.sign()

        target_batch = target.transpose(0, 1).contiguous()  # [H, N, Dh]
        x_batch = xh.transpose(0, 1).contiguous()
        return compute_cross_cov_per_head(target_batch, x_batch).to(torch.float64)

    def _make_quant_pre_hook(self, name: str, layer_idx: int):
        cfg = self.cfg
        if name == self.O_PROJ_NAME:
            def hook(module, inputs):
                x = quant_per_head(inputs[0], self._expand_r2(self.r2[layer_idx]), cfg.act_bits, cfg.clip_ratio)
                return (x,) + inputs[1:]
        elif name == "mlp.down_proj":
            def hook(module, inputs):
                return (quant_hadamard(inputs[0], cfg.act_bits, cfg.clip_ratio),) + inputs[1:]
        else:
            def hook(module, inputs):
                return (quant_shared(inputs[0], self.r1, cfg.act_bits, cfg.clip_ratio),) + inputs[1:]
        return hook

    def _make_v_proj_hook(self, layer_idx: int):
        cfg = self.cfg

        def hook(module, inputs, out):
            return quant_per_head(out, self.r2[layer_idx], cfg.v_bits, cfg.v_clip_ratio)
        return hook

    @torch.no_grad()
    def _run_layer(self, layer_idx, decoder_layer, hidden, batch_kwargs):
        """Run one decoder layer on the batch; return its outputs and the clean (pre-quant) inputs."""
        captured = {name: [] for name in self.R1_CAPTURE_NAMES + (self.O_PROJ_NAME,)}
        hooks = []

        # Capture hooks are registered first so they see the input before it is quantized.
        for name, module in decoder_layer.named_modules():
            if isinstance(module, nn.Linear) and name in captured:
                hooks.append(module.register_forward_pre_hook(
                    lambda mod, inp, name=name: captured[name].append(inp[0].detach().cpu().squeeze(0))
                ))

        # Note: v_proj only has its output (V-cache) quantized here; its input is left in full precision.
        for name, module in decoder_layer.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name == "self_attn.v_proj":
                if self.cfg.v_bits < 16:
                    hooks.append(module.register_forward_hook(self._make_v_proj_hook(layer_idx)))
            elif self.cfg.act_bits < 16 and (
                name in self.SHARED_QUANT_NAMES or name in (self.O_PROJ_NAME, "mlp.down_proj")
            ):
                hooks.append(module.register_forward_pre_hook(self._make_quant_pre_hook(name, layer_idx)))

        next_hidden = []
        for sample_hidden, kwargs_cpu in zip(hidden, batch_kwargs):
            kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs_cpu.items()}
            kwargs.pop("past_key_value", None)
            out = decoder_layer(sample_hidden.to(self.device).unsqueeze(0), **kwargs)[0]
            next_hidden.append(out.squeeze(0).detach().cpu())

        for hook in hooks:
            hook.remove()

        for name, acts in captured.items():
            if len(acts) != len(hidden):
                raise RuntimeError(f"Expected {len(hidden)} captures for {name}, got {len(acts)}")
        return next_hidden, captured

    @torch.no_grad()
    def solve(self):
        num_done = 0
        pbar = tqdm(total=self.cfg.nsamples, desc="Calibrating rotations")
        group_size = self.n_heads // self.kv_head

        while num_done < self.cfg.nsamples:
            batch_hidden, batch_kwargs = self._next_batch(self.cfg.nsamples - num_done)
            batch_n = len(batch_hidden)
            is_last = (num_done + batch_n) >= self.cfg.nsamples

            c_shared = torch.zeros(self.hidden_size, self.hidden_size, device=self.device, dtype=torch.float64)
            hidden = batch_hidden

            for layer_idx in range(self.n_layers):
                decoder_layer = self.first_layer if layer_idx == 0 else self.layers[layer_idx]
                decoder_layer = decoder_layer.to(self.device)

                next_hidden, captured = self._run_layer(layer_idx, decoder_layer, hidden, batch_kwargs)

                if layer_idx != 0:
                    decoder_layer.to("cpu")

                # R1 statistics accumulate over all layers' attention and MLP inputs.
                for name in self.R1_CAPTURE_NAMES:
                    self._accumulate_r1_stats(captured[name], c_shared)

                # R2 is updated immediately for this layer (one Procrustes problem per KV head).
                c_block = self._r2_stats(captured[self.O_PROJ_NAME], layer_idx)
                c_block = c_block / torch.linalg.norm(c_block, dim=(1, 2), keepdim=True).clamp_min(1e-8)
                c_block = c_block.view(self.kv_head, group_size, self.head_dim, self.head_dim).mean(dim=1)
                self.r2[layer_idx] = procrustes(c_block.double() if is_last else c_block.float())

                hidden = next_hidden
                del captured, next_hidden
                torch.cuda.empty_cache()
                gc.collect()

            c_shared = c_shared / torch.linalg.norm(c_shared).clamp_min(1e-8)
            self.r1 = procrustes(c_shared.double() if is_last else c_shared.float())

            num_done += batch_n
            pbar.update(batch_n)

            ortho_r1 = ((self.r1 @ self.r1.T) - torch.eye(self.hidden_size, device=self.device)).pow(2).sum().item()
            ortho_r2 = (
                (self.r2 @ self.r2.transpose(2, 3))
                - torch.eye(self.head_dim, device=self.device).view(1, 1, self.head_dim, self.head_dim)
            ).pow(2).sum(dim=(2, 3)).mean().item()
            pbar.set_postfix(r1_ortho=f"{ortho_r1:.2e}", r2_ortho=f"{ortho_r2:.2e}")

        pbar.close()

        # Stored transposed: the evaluation code folds W @ R1 / W @ R2 into the weights.
        save_dict = {"R1": self.r1.detach().cpu().T}
        for layer_idx in range(self.n_layers):
            save_dict[f"model.layers.{layer_idx}.self_attn.R2"] = self.r2[layer_idx].detach().cpu().transpose(1, 2)
        torch.save(save_dict, self.cfg.output_path)
        return save_dict

    def close(self):
        self.layers[0] = self.first_layer.cpu()
        self.embed_tokens.cpu()
        self.norm.cpu()
        if self.rotary_emb is not None:
            self.rotary_emb.cpu()
        self.model.config.use_cache = self.use_cache
        torch.cuda.empty_cache()
        gc.collect()


def parse_args() -> CalibConfig:
    parser = argparse.ArgumentParser(description="ConQuR online rotation calibration")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", choices=utils.supported_models)
    parser.add_argument("--nsamples", type=int, default=128, help="Number of WikiText-2 calibration sequences.")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1, help="Sequences per Procrustes update.")
    parser.add_argument("--output_path", type=str, default="rotations.pt")
    parser.add_argument("--act_bits", type=int, default=16,
                        help="Activation bits used during calibration (16 = full-precision calibration).")
    parser.add_argument("--clip_ratio", type=float, default=1.0, help="Activation clip ratio during calibration.")
    parser.add_argument("--k_bits", type=int, default=16)
    parser.add_argument("--v_bits", type=int, default=16)
    parser.add_argument("--k_groupsize", type=int, default=-1)
    parser.add_argument("--k_clip_ratio", type=float, default=1.0)
    parser.add_argument("--v_clip_ratio", type=float, default=1.0)
    return CalibConfig(**vars(parser.parse_args()))


@torch.no_grad()
def main():
    cfg = parse_args()
    print(cfg)
    transformers.set_seed(cfg.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model)
    dataloader = get_wikitext2(nsamples=cfg.nsamples, seed=cfg.seed, seqlen=cfg.seqlen, tokenizer=tokenizer)

    model = model_utils.get_model(cfg.model)
    rotation_utils.fuse_layer_norms(model)

    solver = OnlineRotationSolver(model=model, dataloader=dataloader, cfg=cfg)
    try:
        solver.solve()
        print(f"Saved rotations to {cfg.output_path}")
    finally:
        solver.close()


if __name__ == "__main__":
    main()
