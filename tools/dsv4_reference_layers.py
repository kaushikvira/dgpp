#!/usr/bin/env python3
"""Per-layer torch reference dump for DeepSeek-V4-Flash (dgpp numerics bisection).

Computes the checkpoint's OWN forward — the shipped inference/model.py
Transformer math (the mHC folds, the CSA2 attention's window + compressed
sources, the 64-head fp4 indexer, the C4A/C128A compressors, the
sqrtsoftplus-routed MoE with its MXFP4 experts and fp8 shared expert, the
head's weighted collapse) — over a prompt, loading tensors PER LAYER out of
the 155 GiB safetensors shards (partial reads, one layer at a time, freed
afterwards), and writes per-layer hidden-state artifacts with the same
layout the dgpp engine's per-layer dump uses, so a plain byte diff localizes
the first diverging layer:

  <out>/layer_<NN>.f32       post-layer NN state at the LAST prompt position
                             (NN 0-indexed, matching the checkpoint's
                             `layers.NN.` names): the `--state-layout`
                             selection — `mean`: the hc_mult stream mean
                             (hidden_size f32), `full`: all hc_mult streams
                             (hc_mult * hidden_size f32, copy-major)
  <out>/layer_<NN>_hc.f32    the full hc_mult-stream state (hc_mult *
                             hidden_size f32, copy-major); written only for
                             the `mean` layout (there the primary file is the
                             mean, this one carries the full state)
  <out>/logits_top.txt       top-<k> (token id, logit) at the last position
                             (only when the last layer is in the run range)
  <out>/meta.txt             prompt ids, hidden_size, num_hidden_layers, the
                             dtypes, the kernel mode, the hadamard source

The exotic kernels (kernel.py's tilelang act_quant / fp4_act_quant /
fp8_gemm / fp4_gemm / sparse_attn / hc_split_sinkhorn and
fast_hadamard_transform's randomized Hadamard) are the reference's own math
when their packages import cleanly (--kernels native, or auto when present);
otherwise this script injects a pure-torch fallback that implements the SAME
math (per-block quantize-dequantize, the fp32-accumulated GEMMs, the
sinkhorn, the explicit Walsh-Hadamard rotation) so the script runs without
them. See the `kernel mode:` line in meta.txt for which one a run used.

One load-time fix beyond the reference's load_model: the 0731 checkpoint
stores attn.wo_a quantized (FP8 e4m3 + a per-128x128-tile e8m0 scale), but
model.py's wo_a is a bf16 Linear whose plain F.linear forward expects
DEQUANTIZED values (what inference/convert.py writes for the bf16 build).
load_model's raw fp8->bf16 copy_ would leave the unscaled e4m3 codes in the
parameter (attention output inflated by ~1/scale, about 2e3, state
magnitudes exploding from layer 2, incoherent logits); this script applies
the scale at load time. Flagged in meta.txt (`wo_a:` line).

Usage (inside the eugr/spark-vllm container, checkpoint at /ckpt):
  python3 dsv4_reference_layers.py --ckpt /ckpt --out /out \
      --token-ids "97873,7400,271" --layers 0-3
  python3 dsv4_reference_layers.py --prompt-text "Hello, world!"   # tokenized
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import types

# ---------------------------------------------------------------------------
# pure-torch fallback for the checkpoint's kernel.py (tilelang) and for
# fast_hadamard_transform. Built after the torch import (main()); injected
# into sys.modules so model.py's `from kernel import ...` and its lazy
# `from fast_hadamard_transform import hadamard_transform` pick them up.
# The math mirrors kernel.py's kernels: per-block (128 for fp8, 32 for fp4)
# quantize-dequantize with the ue8m0 power-of-2 scales (the fast_log2_ceil
# bit-trick), the fp32-accumulated scaled GEMMs, the bf16-rounded sparse
# attention with the attn_sink, and the sinkhorn iteration.
# ---------------------------------------------------------------------------


def build_torch_kernel_module(torch, FP4_TABLE):
    mod = types.ModuleType("kernel")

    def pow2_ceil(x):
        """2^ceil(log2(x)) from the fp32 bit fields (kernel.py fast_round_scale)."""
        xb = x.contiguous().view(torch.int32)
        exp = (xb >> 23) & 0xFF
        man = xb & 0x7FFFFF
        return torch.exp2(exp - 127 + (man != 0).to(torch.float32))

    def e8m0_decode(s):
        """e8m0 (power-of-2) scale bytes -> fp32 values."""
        u = s.contiguous().view(torch.uint8).to(torch.float32)
        return torch.where(u == 0, torch.zeros_like(u), torch.exp2(u - 127))

    def scale_to_fp32(s, scale_dtype):
        if scale_dtype == torch.float8_e8m0fnu:
            return e8m0_decode(s)
        return s.to(torch.float32)

    def pow2_to_e8m0(s):
        """Exact power-of-2 fp32 values -> e8m0 bytes (the kernel's scale store)."""
        try:
            return s.to(torch.float8_e8m0fnu)
        except RuntimeError:
            k = torch.floor(torch.log2(s)).long() + 127
            return k.clamp(0, 255).to(torch.uint8).contiguous().view(torch.float8_e8m0fnu)

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        """Block-wise FP8 quantize-dequantize (kernel.py act_quant_kernel)."""
        shape, n = x.shape, x.shape[-1]
        assert n % block_size == 0
        xf = x.reshape(-1, n).float()
        blk = xf.unflatten(1, (-1, block_size))
        amax = blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
        if scale_fmt is not None:  # ue8m0: power-of-2 scale
            s = pow2_ceil(amax * torch.tensor(1.0 / 448.0, dtype=torch.float32, device=x.device))
            s_store = pow2_to_e8m0(s) if scale_dtype == torch.float8_e8m0fnu else s
            s_use = scale_to_fp32(s_store, scale_dtype)
        else:
            s = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32, device=x.device)
            s_store, s_use = s, s
        q = (blk / s_use).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        if inplace:
            x.copy_((q.float() * s_use).flatten(1).reshape(shape).to(x.dtype))
            return x
        return q.reshape(shape).contiguous(), s_store.reshape(shape[:-1] + (n // block_size,))

    def e2m1_round(v):
        """Nearest e2m1 value on {0,.5,1,1.5,2,3,4,6} (ties to the even code)."""
        a = v.abs()
        out = torch.full_like(a, 6.0)
        out = torch.where(a <= 5.0, torch.full_like(a, 4.0), out)
        out = torch.where(a < 3.5, torch.full_like(a, 3.0), out)
        out = torch.where(a <= 2.5, torch.full_like(a, 2.0), out)
        out = torch.where(a < 1.75, torch.full_like(a, 1.5), out)
        out = torch.where(a <= 1.25, torch.full_like(a, 1.0), out)
        out = torch.where(a < 0.75, torch.full_like(a, 0.5), out)
        out = torch.where(a <= 0.25, torch.zeros_like(a), out)
        return out * torch.where(v < 0, torch.full_like(a, -1.0), torch.ones_like(a))

    def fp4_act_quant(x, block_size=32, inplace=False):
        """Block-wise FP4 quantize-dequantize (kernel.py fp4_quant_kernel)."""
        shape, n = x.shape, x.shape[-1]
        assert n % block_size == 0
        xf = x.reshape(-1, n).float()
        blk = xf.unflatten(1, (-1, block_size))
        amax = blk.abs().amax(dim=-1, keepdim=True).clamp_min(
            torch.tensor(6.0 * (2.0 ** -126), dtype=torch.float32, device=x.device))
        s = pow2_ceil(amax * torch.tensor(1.0 / 6.0, dtype=torch.float32, device=x.device))
        s_use = e8m0_decode(pow2_to_e8m0(s))  # the kernel stores e8m0 scales
        q = e2m1_round((blk / s_use).clamp(-6.0, 6.0))
        if inplace:
            x.copy_((q * s_use).flatten(1).reshape(shape).to(x.dtype))
            return x
        return ((q * s_use).flatten(1).reshape(shape),
                pow2_to_e8m0(s).reshape(shape[:-1] + (n // block_size,)))

    def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
        """C = A @ B^T with per-128-block fp8 scales (kernel.py fp8_gemm)."""
        k, n = a.shape[-1], b.shape[0]
        lead = a.shape[:-1]
        a_dq = a.float().unflatten(-1, (-1, 128)) * scale_to_fp32(a_s, scale_dtype).view(lead + (k // 128, 1))
        b_dq = (b.float().unflatten(-1, (-1, 128))
                * scale_to_fp32(b_s, scale_dtype).repeat_interleave(128, dim=0).unsqueeze(-1)).reshape(n, k)
        c = a_dq.reshape(-1, k) @ b_dq.t()
        return c.view(*lead, n).to(torch.get_default_dtype())

    def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
        """C = A_fp8 @ B_fp4^T (kernel.py fp4_gemm; B packed 2 fp4/byte along K)."""
        k, n = a.shape[-1], b.shape[0]
        lead = a.shape[:-1]
        a_dq = a.float().unflatten(-1, (-1, 128)) * scale_to_fp32(a_s, scale_dtype).view(lead + (k // 128, 1))
        packed = b.view(torch.uint8) if b.dtype != torch.uint8 else b
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        vals = torch.stack([FP4_TABLE[low.long().to(FP4_TABLE.device)],
                            FP4_TABLE[high.long().to(FP4_TABLE.device)]], dim=-1).flatten(1, 2)
        sc = scale_to_fp32(b_s, scale_dtype).repeat_interleave(32, dim=-1)
        c = a_dq.reshape(-1, k) @ (vals * sc).t()
        return c.view(*lead, n).to(torch.get_default_dtype())

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        """Sparse attention over the top-k KV rows (kernel.py sparse_attn_kernel).
        Mirrors the kernel's rounding: fp32-accumulated QK, the probabilities
        rounded to bf16 before the PV product, the unrounded fp32 denominator
        with the attn_sink term."""
        b, s, h, d = q.shape
        topk = topk_idxs.shape[-1]
        outs = []
        for i in range(0, s, 256):
            j = slice(i, min(i + 256, s))
            idxs = topk_idxs[:, j].long()
            safe = idxs.clamp(min=0)
            kv_g = kv.unsqueeze(2).expand(b, -1, topk, d).gather(1, safe.unsqueeze(-1).expand(b, -1, topk, d))
            valid = (idxs >= 0).unsqueeze(2)  # [b, chunk, 1, topk]
            sc = torch.einsum("bshd,bstd->bsht", q[:, j].float(), kv_g.float()) * softmax_scale
            sc = sc.masked_fill(~valid, float("-inf"))
            m = sc.amax(dim=-1, keepdim=True)
            pr = (sc - m).exp()
            den = pr.sum(dim=-1) + torch.exp(attn_sink.unsqueeze(0) - m.squeeze(-1))
            o = torch.einsum("bsht,bstd->bshd", pr.to(torch.bfloat16).float(), kv_g.float())
            outs.append((o / den.unsqueeze(-1)).to(q.dtype))
        return torch.cat(outs, dim=1)

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        """pre / post / comb from the mix logits (kernel.py hc_split_sinkhorn)."""
        orig = mixes.shape
        m = mixes.reshape(-1, orig[-1])
        hc = hc_mult
        pre = torch.sigmoid(m[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
        post = 2.0 * torch.sigmoid(m[:, hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
        comb = (m[:, 2 * hc:].view(-1, hc, hc) * hc_scale[2] + hc_base[2 * hc:].view(hc, hc))
        comb = comb.softmax(dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return (pre.view(*orig[:-1], hc), post.view(*orig[:-1], hc), comb.view(*orig[:-1], hc, hc))

    for name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm", "sparse_attn", "hc_split_sinkhorn"):
        setattr(mod, name, locals()[name])
    return mod


def build_hadamard_stub(torch):
    """Pure-torch stand-in for fast_hadamard_transform.hadamard_transform:
    the seeded random sign flip, the unnormalized Walsh-Hadamard (the exact
    orthogonal rotation the CUDA kernel computes), the scale. The sign
    stream uses the package's default seed 8191 via a CPU generator, so the
    rotation is deterministic across runs. The rotation is applied IDENTICALLY
    to the indexer's q and its compressed kv (same seed, same d), so the
    indexer's scores are invariant to any coordinate convention of the
    transform; only a bit-exact match of the reference package's sign stream
    is approximated (flag it in meta.txt)."""
    mod = types.ModuleType("fast_hadamard_transform")
    _signs = {}

    def _signs_vec(d, seed):
        key = (d, seed)
        if key not in _signs:
            gen = torch.Generator()
            gen.manual_seed(seed)
            _signs[key] = torch.randint(0, 2, (d,), generator=gen, device="cpu") * 2 - 1
        return _signs[key]

    def fwht(x):
        """The unnormalized Walsh-Hadamard over the last dim (dtype-preserving,
        one rounding per stage, like the bf16 CUDA kernel)."""
        n = x.shape[-1]
        h = 1
        while h < n:
            x = x.unflatten(-1, (-1, 2 * h))  # [..., M, 2h]
            a, b = x[..., :h], x[..., h:]      # [..., M, h] each
            x = torch.cat([a + b, a - b], dim=-1).flatten(-2)  # [..., M, 2h] -> [..., n]
            h *= 2
        return x

    def hadamard_transform(x, scale=1.0, num_heads=1, random_seed=8191):
        d = x.shape[-1]
        assert not (d & (d - 1)), "the last dimension must be a power of 2"
        sign = _signs_vec(d, random_seed).to(x.dtype).to(x.device)
        return fwht(x * sign) * scale

    mod.hadamard_transform = hadamard_transform
    return mod


# ---------------------------------------------------------------------------
# the main walk
# ---------------------------------------------------------------------------


def parse_layer_range(spec, n_layers):
    """`0-3` / `5` / `0-42` -> the inclusive list of layer ids (the walk is
    contiguous from 0: a layer's state is its input, so no skipping)."""
    parts = spec.split("-")
    if len(parts) == 1:
        lo = hi = int(parts[0])
    elif len(parts) == 2:
        lo, hi = int(parts[0]), int(parts[1])
    else:
        raise SystemExit("--layers: expected `A` or `A-B`")
    if lo < 0 or hi >= n_layers or lo > hi:
        raise SystemExit(f"--layers {spec}: must lie within 0..{n_layers - 1}")
    if lo != 0:
        raise SystemExit(f"--layers {spec}: the walk must start at layer 0 "
                         "(each layer's state is the next one's input)")
    return list(range(lo, hi + 1))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/ckpt",
                    help="the checkpoint dir: the model-*.safetensors shards + "
                         "config.json + inference/model.py (default /ckpt, the "
                         "container mount)")
    ap.add_argument("--out", required=True,
                    help="the output dir for the layer_<NN>.f32 / logits_top.txt / "
                         "meta.txt artifacts (created if missing)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--token-ids", default="",
                     help='comma-separated prompt token ids, e.g. "97873,7400,271" '
                          "(deterministic; preferred)")
    src.add_argument("--prompt-text", default="",
                     help="prompt text, tokenized with the checkpoint's "
                          "tokenizer.json (mutually exclusive with --token-ids; "
                          "one of the two is required)")
    ap.add_argument("--chat", action="store_true",
                    help="wrap --prompt-text in the checkpoint's chat template "
                         "(encoding/encoding_dsv4.py's encode_messages, "
                         "thinking_mode=chat) exactly as inference/generate.py "
                         "does (the model is a chat model: raw text without the "
                         "template is out-of-distribution and yields a flat, "
                         "incoherent logit distribution)")
    ap.add_argument("--layers", default="0-3",
                    help="the inclusive layer range to walk AND dump, `A-B` or "
                         "`A` (must start at 0; default 0-3, the first layers — "
                         "a cheap run; 0-42 walks the whole backbone and writes "
                         "logits_top.txt)")
    ap.add_argument("--kernels", choices=("auto", "torch", "native"), default="auto",
                    help="auto: the pure-torch fallback (deterministic, CPU-capable; "
                         "default); torch: force the pure-torch fallback; native: "
                         "force the checkpoint's tilelang kernel.py (its sparse_attn "
                         "needs 141 KiB of dynamic shared memory, which the GB10 "
                         "driver rejects — use it only where that launches)")
    ap.add_argument("--state-layout", choices=("mean", "full"), default="mean",
                    help="layer_<NN>.f32's content: mean = the hc_mult stream "
                         "mean (hidden_size floats; the full state goes to "
                         "layer_<NN>_hc.f32), full = all hc_mult streams "
                         "(hc_mult * hidden_size floats, copy-major) (default mean)")
    ap.add_argument("--topk", type=int, default=20,
                    help="the number of (token id, logit) rows in logits_top.txt "
                         "(default 20)")
    ap.add_argument("--seed", type=int, default=0,
                    help="the torch RNG seed (default 0; the walk itself is "
                         "deterministic — no sampling — the seed covers the "
                         "fallback hadamard sign stream and any residual "
                         "randomness)")
    ap.add_argument("--max-seq-len", type=int, default=0,
                    help="the model's max_seq_len (the rope table / kv buffer "
                         "size; default: max(65536, the prompt length), matching "
                         "inference/generate.py's interactive mode)")
    ap.add_argument("--device", default="",
                    help="the compute device (default: cuda:0 when available, "
                         "else cpu)")
    args = ap.parse_args(argv)
    if not args.token_ids and not args.prompt_text:
        ap.error("one of --token-ids / --prompt-text is required")
    return args


def load_module_from_file(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_f32(path, tensor_cpu):
    """A little-endian float32 blob (the dump's raw layout)."""
    with open(path, "wb") as f:
        f.write(tensor_cpu.contiguous().numpy().tobytes())


def main(argv=None):
    args = parse_args(argv)

    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"

    # the prompt
    if args.token_ids:
        ids = [int(t) for t in args.token_ids.replace(" ", "").split(",") if t]
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.ckpt)
        text = args.prompt_text
        if args.chat:
            enc_dir = os.path.join(args.ckpt, "encoding")
            if not os.path.isdir(enc_dir):
                raise SystemExit(f"--chat needs the checkpoint's encoding dir: {enc_dir}")
            sys.path.insert(0, enc_dir)
            from encoding_dsv4 import encode_messages
            text = encode_messages(
                [{"role": "user", "content": args.prompt_text}],
                thinking_mode="chat")
        ids = list(tok.encode(text))
    seqlen = len(ids)
    if seqlen == 0:
        raise SystemExit("empty prompt")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    torch.set_default_device(dev)  # generate.py: the cached index tensors' device
    print(f"prompt: {seqlen} tokens, device {device}, seed {args.seed}", flush=True)

    inf_dir = os.path.join(args.ckpt, "inference")
    if not os.path.isfile(os.path.join(inf_dir, "model.py")):
        raise SystemExit(f"the reference implementation is missing: {inf_dir}/model.py")

    # the reference's ModelArgs from the inference dir's config.json (the flat
    # args file inference/generate.py's `--config` points at; the checkpoint
    # root's HF-style config.json carries extra keys)
    cfg_path = os.path.join(inf_dir, "config.json")
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(args.ckpt, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    # the kernel + hadamard modules (before model.py's imports see them)
    hadamard_mode = "native"
    try:
        import fast_hadamard_transform  # noqa: F401
    except Exception:
        hadamard_mode = "pure-torch fallback (seed 8191)"
        sys.modules.setdefault("fast_hadamard_transform", build_hadamard_stub(torch))

    kernel_mode = "pure-torch fallback"
    if args.kernels == "native":
        try:
            load_module_from_file("kernel", os.path.join(inf_dir, "kernel.py"))
            kernel_mode = "native (tilelang)"
        except Exception as e:
            raise SystemExit(f"native kernel.py unavailable: {e}")
    elif args.kernels == "auto":
        # Determinism first: the pure-torch fallback mirrors kernel.py's math
        # (quantize-dequantize, fp32-accumulated GEMMs) and is CPU-capable;
        # the native tilelang kernels are available with --kernels native where
        # they launch (the GB10 rejects sparse_attn's 141 KiB shared memory).
        print("kernels: auto -> the pure-torch fallback (deterministic); "
              "--kernels native for the tilelang kernels", flush=True)
    if kernel_mode != "native (tilelang)":
        sys.modules["kernel"] = build_torch_kernel_module(
            torch, torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                                dtype=torch.float32, device=dev))
    if kernel_mode == "native (tilelang)":
        print(f"kernels: {kernel_mode}; hadamard: {hadamard_mode}", flush=True)

    # the reference model module (its math IS the reference)
    model = load_module_from_file("model", os.path.join(inf_dir, "model.py"))

    # the reference's globals, exactly as Transformer.__init__ would set them
    # (this script never builds the whole Transformer: the 155 GiB parameter
    # tree is loaded per layer instead)
    model.world_size = 1
    model.rank = 0
    model.default_dtype = torch.float8_e4m3fn if cfg.get("dtype", "bf16") == "fp8" else torch.bfloat16
    model.scale_fmt = "ue8m0" if cfg.get("scale_dtype", "fp8") == "fp8" else cfg.get("scale_fmt")
    model.scale_dtype = torch.float8_e8m0fnu if cfg.get("scale_dtype", "fp8") == "fp8" else torch.float32
    torch.set_default_dtype(torch.bfloat16)  # the kv buffers' / gemm outputs' dtype (generate.py)

    import dataclasses
    valid = {f.name for f in dataclasses.fields(model.ModelArgs)}
    margs = model.ModelArgs(**{k: v for k, v in cfg.items() if k in valid})
    margs.max_batch_size = 1
    margs.max_seq_len = max(args.max_seq_len or 0, 64 * 1024, seqlen)
    layers = parse_layer_range(args.layers, margs.n_layers)
    hc = margs.hc_mult

    # the shard map (the index's weight_map)
    with open(os.path.join(args.ckpt, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    shard_of = {}

    def shard_for(name):
        if name not in shard_of:
            shard_of[name] = os.path.join(args.ckpt, wmap[name])
        return shard_of[name]

    from safetensors import safe_open

    def assign(param, t):
        """The safetensors load_model's copy_ semantics: the checkpoint's value
        cast INTO the parameter's dtype (the fp4 weights' packed int8 bytes and
        the 1-byte fp8/e8m0 codes via the raw views)."""
        t = t.to(dev)
        if t.dtype == param.dtype:
            param.data.copy_(t)
        elif param.dtype == torch.float4_e2m1fn_x2 and t.dtype in (torch.int8, torch.uint8):
            param.data.view(torch.int8).copy_(t.contiguous().view(torch.uint8))
        elif param.dtype == torch.float8_e4m3fn and t.dtype == torch.uint8:
            param.data.copy_(t.contiguous().view(torch.float8_e4m3fn))
        elif param.dtype == torch.float8_e8m0fnu and t.dtype == torch.uint8:
            param.data.view(torch.uint8).copy_(t)
        elif param.dtype == torch.int32 and t.dtype == torch.int64:
            param.data.copy_(t.to(torch.int32))
        else:
            param.data.copy_(t.to(param.dtype))

    # the embedding (the walk's input)
    with torch.device(dev):
        embed = model.ParallelEmbedding(margs.vocab_size, margs.dim)
    with safe_open(shard_for("embed.weight"), framework="pt", device=str(dev)) as f:
        assign(embed.weight, f.get_tensor("embed.weight"))

    def assign_layer(f, i, block):
        """Load layers.i.* into block (the reference's load_model copy_ semantics,
        with one 0731-checkpoint fix: wo_a). The 0731 checkpoint stores
        attn.wo_a in FP8 (e4m3 + a per-128x128-tile e8m0 scale), but model.py's
        wo_a is a bf16 Linear — its `linear()` dispatch does a plain F.linear
        on the parameter, which must hold DEQUANTIZED values (exactly what
        inference/convert.py writes for the bf16 build: value * scale,
        flattened). The reference's load_model raw fp8->bf16 copy_ would leave
        the unscaled e4m3 codes in the parameter, inflating every attention
        output by ~1/scale (about 2e3 here) and blowing up the walk; so the
        scale is applied at load time. Everything else: plain copy_."""
        for name, param in block.state_dict().items():
            ckey = f"layers.{i}.{name}"
            if name == "attn.wo_a.weight":
                w = f.get_tensor(ckey).to(dev)
                if w.dtype == torch.float8_e4m3fn:
                    sc = f.get_tensor(f"layers.{i}.attn.wo_a.scale").to(dev)
                    scf = sc.contiguous().view(torch.uint8).to(torch.float32)
                    scf = torch.where(scf == 0, torch.zeros_like(scf),
                                     torch.exp2(scf - 127))
                    wd = (w.float().unflatten(0, (-1, 128)).unflatten(-1, (-1, 128))
                          * scf.view(w.shape[0] // 128, 1, -1, 1))
                    param.data.copy_(wd.flatten(2, 3).flatten(0, 1).to(param.dtype))
                    continue
            assign(param, f.get_tensor(ckey))
    input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
    with torch.inference_mode():
        h = embed(input_ids)
    h = h.unsqueeze(2).repeat(1, 1, hc, 1)  # the hc_mult copies (the walk's state)
    print(f"embed: {h.shape} (hc_mult {hc} x dim {margs.dim})", flush=True)

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    for i in layers:
        ti = time.time()
        with torch.inference_mode():  # generate.py's forward's context
            with torch.device(dev):
                block = model.Block(i, margs)
            with safe_open(shard_for(f"layers.{i}.attn_norm.weight"),
                           framework="pt", device=str(dev)) as f:
                assign_layer(f, i, block)
            h = block(h, 0, input_ids)
            # the dump: the state at the LAST prompt position, post layer i
            state = h[0, -1].float().contiguous().cpu()  # [hc, dim] f32
            if args.state_layout == "full":
                write_f32(f"{args.out}/layer_{i:02d}.f32", state)
            else:
                write_f32(f"{args.out}/layer_{i:02d}.f32", state.mean(dim=0))
                write_f32(f"{args.out}/layer_{i:02d}_hc.f32", state)
        del block
        gc.collect()
        print(f"[layer {i:02d}] done in {time.time() - ti:.1f}s", flush=True)

    logits_written = False
    if layers[-1] == margs.n_layers - 1:
        # the head's chain (the reference's forward's tail): the hc_head's
        # weighted collapse, the final norm, the lm head
        with torch.inference_mode():
            with safe_open(shard_for("head.weight"), framework="pt", device=str(dev)) as f:
                hc_head_fn = f.get_tensor("hc_head_fn").to(dev)
                hc_head_base = f.get_tensor("hc_head_base").to(dev)
                hc_head_scale = f.get_tensor("hc_head_scale").to(dev)
                norm_w = f.get_tensor("norm.weight").to(dev, torch.float32)
                head_w = f.get_tensor("head.weight").to(dev, torch.float32)
            shape, dtype = h.size(), h.dtype
            x = h.flatten(2).float()
            rsqrt = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + margs.norm_eps)
            mixes = F.linear(x, hc_head_fn) * rsqrt
            pre = torch.sigmoid(mixes * hc_head_scale + hc_head_base) + margs.hc_eps
            h = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2).to(dtype)
            with torch.device(dev):
                norm = model.RMSNorm(margs.dim, margs.norm_eps)
                head = model.ParallelHead(margs.vocab_size, margs.dim, margs.norm_eps, margs.hc_eps)
            norm.weight.data.copy_(norm_w)
            head.weight.data.copy_(head_w)
            logits = head(norm(h))  # [1, vocab] fp32 (the last position)
            topv, topi = logits[0].topk(args.topk)
            topv, topi = topv.cpu(), topi.cpu()
        with open(f"{args.out}/logits_top.txt", "w") as f:
            f.write(f"# top-{args.topk} logits at position {seqlen - 1} "
                    f"(post layer {layers[-1]} + hc_head + norm + head)\n")
            for rank in range(args.topk):
                f.write(f"{topi[rank].item()} {topv[rank].item():.6f}\n")
        logits_written = True
        print(f"logits top-{args.topk} at position {seqlen - 1} written", flush=True)

    with open(f"{args.out}/meta.txt", "w") as f:
        f.write(f"prompt_ids: {' '.join(map(str, ids))}\n")
        f.write(f"prompt_len: {seqlen}\n")
        f.write(f"chat_template: {'on (encode_messages, thinking_mode=chat)' if args.chat else 'off (raw ids / raw text)'}\n")
        f.write(f"hidden_size: {margs.dim}\n")
        f.write(f"hc_mult: {hc}\n")
        f.write(f"state_layout: {args.state_layout}\n")
        f.write(f"state_floats: {margs.dim * hc if args.state_layout == 'full' else margs.dim}\n")
        f.write(f"num_hidden_layers: {margs.n_layers}\n")
        f.write(f"layers_dumped: {layers[0]}-{layers[-1]}\n")
        f.write(f"dtype: bf16 (activations, projections), fp32 (gate/hc mixes, dump)\n")
        f.write(f"dump_dtype: float32 little-endian\n")
        f.write(f"kernel_mode: {kernel_mode}\n")
        f.write(f"hadamard_mode: {hadamard_mode}\n")
        f.write("wo_a: dequantized at load (the 0731 checkpoint stores attn.wo_a "
                "in FP8 + a per-128x128 e8m0 scale, but model.py's bf16 wo_a "
                "Linear expects dequantized values — the reference's raw "
                "fp8->bf16 load_model copy would drop the scale and inflate "
                "the attention output by ~1/scale)\n")
        f.write(f"logits_top: {'written' if logits_written else f'skipped (layer {margs.n_layers - 1} not in range)'}\n")
        f.write(f"torch: {torch.__version__}\n")
        f.write(f"seed: {args.seed}\n")

    peak = torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else float("nan")
    total = time.time() - t0
    print(f"DONE: {len(layers)} layers ({layers[0]}-{layers[-1]}) dumped to {args.out} "
          f"in {total:.1f}s, peak {dev.type} mem {peak:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
