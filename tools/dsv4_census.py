#!/usr/bin/env python3
"""Inventory a DeepSeek-V4 (-Flash) safetensors checkpoint: tensor census, memory plan, KV cost.

The DeepseekV4ForCausalLM path of the G0 facts package
(docs/deepseek_v4_flash_plan.md §1.9, §2; docs/checkpoint_budget_dsv4.md is its
output). Only safetensors JSON headers are read — 8-byte little-endian header
length, then the JSON — plus the file sizes and model.safetensors.index.json.
No tensor payload is read and nothing under the model directory is written.

What it does:

* scans every shard header and classifies every tensor into the plan's classes
  (mutually exclusive), reporting name pattern, dtype, shape, count and bytes;
* checks the quantized pairs against the config: fp8 e4m3 `.weight` with an
  e8m0 `.scale` on the `weight_block_size` grid (V4: 128x128, where V4.1 used
  32x32), and the MXFP4 expert pairs (I8 [N, K/2] + e8m0 [N, K/32]);
* reconciles the tensor bytes against the shard file sizes and the index;
* evaluates the plan's §2 placement (experts and the fp4/dense projections by
  slice, the indexers/compressors/routers/mHC replicated, the head by
  vocabulary) at the given worlds, and the global KV cost per context token
  from the reference's own cache formats (inference/model.py);
* models T=1 decode traffic per rank and its bandwidth floor.

Usage:
  python3 tools/dsv4_census.py [model_dir] [--world 1,2] [--bandwidth-gbps 215]
      [--json OUT] [--no-write]

Exit status: 0 when the census reconciles (every tensor classified, every
quantized pair well-formed, the byte sums agreeing with the files); 1 on any
mismatch, which is a G0 failure.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

DEFAULT_MODEL_DIR = Path("/data/models/DeepSeek-V4-Flash-0731")
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "docs" / "checkpoint_budget_dsv4.md"
DEFAULT_WORLDS = (1, 2)
# dsv4-native docs/STATUS.md D50: the measured copy peak on GB10 (78.8 % of spec).
DEFAULT_BANDWIDTH_GBPS = 215.0
# The DGX Spark budget rule: 130 GiB unified, ~14 GiB of CUDA context + page-cache
# headroom reserved (docs/deepseek_v41_flash_plan.md §2.1's method).
NODE_GIB = 130.0
RESERVE_GIB = 14.0

DT_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U64": 8,
    "BOOL": 1,
}
# Payload dtypes whose element is two packed nibbles (fp4 e2m1 in an I8 byte).
PACKED_2 = {"I8"}

LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
MTP_RE = re.compile(r"^mtp\.(\d+)\.(.+)$")
EXPERT_RE = re.compile(r"^ffn\.experts\.(\d+)\.(.+)$")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", nargs="?", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--world", default=",".join(str(w) for w in DEFAULT_WORLDS),
                   help="comma-separated world sizes to plan for")
    p.add_argument("--bandwidth-gbps", type=float, default=DEFAULT_BANDWIDTH_GBPS)
    p.add_argument("--json", type=Path, help="also write the raw census as JSON here")
    p.add_argument("--no-write", action="store_true", help="print only; do not write the report")
    return p.parse_args(argv)


def numel(shape: Iterable[int]) -> int:
    return math.prod(int(d) for d in shape) or 1


def read_header(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"short safetensors header length: {path}")
        size = struct.unpack("<Q", raw)[0]
        blob = stream.read(size)
        if len(blob) != size:
            raise ValueError(f"short safetensors header: {path}")
    header = json.loads(blob.decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def classify(name: str, n_layers: int, n_hash_layers: int) -> str:
    """A mutually exclusive storage/traffic class for one tensor name."""
    if name.startswith(("vision.", "visual.", "aligner.", "image_")):
        return "vision"        # the Vision-Exp sibling snapshot; never loaded (D9)
    if name in ("embed.weight",):
        return "embed"
    if name in ("head.weight",):
        return "lm_head"
    if name in ("norm.weight",) or name.endswith("_norm.weight") and not name.startswith(("layers.", "mtp.")):
        return "norm"
    if name.startswith("hc_head"):
        return "mhc"
    m = MTP_RE.match(name)
    if m:
        rest = m.group(2)
        e = EXPERT_RE.match(rest)
        if e:
            return "draft_expert"
        if rest.startswith("ffn.shared_experts."):
            return "draft_shared_expert"
        if rest.startswith(("attn.wq_b", "attn.wo_a", "attn.wo_b")):
            return "draft_attention_sharded"
        if rest.startswith(("attn.wq_a", "attn.wkv")):
            return "draft_attention_replicated"
        if rest.startswith("attn."):
            return "draft_norm"
        if rest.startswith("ffn.gate."):
            return "draft_router"
        if rest.startswith("ffn."):
            return "draft_norm"
        if rest.startswith("main_proj"):
            return "draft_proj"
        if rest.startswith("markov_head.markov_w2"):
            return "draft_head"
        if rest.startswith("markov_head.markov_w1"):
            return "draft_markov_embed"
        if rest.startswith("hc_"):
            return "draft_mhc"
        return "draft"
    m = LAYER_RE.match(name)
    if m:
        layer, rest = int(m.group(1)), m.group(2)
        e = EXPERT_RE.match(rest)
        if e:
            return "routed_expert"
        if rest.startswith("ffn.shared_experts."):
            return "shared_expert"
        if rest.startswith("ffn.gate.tid2eid"):
            return "hash_table"
        if rest.startswith("ffn.gate."):
            return "router"
        if rest.startswith("attn.indexer."):
            return "indexer"
        if rest.startswith("attn.compressor."):
            return "compressor"
        if rest.startswith(("attn.wq_b", "attn.wo_a", "attn.wo_b")):
            return "attention_sharded"
        if rest.startswith(("attn.wq_a", "attn.wkv")):
            return "attention_replicated"
        if rest.startswith("attn."):
            return "norm"          # attn_sink, q_norm, kv_norm
        if rest.startswith("hc_"):
            return "mhc"
        if rest.endswith("_norm.weight"):
            return "norm"          # attn_norm, ffn_norm
        raise ValueError(f"unclassified tensor: {name}")
    if name == "norm.weight":
        return "norm"
    raise ValueError(f"unclassified tensor: {name}")


# ---- placement (docs/deepseek_v4_flash_plan.md §2) -------------------------
# how a class's bytes scale with the world size
SHARDED = {
    "routed_expert": "slice",      # moe_intermediate / W (fp4 32-aligned, fp8 128-aligned)
    "draft_expert": "slice",
    "shared_expert": "slice",
    "draft_shared_expert": "slice",
    "attention_sharded": "slice",  # wq_b by head, wo_a by output group, wo_b column-packed
    "draft_attention_sharded": "slice",
    "lm_head": "slice",            # vocabulary rows
    "draft_head": "slice",         # the Markov head's vocab rows
}
# classes that are in the file and never loaded
NOT_LOADED = {"vision"}
# the classes that belong to a DSpark pass rather than to the T = 1 step
DRAFT_CLASSES = ("draft_expert", "draft_shared_expert", "draft_attention_sharded",
                 "draft_attention_replicated", "draft_router", "draft_norm", "draft_mhc",
                 "draft_proj", "draft_head", "draft_markov_embed", "draft")


def resident_bytes(total: int, cls: str, world: int) -> int:
    if cls in NOT_LOADED:
        return 0
    return total // world if SHARDED.get(cls) == "slice" else total


def kv_bytes_per_token(cfg: dict) -> dict:
    """The reference's cache formats (inference/model.py:Attention/Compressor/Indexer).

    A 512-wide latent row is 448 e4m3 (one e8m0 per 64) + 64 bf16 rope dims:
    448 + 7 + 128 = 583 bytes, stored in a 584-byte record. An index-key row is
    128 dims after the Hadamard rotation, e2m1 with an e8m0 per 32: 64 + 4 = 68.
    """
    head_dim = int(cfg["head_dim"])
    rd = int(cfg["qk_rope_head_dim"])
    nope = head_dim - rd
    row = nope + nope // 64 + rd * 2
    index_head_dim = int(cfg["index_head_dim"])
    index_row = index_head_dim // 2 + index_head_dim // 32
    ratios = [int(r) for r in cfg["compress_ratios"][: int(cfg["num_hidden_layers"])]]
    per_tok = 0.0
    for r in ratios:
        if r:
            per_tok += (row + (index_row if r == 4 else 0)) / r
    window_rows = sum(1 for _ in ratios) + int(cfg.get("n_mtp_layers", 0))
    return {
        "row": row,
        "index_row": index_row,
        "ratios": ratios,
        "global_per_token": int(round(per_tok)),
        "ring_per_slot": window_rows * 128 * row,
    }


def decode_traffic(classes: dict[str, int], cfg: dict, world: int) -> dict[str, float]:
    """T=1 weight bytes read per rank per token, per class (MB, decimal)."""
    n_experts = int(cfg["n_routed_experts"])
    topk = int(cfg["num_experts_per_tok"])
    inter = int(cfg["moe_intermediate_size"])
    dim = int(cfg["hidden_size"])
    mb = 1e6
    out: dict[str, float] = {}
    # the MoE classes: a row touches `topk` of the `n_experts` routed experts
    out["routed_expert"] = classes["routed_expert"] / mb * topk / n_experts / world
    out["shared_expert"] = classes["shared_expert"] / mb / world
    out["attention_sharded"] = classes["attention_sharded"] / mb / world
    out["attention_replicated"] = classes["attention_replicated"] / mb
    out["indexer"] = classes["indexer"] / mb
    out["compressor"] = classes["compressor"] / mb
    out["router"] = classes["router"] / mb
    out["mhc"] = classes["mhc"] / mb
    out["hash_table"] = topk * 8 * 3 / mb          # three gathered int64 rows
    out["norm"] = classes["norm"] / mb
    out["lm_head"] = classes["lm_head"] / mb / world
    out["embed"] = 0.0                             # one row
    out["vision"] = 0.0                            # never loaded
    return out


def draft_traffic(classes: dict[str, int], cfg: dict, world: int, rows: int) -> dict[str, float]:
    """One DSpark pass's extra weight bytes per rank (the block's `rows` draft rows
    through the draft stages, then the Markov chain), in MB."""
    n_experts = int(cfg["n_routed_experts"])
    topk = int(cfg["num_experts_per_tok"])
    mb = 1e6
    return {
        "draft_expert": classes["draft_expert"] / mb * topk / n_experts * rows / world,
        "draft_shared_expert": classes["draft_shared_expert"] / mb * rows / world,
        "draft_attention_sharded": classes["draft_attention_sharded"] / mb / world,
        "draft_attention_replicated": classes["draft_attention_replicated"] / mb,
        "draft_router": classes["draft_router"] / mb * rows,
        "draft_norm": classes["draft_norm"] / mb,
        "draft_mhc": classes["draft_mhc"] / mb,
        "draft_proj": classes["draft_proj"] / mb,
        "draft_head": classes["draft_head"] / mb * rows / world,
        "draft_markov_embed": 0.0,                 # one gathered row per draft row
        "draft": classes["draft"] / mb,
        "vision": 0.0,
    }


def fmt_gib(n: int) -> str:
    return f"{n / 2**30:.3f}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root: Path = args.model_dir.expanduser()
    worlds = [int(w) for w in str(args.world).split(",") if w]
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("text_config"):
        raise SystemExit("this census expects the FLAT V4 config (no text_config)")
    quant = config.get("quantization_config", {})
    block = [int(x) for x in quant.get("weight_block_size", [1, 1])]
    n_layers = int(config["num_hidden_layers"])
    n_hash = int(config.get("num_hash_layers", 0))

    shards = sorted(root.glob("model-*-of-*.safetensors")) or sorted(root.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors shards under {root}")

    tensors: dict[str, tuple[str, tuple[int, ...], str, tuple[int, int]]] = {}
    shard_bytes: dict[str, int] = {}
    for path in shards:
        header = read_header(path)
        shard_bytes[path.name] = path.stat().st_size
        for name, meta in header.items():
            if name in tensors:
                raise SystemExit(f"tensor {name} in two shards")
            off = meta["data_offsets"]
            tensors[name] = (meta["dtype"], tuple(meta["shape"]), path.name, (int(off[0]), int(off[1])))

    # ---- index.json reconciliation ----------------------------------------
    index_path = root / "model.safetensors.index.json"
    index_note = "absent"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        mapped = set(index.get("weight_map", {}))
        missing = sorted(mapped - set(tensors))
        extra = sorted(set(tensors) - mapped)
        index_note = (f"{len(mapped)} names in weight_map, {len(missing)} not in the shards, "
                      f"{len(extra)} not in the index")
        if missing or extra:
            print(f"INDEX MISMATCH: {index_note}", file=sys.stderr)
            return 1

    # ---- classify + pair checks -------------------------------------------
    by_class: dict[str, int] = defaultdict(int)
    count_class: dict[str, int] = Counter()
    by_dtype: dict[str, int] = defaultdict(int)
    count_dtype: dict[str, int] = Counter()
    detail: dict[tuple[str, str, tuple[int, ...]], list] = defaultdict(lambda: [0, 0, None])
    unmatched = 0
    fp8_pairs = mxfp4_pairs = bad_pairs = 0
    scales_orphan = 0
    params = 0
    n_draft = max((int(n.split(".")[1]) for n in tensors if n.startswith("mtp.")), default=-1) + 1
    for name, (dtype, shape, shard, (lo, hi)) in tensors.items():
        nbytes = hi - lo
        try:
            cls = classify(name, n_layers, n_hash)
        except ValueError:
            unmatched += 1
            cls = "unmatched"
        by_class[cls] += nbytes
        count_class[cls] += 1
        by_dtype[dtype] += nbytes
        count_dtype[dtype] += 1
        d = detail[(cls, dtype, shape)]
        d[0] += 1
        d[1] += nbytes
        if d[2] is None:
            pat = re.sub(r"(?<!\d)\d+(?!\d)(?=\.)", "N", name)
            pat = re.sub(r"(experts\.)\d+\.", r"\1N.", pat)
            pat = re.sub(r"^(layers|mtp)\.\d+\.", r"\1.N.", pat)
            d[2] = pat
        if not name.endswith(".scale") and dtype != "I64":
            # scale bytes and the routing lookup table are storage, not parameters
            params += numel(shape) * 2 if dtype in PACKED_2 else numel(shape)
        if name.endswith(".scale"):
            base = name[: -len(".scale")] + ".weight"
            meta = tensors.get(base)
            if meta is None:
                scales_orphan += 1
                continue
            bdt, bshape = meta[0], meta[1]
            if bdt == "F8_E4M3":
                ok = list(shape) == [max(1, bshape[0] // block[0]), max(1, bshape[1] // block[1])]
                fp8_pairs += 1
            elif bdt == "I8":
                ok = list(shape) == [bshape[0], (bshape[1] * 2) // 32]
                mxfp4_pairs += 1
            else:
                ok = False
            if not ok:
                bad_pairs += 1
                print(f"PAIR MISMATCH {name}: {dtype}{list(shape)} for {bdt}{list(bshape)}", file=sys.stderr)

    total_bytes = sum(hi - lo for _, _, _, (lo, hi) in tensors.values())
    file_bytes = sum(shard_bytes.values())
    if abs(total_bytes - file_bytes) > 0.01 * file_bytes:
        print(f"BYTE MISMATCH: tensors {total_bytes} vs files {file_bytes}", file=sys.stderr)
        return 1

    kv = kv_bytes_per_token(config)

    # ---- placement tables -------------------------------------------------
    resident = {w: {c: resident_bytes(b, c, w) for c, b in by_class.items()} for w in worlds}
    traffic = {w: decode_traffic(by_class, config, w) for w in worlds}
    block_rows = int(config.get("dspark_block_size", 0)) or 1
    dpass = {w: draft_traffic(by_class, config, w, block_rows) for w in worlds}

    # ---- active parameters per token (derived from the census) ----------------
    def tparams(name: str) -> int:
        meta = tensors.get(name)
        if meta is None:
            return 0
        dt, shape = meta[0], meta[1]
        if name.endswith(".scale") or dt in ("F8_E8M0", "I64"):
            return 0
        return numel(shape) * 2 if dt in PACKED_2 else numel(shape)

    topk = int(config["num_experts_per_tok"])
    active = tparams("embed.weight") + tparams("head.weight")
    for lay in range(n_layers):
        pre = f"layers.{lay}."
        active += sum(tparams(pre + sfx) for sfx in (
            "attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight",
            "attn.wo_a.weight", "attn.wo_b.weight", "ffn.gate.weight",
            "ffn.shared_experts.w1.weight", "ffn.shared_experts.w2.weight",
            "ffn.shared_experts.w3.weight", "hc_attn_fn", "hc_ffn_fn",
            "attn.compressor.wkv.weight", "attn.compressor.wgate.weight",
            "attn.indexer.wq_b.weight", "attn.indexer.weights_proj.weight",
            "attn.indexer.compressor.wkv.weight", "attn.indexer.compressor.wgate.weight"))
        active += topk * sum(tparams(f"{pre}ffn.experts.0.w{w}.weight") for w in (1, 2, 3))

    # ---- config vs checkpoint (the G0 checks) ---------------------------------
    layers_seen = sorted({int(n.split(".")[1]) for n in tensors if n.startswith("layers.")})
    idx_layers = sorted({int(n.split(".")[1]) for n in tensors
                         if n.startswith("layers.") and ".attn.indexer." in n})
    cmp_layers = sorted({int(n.split(".")[1]) for n in tensors if n.startswith("layers.")
                         and len(n.split(".")) > 3 and n.split(".")[3] == "compressor"})
    hash_layers = sorted({int(n.split(".")[1]) for n in tensors if n.endswith("gate.tid2eid")})
    bias_layers = sorted({int(n.split(".")[1]) for n in tensors
                              if n.startswith("layers.") and n.endswith("ffn.gate.bias")})
    ratios = [int(r) for r in config["compress_ratios"]]
    ratio_idx = [l for l, r in enumerate(ratios[:n_layers]) if r == 4]
    ratio_any = [l for l, r in enumerate(ratios[:n_layers]) if r]
    experts_per_layer = sorted({len({n.split(".")[4] for n in tensors
                                     if n.startswith(f"layers.{lay}.ffn.experts.")})
                                for lay in layers_seen})
    checks: list[tuple[str, str]] = [
        ("x" if layers_seen == list(range(n_layers)) else "!",
         f"`num_hidden_layers` {n_layers} vs {len(layers_seen)} `layers.N` blocks in the shards"),
        ("x" if int(config.get("num_nextn_predict_layers", -1)) == n_draft else "!",
         f"{n_draft} `mtp.N` draft blocks in the shards (`mtp.0..{n_draft - 1}`) and "
         f"`num_nextn_predict_layers` says {config.get('num_nextn_predict_layers')}"
         + ("" if int(config.get("num_nextn_predict_layers", -1)) == n_draft else
            ": the inventory wins (the reference's `inference/config.json` has `n_mtp_layers: 3`, and "
            "only `mtp.0` carries `main_proj` while only `mtp.2` carries the head-side tensors) - the "
            "config parser and the binding must take the draft count from the shards")),
        ("x" if len(ratios) == n_layers + n_draft else "!",
         f"`compress_ratios` has {len(ratios)} entries = {n_layers} backbone + {n_draft} draft"),
        ("x" if idx_layers == ratio_idx else "!",
         f"the indexer exists on exactly the {len(ratio_idx)} ratio-4 layers "
         f"({idx_layers[0]}..{idx_layers[-1]}), on no ratio-128 and no window-only layer"),
        ("x" if cmp_layers == ratio_any else "!",
         f"the main-KV compressor exists on exactly the {len(ratio_any)} layers with ratio > 0"),
        ("x" if hash_layers == list(range(n_hash)) and bias_layers == [l for l in range(n_layers) if l >= n_hash] else "!",
         f"`num_hash_layers` {n_hash}: `gate.tid2eid` on layers {hash_layers}, `gate.bias` on the other "
         f"{len(bias_layers)} (a hash-routed layer carries NO bias)"),
        ("x" if experts_per_layer == [int(config["n_routed_experts"])] else "!",
         f"every layer carries {experts_per_layer[0]} distinct experts"),
        ("x" if bad_pairs == 0 and scales_orphan == 0 else "!",
         f"every fp8 `.weight` carries its e8m0 `.scale` on the {block[0]}x{block[1]} grid and every "
         f"MXFP4 pair its e8m0 per 32 ({fp8_pairs} fp8, {mxfp4_pairs} fp4 matrices)"),
        ("x" if not [n for n in tensors if re.search(r"engram|ngram|prime|conv1d|visual|bias_vl", n, re.I)]
         else "!", "no Engram / n-gram / prime / vision / bias_vl tensor anywhere in the 48 shards"),
        ("x" if tparams("embed.weight") == int(config["vocab_size"]) * int(config["hidden_size"]) else "!",
         f"`embed.weight` [{int(config['vocab_size'])}, {int(config['hidden_size'])}], untied "
         "(`head.weight` is a separate tensor, `tie_word_embeddings` false)"),
    ]

    # ---- report -----------------------------------------------------------
    GiB = 2**30
    lines: list[str] = []
    A = lines.append
    A("# DeepSeek-V4-Flash Checkpoint Budget Report")
    A("")
    A(f"- Model directory: `{root}`")
    A(f"- Architecture: `{config['architectures'][0]}` / `model_type {config['model_type']}` (flat config, no `text_config`)")
    A(f"- Files: {len(shards)} shards; tensors: {len(tensors)}; parameters: {params / 1e9:.1f} B "
      "(fp4 payloads counted as two nibbles per byte)")
    A(f"- Total weights: **{total_bytes / 1e9:.2f} GB ({total_bytes / GiB:.3f} GiB)**; "
      f"file bytes {file_bytes / 1e9:.2f} GB (delta {100 * (file_bytes - total_bytes) / file_bytes:.4f} %, headers only)")
    A(f"- Index: {index_note}")
    A("- Generated by `tools/dsv4_census.py`; no tensor payloads were read.")
    A(f"- Layers: {n_layers} backbone + {n_draft} DSpark draft stages (`mtp.0..{n_draft - 1}`) while "
      f"`num_nextn_predict_layers` says {config.get('num_nextn_predict_layers')} - the inventory wins "
      "(the reference's `inference/config.json` has `n_mtp_layers: 3`); see the G0 checks below.")
    A(f"- Experts {int(config['n_routed_experts'])} top-{int(config['num_experts_per_tok'])} "
      f"(intermediate {int(config['moe_intermediate_size'])}, `scoring_func` {config['scoring_func']}, "
      f"`topk_method` {config['topk_method']}, `routed_scaling_factor` {config['routed_scaling_factor']}, "
      f"`swiglu_limit` {config['swiglu_limit']}); hidden {int(config['hidden_size'])}; "
      f"{int(config['num_attention_heads'])} heads x head_dim {int(config['head_dim'])}; "
      f"mHC `hc_mult` {int(config['hc_mult'])}; hash-routed layers: the first {n_hash}.")
    A(f"- Quantized pairs: {fp8_pairs} fp8 matrices (e4m3 + e8m0 on the {block[0]}x{block[1]} grid), "
      f"{mxfp4_pairs} MXFP4 matrices (I8 codes + e8m0 per 32); "
      f"{bad_pairs} mismatched, {scales_orphan} orphan scales.")
    A(f"- Parameters: {params / 1e9:.1f} B total (scale and lookup bytes excluded); "
      f"**{active / 1e9:.2f} B active per token** in the backbone walk "
      f"(= {2 * active / 1e9:.0f} GFLOP per prefill token, derived from the census).")
    A("")
    A("## Config vs checkpoint (the G0 checks)")
    A("")
    for st, text in checks:
        A(f"- [{st}] {text}")
    A("")
    A("## Storage inventory")
    A("")
    A("| class | tensors | GB | share |")
    A("|---|---:|---:|---:|")
    for cls, nbytes in sorted(by_class.items(), key=lambda kv: -kv[1]):
        A(f"| {cls} | {count_class[cls]:,} | {nbytes / 1e9:.3f} | {100 * nbytes / total_bytes:.2f}% |")
    A(f"| **total** | **{len(tensors):,}** | **{total_bytes / 1e9:.3f}** | 100.00% |")
    A("")
    A("## Tensor census (name pattern / dtype / shape / count / bytes)")
    A("")
    A("| class | name pattern | dtype | shape | count | bytes | MiB |")
    A("|---|---|---|---|---:|---:|---:|")
    for (cls, dtype, shape), (cnt, nbytes, pat) in sorted(detail.items(), key=lambda kv: (kv[0][0], -kv[1][1])):
        A(f"| {cls} | `{pat}` | {dtype} | {list(shape)} | {cnt:,} | {nbytes:,} | {nbytes / 2**20:.3f} |")
    A("")
    A("## Storage by dtype")
    A("")
    A("| dtype | tensors | GB |")
    A("|---|---:|---:|")
    for dt, n in count_dtype.most_common():
        A(f"| {dt} | {n:,} | {by_dtype[dt] / 1e9:.3f} |")
    A("")
    A("## Shards")
    A("")
    A(f"{len(shards)} shards, {min(shard_bytes.values()) / 1e9:.2f} to {max(shard_bytes.values()) / 1e9:.2f} GB each.")
    A("")
    A("| shard | tensors | GB | classes (bytes) |")
    A("|---|---:|---:|---|")
    per_shard: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_shard_n: dict[str, int] = Counter()
    for name, (dtype, shape, shard, (lo, hi)) in tensors.items():
        per_shard[shard][classify(name, n_layers, n_hash)] += hi - lo
        per_shard_n[shard] += 1
    for shard in shards:
        f = shard.name
        cs = ", ".join(f"{c} {b / 2**30:.2f} GiB" for c, b in
                       sorted(per_shard[f].items(), key=lambda kv: -kv[1]))
        A(f"| `{f}` | {per_shard_n[f]:,} | {shard_bytes[f] / 1e9:.3f} | {cs} |")
    A("")
    A("## Resident bytes per rank")
    A("")
    A("Placement follows docs/deepseek_v4_flash_plan.md §2: the routed, draft and shared experts "
      "are sliced on `moe_intermediate_size` (every slice starts on a 32-element fp4 scale block and "
      "a 128-column fp8 block), `wq_b` by attention head, `wo_a` by output group, `wo_b` column-packed, "
      "`head.weight` and the DSpark Markov `markov_w2` by vocabulary rows; `wq_a`, `wkv`, the indexers, "
      "the compressors, the routers, the hash-routed `tid2eid` tables, the mHC coefficients, the norms, "
      "`embed.weight`, `main_proj` and `markov_w1` are replicated. There is no Engram table in this "
      "checkpoint (§1.7 of the plan), so nothing is mmap'ed from the NVMe: every weight is resident.")
    A("")
    header = "| class | " + " | ".join(f"W={w} GiB" for w in worlds) + " |"
    A(header)
    A("|---|" + "---:|" * len(worlds))
    order = [c for c, _ in sorted(by_class.items(), key=lambda kv: -kv[1])]
    for d in traffic.values():
        d.setdefault("vision", 0.0)
    for d in dpass.values():
        d.setdefault("vision", 0.0)
    for cls in order:
        A(f"| {cls} | " + " | ".join(f"{resident[w][cls] / GiB:.3f}" for w in worlds) + " |")
    A(f"| **weights** | " + " | ".join(f"**{sum(resident[w].values()) / GiB:.2f}**" for w in worlds) + " |")
    A("")
    for w in worlds:
        tot = sum(resident[w].values()) / GiB
        fit = tot + RESERVE_GIB <= NODE_GIB
        A(f"- W={w}: **{tot:.2f} GiB of weights** + {RESERVE_GIB:.0f} GiB reserve "
          f"(CUDA context ~14 GiB, arenas/staging) = {tot + RESERVE_GIB:.2f} GiB against the "
          f"{NODE_GIB:.0f} GiB node -> **{'FITS' if fit else 'DOES NOT FIT'}** "
          f"({NODE_GIB - tot - RESERVE_GIB:+.2f} GiB left for the KV cache, the rings, the prefix arena "
          "and the graphs).")
    A("")
    A("## KV cache cost")
    A("")
    A(f"- A latent row (window and compressed entries alike): {kv['row']} bytes "
      "— `head_dim - rope_head_dim` e4m3 values with one e8m0 per 64, plus the 64 rope dims kept bf16 "
      "(`model.py: act_quant(kv[..., :-rd], 64, ...)`); stored in a 584-byte record.")
    A(f"- An index-key row: {kv['index_row']} bytes — 128 dims after the Hadamard rotation, e2m1 with an "
      "e8m0 per 32 (`fp4_act_quant(kv, 32, True)`).")
    A(f"- Global (paged) cache: **{kv['global_per_token']} bytes per context token per rank** — every "
      "compressed layer keeps its OWN cache "
      f"({kv['ratios'].count(4)} layers at ratio 4 = {(kv['row'] + kv['index_row']) / 4:.0f} B/token each, "
      f"{kv['ratios'].count(128)} layers at ratio 128 = {kv['row'] / 128:.1f} B/token each).")
    A(f"- Per request slot: {kv['ring_per_slot'] / 2**20:.2f} MiB of 128-row window rings "
      f"({kv['ring_per_slot'] // 128 // kv['row']} layers x 128 rows) plus the compressor tail state "
      f"({kv['ratios'].count(4)} x 2 x 8 x 512 fp32 = "
      f"{kv['ratios'].count(4) * 2 * 8 * 512 * 4 / 1024:.0f} KiB at ratio 4, "
      f"{kv['ratios'].count(128) * 2 * 128 * 512 * 4 / 2**20:.1f} MiB at ratio 128).")
    A(f"- At 1,048,576 positions: {kv['global_per_token'] * 1048576 / 2**30:.2f} GiB per rank "
      f"(128 KiB of positions: {kv['global_per_token'] * 131072 / 2**20:.2f} MiB).")
    A("")
    A("## Decode traffic model")
    A("")
    A("### The T = 1 step (per rank, per token)")
    A("")
    A("The resident set minus the experts a token does not route to "
      f"({int(config['num_experts_per_tok'])} of {int(config['n_routed_experts'])} per layer), minus the "
      "embedding (one row), with the DSpark classes excluded (they are a pass cost, below). A DSpark "
      "verify of six rows reads the SAME bytes: the class is read once per step whatever the row count.")
    A("")
    A("| class | " + " | ".join(f"W={w} MB/step" for w in worlds) + " |")
    A("|---|" + "---:|" * len(worlds))
    step_order = [c for c in order if c not in DRAFT_CLASSES]
    for cls in step_order:
        A(f"| {cls} | " + " | ".join(f"{traffic[w][cls]:,.1f}" for w in worlds) + " |")
    A("| **step total** | " + " | ".join(f"**{sum(traffic[w].values()):,.1f}**" for w in worlds) + " |")
    A("")
    for w in worlds:
        t = sum(traffic[w].values())
        rep = sum(v for c, v in traffic[w].items() if c not in SHARDED)
        A(f"- W={w}: **{t / 1000:.2f} GB/step = {t / args.bandwidth_gbps:.1f} ms at "
          f"{args.bandwidth_gbps:g} GB/s** (dsv4-native docs/STATUS.md D50's measured copy peak on GB10); "
          f"replicated share {100 * rep / t:.0f} %.")
    A("")
    A(f"### The DSpark pass (per rank, {block_rows} draft rows through {n_draft} stages + the Markov chain)")
    A("")
    A("| class | " + " | ".join(f"W={w} MB/pass" for w in worlds) + " |")
    A("|---|" + "---:|" * len(worlds))
    for cls in [c for c in order if c in DRAFT_CLASSES]:
        A(f"| {cls} | " + " | ".join(f"{dpass[w][cls]:,.1f}" for w in worlds) + " |")
    A("| **pass total (draft only)** | " + " | ".join(f"**{sum(dpass[w].values()):,.1f}**" for w in worlds) + " |")
    A("| **step + pass** | " + " | ".join(
        f"**{sum(traffic[w].values()) + sum(dpass[w].values()):,.1f}**" for w in worlds) + " |")
    A("")
    for w in worlds:
        st, dp = sum(traffic[w].values()), sum(dpass[w].values())
        ms = (st + dp) / args.bandwidth_gbps
        A(f"- W={w}: a pass moves {(st + dp) / 1000:.2f} GB = {ms:.1f} ms of weight traffic; at "
          f"{block_rows + 1}-row verification and the block's acceptance length the floor per accepted "
          "token is that divided by the tokens the pass commits.")
    A("")
    A("## Reconciliation and exclusions")
    A("")
    A(f"- Unmatched tensors: **{unmatched}**.")
    A(f"- Tensor bytes vs file bytes: {total_bytes:,} vs {file_bytes:,} "
      f"({100 * (file_bytes - total_bytes) / file_bytes:.4f} % of JSON headers).")
    A(f"- Quantized pairs: {fp8_pairs + mxfp4_pairs:,} checked, {bad_pairs} wrong, {scales_orphan} orphan scales.")
    A("- No Engram / n-gram tables, no vision tower, no `bias_vl`: the classes those would need are absent "
      "from this checkpoint (the plan's §1.7 states what replaces the Engram).")
    A("")

    if not args.no_write:
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"... {len(lines)} report lines")
    if args.json:
        args.json.write_text(json.dumps({
            "tensors": {k: [v[0], list(v[1]), v[2], list(v[3])] for k, v in tensors.items()},
            "by_class": by_class, "count_class": count_class,
            "by_dtype": by_dtype, "count_dtype": count_dtype,
            "detail": {f"{c}|{d}|{s}": [v[0], v[1], v[2]] for (c, d, s), v in detail.items()},
            "params": params, "total_bytes": total_bytes, "file_bytes": file_bytes,
            "kv": kv, "resident": {str(w): resident[w] for w in worlds},
            "traffic": {str(w): traffic[w] for w in worlds},
        }, separators=(",", ":")) + "\n", encoding="utf-8")

    ok = unmatched == 0 and bad_pairs == 0 and scales_orphan == 0
    print(f"SUMMARY tensors={len(tensors)} params={params / 1e9:.1f}B "
          f"total={total_bytes / GiB:.3f}GiB files={file_bytes / GiB:.3f}GiB "
          f"classes={len(by_class)} unmatched={unmatched} bad_pairs={bad_pairs} orphan_scales={scales_orphan} "
          f"kv_per_token={kv['global_per_token']} "
          + " ".join(f"resident_w{w}={sum(resident[w].values()) / GiB:.2f}GiB" for w in worlds))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
