# DeepSeek-V4-Flash (0731) attention numerics spec

The numeric oracle for the dsv4 csa2 CUDA wiring: what the six decode
attention partials hold, the 64-head indexer selection, the ratio-4 (C4A)
overlapping-compressor tail update, and the DSpark union attention over
the three KV sources. Every numeric claim carries a source reference
(`file:line` in the pinned Python reference, the port spec, or the C++
tree). Where the Python reference and the C++ oracle agree, the doc says
so; where they are ambiguous or disagree, §5 (GAPS) flags it.

## 0. Scope, sources, and the V4-0731 geometry

### 0.1 Sources and their roles

| source | role | path |
|---|---|---|
| **checkpoint reference** (V4.1-style, the one the GPU parity gate uses) | the parity oracle for the C4A compressor, the attention, and the DSpark attention | `/data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local/inference/{model,kernel}.py` (no `engram.py` — V4-0731 has no engram, the port spec §2.3) |
| **dsv4-flash-native** (the newer rewrite, with the engram + the two-level candidate selection) | a second reference; its `ANALYSIS.md` is a detailed analysis of it. Deltas vs the checkpoint reference are noted where they matter | `~/work/dsv4-flash-native/inference/{kernel,model,engram}.py` |
| **the port spec** | the V4-0731 geometry (46-entry `compress_ratios`, C4A/C128A mapping, the 64-index-head fold, the compressor shapes) | `docs/dsv4_kernel_port_spec.md` |
| **the C++ tree** | what is already pinned: `src/models/dsv4/{csa2_layer,compress,dspark_layer,model}.{cu,hpp}`, the CPU oracles `tests/unit/{dsv4_csa2_oracle,dsv4_compress_tail,dsv4_dspark_oracle}_test.cpp`. This doc pins the DELTA the spec adds, not a re-statement | this repo |

The two Python references are DIFFERENT code (not just different
defaults): the checkpoint reference has the overlapping ratio-4
compressor (the `coff 2` planes, the `ape`, the 8-slot decode state,
the indexer's own 128-wide rotated compressor, the per-64 fp8 quant of
the non-RoPE 448 dims, the ad-hoc q re-normalization, the single
`kv_cache` buffer window+compressed), while dsv4-flash-native has the
plain (non-overlapping) compressor, the shared-latent indexer, the
fp4 group-16 compressed KV, the two-level candidate selection, and the
separate window/compressed caches. **The checkpoint reference is the
parity oracle; where it and flash-native disagree, this doc follows the
checkpoint reference and flags the delta in §5.**

Line references below use these shorthands:
- `ck:model.py:N` / `ck:kernel.py:N` — the checkpoint reference
  (`/data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local/inference/`).
- `fn:model.py:N` / `fn:kernel.py:N` — dsv4-flash-native
  (`~/work/dsv4-flash-native/inference/`).
- `spec:§x` — `docs/dsv4_kernel_port_spec.md`.
- `src/...` / `tests/unit/...` — this repo.

### 0.2 The V4-0731 geometry (the checkpoint's flat `config.json`)

Every value below is the checkpoint's own `inference/config.json`
(flat layout), cross-checked against the HF-layout config the port spec
censused (`spec:§1`) and the C++ test fixture
(`tests/unit/dsv4_config_json.hpp:12-69`):

| key | value | evidence |
|---|---|---|
| hidden (`dim`) | **4096** | ck config.json `"dim": 4096`; `spec:§1` |
| layers | **43 backbone** + 3 ratio-tail slots (43,44,45); the checkpoint ships **one** MTP stage (`mtp.0`) | ck config.json `"n_layers": 43`, `"n_mtp_layers": 3`; HF config `num_nextn_predict_layers 1` + `mtp.0.*` only (`spec:§1`); the C++ config takes `num_nextn_predict_layers ∈ [0, num_draft_stages()]` (`src/models/dsv4/config.cpp:299-302`) — the 3 tail slots are ratio-0, one shipped stage |
| `compress_ratios` | **46 entries**: `[0,0,(4,128)×20,4,0,0,0]` — C4A (ratio 4, the indexer) = even layers 2..42 (21), C128A (ratio 128) = odd layers 3..41 (20), SWA-only 0,1, tail 43..45 = 0,0,0 | ck config.json `"compress_ratios"`; `spec:§1` (the `(4,128)×21` shorthand counts the final lone 4); `tests/unit/dsv4_config_json.hpp:63` |
| head_dim (`kCsa2Latent`) | **512** (448 NoPE + 64 RoPE) | ck config.json `"head_dim": 512`, `"rope_head_dim": 64`; `src/kernels/csa2.hpp:33-35` |
| index | **64 heads × 128 dim, topk 512** | ck config.json `"index_n_heads": 64`, `"index_head_dim": 128`, `"index_topk": 512`; `spec:§1` (`indexer.wq_b [8192, 1024]`) |
| q_lora / o_lora / o_groups / n_heads | 1024 / 1024 / 8 / 64 | ck config.json; `src/models/dsv4/csa2_layer.hpp:57-64` (`Dsv4Csa2Config`) |
| window | **128** (the fp8 ring) | ck config.json `"window_size": 128` |
| dspark | targets **[40,41,42]**, block **5**, markov **256**, noise **128799** | ck config.json `"dspark_target_layer_ids"`, `"dspark_block_size"`, `"dspark_markov_rank"`, `"dspark_noise_token_id"`; `spec:§1` |
| rope | YaRN factor **16**, β_fast 32, β_slow 1, θ **10000**, original **65536**; compressed θ **160000** | ck config.json `"rope_factor"`, `"rope_theta"`, `"original_seq_len"`, `"compress_rope_theta"`; `spec:§1` |
| compressor (C4A) | `wkv` / `wgate` **BF16 [1024, 4096]** + `ape` **F32 [4, 1024]** | `spec:§1` tensor census; the C++ binding `src/models/dsv4/binding.cpp:48-57` (`coff * hd` = 2 × 512 = 1024) |
| compressor (C128A) | `wkv` / `wgate` **BF16 [512, 4096]** + `ape` **F32 [128, 512]** | `spec:§1`; `src/models/dsv4/binding.cpp:48-57` (`coff` = 1) |
| indexer compressor (C4A only) | `wkv` / `wgate` **BF16 [256, 4096]** + `ape` **F32 [4, 256]** + `norm` [128] | `spec:§1`; `src/models/dsv4/binding.cpp:62-77` (`2 * id` = 2 × 128 = 256, `coff 2`) |
| fp8 dense grid | **128×128**, e8m0 scales, `dynamic` activation scheme | ck config.json `"quantization_config"`; `spec:§1,§3` |

The three-class dispatch (`src/models/dsv4/config.hpp:131-135`):
ratio 0 = SWA-only (layers 0,1, 43..45), ratio 4 = C4A (the indexer,
`coff 2`), ratio 128 = C128A (`coff 1`, no indexer — the plain
selection over every visible entry, §2.4).

### 0.3 The geometry question, settled: the C4A tail width W is 1024

The dsv4 csa2 comment says the per-request tail is "fp32 [2, 512]"
(`src/models/dsv4/csa2_layer.cu:411-414`), and the dsv41 base strides
its tail as `2 * kCsa2Latent` = 2 × 512 = 1024 floats
(`src/kernels/csa2.cu:299` `tails + int64_t(req_ids[start]) * 2 *
kCsa2Latent`, where dsv41's `kCsa2Latent` IS its compressor output
width, 512). The port spec lists the C4A `compressor.{wkv,wgate}` as
`[1024, 4096]` (`spec:§1`), i.e. a **1024-wide projection**. The
disagreement was whether `tails_w_` in `src/models/dsv4/compress.cu`
is 512 or 1024.

**Resolution: W = 1024 = coff × kCsa2Latent = 2 × 512. `tails_w_`
must be 1024.**

Proof chain:

1. **The tail holds one pending token's (kv, score) pair, and kv/score
   ARE the `wkv`/`wgate` GEMM outputs.** The checkpoint's Compressor
   computes `kv = self.wkv(x)`, `score = self.wgate(x)` (ck:model.py:329-330)
   and stores those vectors verbatim in the state buffers (ck:model.py:353-354
   `self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)`).
   The C++ tail kernel does the same: `tail[c] = kvt[c]` (the kv plane)
   and `tail[W + c] = st[c]` (the score plane) from the `wkv`/`wgate`
   GEMM outputs (`src/models/dsv4/compress.cu:104-108`, the contract at
   `src/models/dsv4/compress.hpp:24-31`: "`comp_kv` / `comp_score` are
   the wkv / wgate's GEMM outputs (fp32 [tokens, W] each)").
2. **The C4A `wkv`/`wgate` output width is 1024, not 512.** The
   checkpoint's Compressor sizes them `coff * head_dim` with
   `coff = 1 + (compress_ratio == 4)` = 2 and `head_dim` = 512
   (ck:model.py:296-304: `self.overlap = compress_ratio == 4`,
   `coff = 1 + self.overlap`,
   `self.wkv = Linear(self.dim, coff * self.head_dim, ...)` —
   `Linear(4096, 1024)`). The checkpoint ships them as BF16
   `[1024, 4096]` (`spec:§1` tensor census; the C++ binding
   `src/models/dsv4/binding.cpp:51-56` `add_bf16(cp + "wkv.weight",
   {coff * hd, H})`).
3. **Therefore each tail plane is 1024 floats wide.** The [2, W] tail
   is `[kv plane (1024) | score plane (1024)]` = 2048 floats per
   request, and the state row width in the checkpoint is the same 1024
   (ck:model.py:309-310: `kv_state` / `score_state` shaped
   `(max_batch_size, coff * compress_ratio, coff * self.head_dim)` =
   `(b, 8, 1024)`). The C4A CPU oracle pins exactly this: `CmpCfg`
   `state_row() = 2 * coff() * 512` = 2048 and the test runs with
   `W = 1024` (`tests/unit/dsv4_csa2_oracle_test.cpp:191`
   `int state_row() const { return 2 * coff() * 512; }`, and
   `DGPP_TEST(dsv4_csa2_compressor_state_store)` `const int ratio = 4,
   W = 1024, bs = 4, K = 512` at line 432).
4. **How 1024 relates to 2×512 and to the pair.** Inside EACH 1024-
   wide plane (kv and score), the first 512 dims are the **overlap
   plane** and the last 512 dims are the **normal plane**:
   "When overlap, the first half of dims is for overlapping compression,
   second half for normal" (ck:model.py:302). At the group boundary the
   pool takes the previous group's overlap plane (dims 0..511 of its
   state rows) and the current group's normal plane (dims 512..1023):
   `torch.cat([kv_state[:, :ratio, :d], kv_state[:, ratio:, d:]],
   dim=1)` with `d = head_dim = 512` (ck:model.py:356-357) — an 8-entry
   × 512-dim pool producing a 512-dim latent. The prefill's
   `overlap_transform` states the same split: rows `[ratio:]` of the
   pooled form take `tensor[..., d:]` (the normal plane), rows `[1:,
   :ratio]` take the PREVIOUS group's `tensor[..., :d]` (the overlap
   plane) (ck:model.py:313-320). So "1024 = 2×512" is the
   **per-plane projection width** (overlap half + normal half), and the
   **[2, W] "pair" is (kv plane, score plane)** — the pending token's
   value and its gate, NOT the two 512-dim halves. The two coders'
   "512" was the dsv41 value: dsv41's tail stride `2 * kCsa2Latent`
   (`src/kernels/csa2.cu:299`) is 2 planes × 512 because dsv41's
   compressor output width IS 512 (its ratio-2 `wkv`/`wgate` are
   `[512, 5120]`, `src/models/dsv41/loader.cpp:336-338`).
5. **The stale 512s.** `src/models/dsv4/model.hpp:269-275`
   ("`tails_w_ = 512`, the dsv41's `kCsa2Latent`'s") and the
   `csa2_layer.cu` "fp32 [2, 512]" comment (`src/models/dsv4/csa2_layer.cu:389`)
   are dsv41 re-expression artifacts — wrong for C4A. The `W` the
   kernel takes as a runtime argument (`src/models/dsv4/compress.cu:142-156`)
   must be **1024** for the ratio-4 layers; the kernel's own guard
   (`W % kThreads == 0 && W <= 8 * kThreads`, `src/models/dsv4/compress.cu:148-150`)
   accepts 1024 (4 × 256, `kPer` = 4 registers/thread — the `pooled[8]`
   array at `src/models/dsv4/compress.cu:54` covers it). The tail test
   runs its synthetic shape at `W = 512`
   (`tests/unit/dsv4_compress_tail_test.cpp:201`) — a valid smaller
   instantiation of the same `[2, W]` contract, not the C4A width.

Consequence for the wiring (no code change in this doc, recorded for
the coder): `tails_w_` → 1024 (the allocation `per_req = tails_ * 2 *
tails_w_` at `src/models/dsv4/model.cpp:225` then sizes 2048 floats ×
`tails_` per request), and the C4A `wkv` GEMM in
`src/models/dsv4/csa2_layer.cu:391-392` must produce **1024** output
rows (it currently passes `kCsa2Latent` = 512 as the GEMM `n` — see
GAP G3).

## 1. The six decode attention partials (m / l / c × main / window)

The csa2 decode attention is a **two-source** attention: a sliding-window
source over the layer's own 128-slot ring (the fp8 ring format) and —
for a kv source (`ratio > 0`) — a main source over the selected
compressed entries (the planar main cache, the fp4 main format). Each
source is computed as a split-KV partial (the flash-decoding shape), and
the six partials are merged in one finish. This is the V4 re-expression
of the checkpoint reference's single `sparse_attn` over the
concatenated `[window | compressed]` KV (ck:model.py:520, 538) — the
split-partial + finish merge is algebraically the SAME single softmax,
within the fp32 accumulation budget.

### 1.1 What each partial holds

Each source is split across the key dimension into `n_split` chunks
(the flash-decoding "split-KV"), one partial per (row, split). The six
partials live in the layer scratch, indexed `[r * n_split + s]`
(**the (row, split) PRODUCT**, not the max — the 2026-09-18 GPU-window
OOB regression the `dsv4_csa2_partial_layout` oracle guards):

| partial | type / shape | holds |
|---|---|---|
| `m_main` | fp32 `[ws_slots, lh]` | the running MAX of the main source's per-split logits (per head) |
| `l_main` | fp32 `[ws_slots, lh]` | the running SUM (normalizer) of `exp(logit - m)` for the main source |
| `c_main` | fp32 `[ws_slots, lh, 512]` | the accumulated `Σ exp(logit - m) · kv` value vector for the main source |
| `m_win` | fp32 `[ws_slots, lh]` | the running MAX of the window source's per-split logits |
| `l_win` | fp32 `[ws_slots, lh]` | the running SUM (normalizer) for the window source |
| `c_win` | fp32 `[ws_slots, lh, 512]` | the accumulated value vector for the window source |

- `lh` = `local_heads()` = `num_heads / tp` = 64 / 2 = **32**
  (`src/models/dsv4/csa2_layer.hpp:57-64,70`).
- `512` = `kCsa2Latent` (the latent width, K == V).
- `ws_slots` = `max(max_decode_rows, 8) * max(decode_n_split, 8)`
  (`src/models/dsv4/csa2_layer.cu:104`), so the `[r * n_split + s]`
  index fits: `r < max_decode_rows` and `s < n_split <= decode_n_split`,
  hence `r * n_split < ws_slots`.
- `n_split_win` = `min(decode_n_split_, kWinDecodeSplit = 8)`
  (`src/models/dsv4/csa2_layer.cu:308`, `kWinDecodeSplit` at
  `csa2_layer.hpp:156`) — a FIXED 8-way key-dimension split for the
  window, independent of the row count, so a one-row decode and row 0 of
  a multi-row verify compute the window identically.
- `n_main` = `decode_n_split_` (= 32, the constructor default) when the
  main source runs, else 0 (`src/models/dsv4/csa2_layer.cu:319`).

### 1.2 The ring format (the window source's KV)

The window source reads the layer's own ring, `[max_decode_rows]
[ring_slots]` rows, **one block per request** (the identity block
table `ring_table_`), `ring_slots` = 160 (`Dsv4Csa2Config::ring_slots`,
`csa2_layer.hpp:53`; the validate requires `ring_slots >= window + 16`,
`csa2_layer.cu:36`). Each row is the **`kFp8Block`** latent format
(`src/kernels/latent_format.hpp:60-64`): the 512-dim RoPE'd kv as
**512 e4m3 codes + 16 e8m0 (power-of-two) scales, one per 32 elements**
= 528 bytes/row, self-describing (the scales are inside the row, no
beside row-scale). The ring slot for a row at absolute position `p` is
`p % ring_slots` (`csa2_ring_slot_positions`, `src/kernels/csa2.hpp:85-87`;
`csa2_layer.cu:380`). The window slot list per row is the ascending
positions `[max(0, p - window + 1), p]` mapped to ring slots, `-1`
padded to `[rows, window]` with counts (`csa2_window_slots_decode`,
`src/kernels/csa2.hpp:97-110`). The ring is READ AFTER the append, so
the batch's own rows are visible to their own queries
(`csa2_layer.cu:380-382`, the `dsa_latent_append` before the
`csa2_window_slots_decode`).

**Delta vs the reference (GAP G8):** the reference's ring is exactly
`window_size` = 128 slots (ck:model.py:261-274 `get_window_topk_idxs`,
the ring `self.kv_cache` of width 128), and its window KV quantizes
ONLY the non-RoPE 448 dims to fp8 (per-64, e8m0 scale) leaving the RoPE
64 in bf16 (ck:model.py:508-512). The C++ ring keeps 160 slots (the 32
extra older slots are never attended — the window set, the last 128
positions, is identical) and quantizes the FULL 512 (e4m3 + e8m0/32,
the RoPE 64 included) to the self-describing `kFp8Block` row. Same
window, a different quantization class.

### 1.3 The main format (the main source's KV)

The main source reads the planar compressed main cache, the
**`kFp4Block`** latent format (`src/kernels/latent_format.hpp:60-64`):
**256 e2m1 nibbles (2/byte) + 32 e4m3 block-scales, one per 16
elements = 288 bytes/row** (the card's number), self-describing. The
entries-per-block `epb = block_tokens / ratio` = 128 / 4 = **32** for
C4A, 128 / 128 = **1** for C128A (`src/models/dsv4/csa2_layer.cu:318`).
The main source is `dsa_attn_listed` (the 16-head-multiple tensor-core
form) over the selected `topk_` / `counts_`, falling back to
`dsa_attn_partial` when `lh` is not a multiple of 16
(`src/models/dsv4/csa2_layer.cu:320-328`). The identity main block
table (`main_block_table_`, one block per request, entry `e` at slot
`e`) is the no-pool positional state (`src/models/dsv4/csa2_layer.cu:95-99`).

**Delta vs the reference (GAP G8):** the reference stores the
compressed KV dequantized in bf16 (the `act_quant(..., inplace=True)`
at ck:model.py:378, the `kv_cache` buffer), and its quantization is the
per-64 e4m3/e8m0 over the 448 NoPE dims + the bf16 RoPE 64 (the 584 B
record, §4.2). The C++ main cache is the coarser self-describing
`kFp4Block` (e2m1/e4m3/16 over the full 512) — a documented quant-class
tolerance, not bit-exact to the reference.

### 1.4 The finish: the merge + the sink's exactly-once

`csa2_attn_finish` (`src/kernels/csa2.hpp:238-245`) merges the two
sources' partials and the sink into the output:

1. **Merge the splits within each source** (`dsa_attn_combine`
   contract, `src/kernels/dsa.hpp:358-365`):
   `c = (Σ_s p_s · c_s) / (Σ_s p_s · l_s)`, `p_s = exp(m_s - max m)` —
   the standard flash-decoding rescale. An empty source (a main source
   with `counts == 0`) has `m = -inf` so its `p_s = 0` and it
   contributes nothing.
2. **Merge the two sources** into one softmax state (the running max `m`
   and normalizer `l` across main + window).
3. **The sink's exactly-once** (`src/kernels/csa2.hpp:240` "the sink
   (exp(sink_h - m) in the denominator only, the reference's)"): the
   learned per-head `attn_sink` (fp32 `[local_heads]`) adds exactly
   `exp(sink_h - m)` to the MERGED denominator, ONCE — not per-split,
   not per-source. This matches the reference's `sum_exp[i] +=
   T.exp(attn_sink[i] - scores_max[i])` (ck:kernel.py:346;
   fn:kernel.py:383).
4. **Normalize + round**: `out = c / l`, ONE bf16 rounding.
5. **Inverse-rotate**: the query's rotation is removed from the last 64
   dims of every head (`csa2_rope_apply(..., inverse=true)`,
   `src/kernels/csa2.hpp:63-70`), matching the reference's `
   apply_rotary_emb(o[..., -rd:], freqs_cis, True)` (ck:model.py:539).
   Padding rows (`pos < 0`) are zeroed.

The sink's exactly-once + the degenerate limits are pinned by the CPU
oracle `dsv4_sparse_attn_sink_exactly_once`
(`tests/unit/dsv4_csa2_oracle_test.cpp:657-720`): the null / `-inf`
sink is bit-identical to no sink (the no-op), the `+inf` sink is the
EXACTLY-zero output, and a finite sink adds exactly ONE `exp(sink - m')`
to the merged denominator (`m' = max(mx, sink)`).

### 1.5 What the C++ already pins (the delta this spec adds)

- `dsv4_csa2_partial_layout`
  (`tests/unit/dsv4_csa2_oracle_test.cpp:721-803`) pins the six
  partials' `ws_slots` (row,split)-PRODUCT sizing and the
  `[r * n_split + s]` indexing bound on a synthetic shape.
- `dsv4_sparse_attn_sink_exactly_once` pins the finish's sink
  exactly-once + the +inf / no-tokens degenerate limits.

The delta this spec adds: the EXACT ring (`kFp8Block`, 160 slots,
`p % 160`) and main (`kFp4Block`, `epb` = 32/1) formats and their
byte layouts, the `n_split_win` = 8 vs `n_main` = 32 split counts, the
`[r * n_split + s]` indexing, and the finish's 5-step merge order — the
values a coder needs to wire the partials without guessing. The
reference-mapping (the single `sparse_attn` over the concatenated KV,
the sink in the denominator only, the inverse rotation) is what makes
the split-partial form numerically the reference's attention.
