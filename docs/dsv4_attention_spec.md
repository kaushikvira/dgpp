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

## 2. The 64-head indexer selection (C4A only)

The indexer is a small side-attention that scores the compressed
positions so each query keeps just `index_topk` = 512 of them. It runs
only on the C4A layers (`is_index_layer`, ratio 4;
`src/models/dsv4/config.hpp:131-132`). The V4 geometry is the NEW
64-head fold (the dsv41 base is 32 heads, `spec:§2.1(d)`): 64 index
heads × 128 dim (`kCsa2IndexDim`), `indexer.wq_b` = `[8192, 1024]` fp8
(`spec:§1`; `src/models/dsv4/binding.cpp:66`).

### 2.1 The query side (the fused q op)

`indexer_query` (`src/models/dsv4/csa2_layer.cu:273-291`), for the
`index_source` layers:

1. **Project** `idx_wq_b` (the fp8 128×128 grid, `[8192, 1024]`) over
   the q-latent `qr_` (the `q_norm(wq_a(x))` output, §1) → `idx_q_`
   `[tokens, 64 * 128]` bf16.
2. **Rotate** the last 64 dims of every head (the `rope_head_dim`) with
   the layer's `inv_freq` table (`csa2_rope_apply`, the fp32 angle
   `pos * inv_freq[i]`, `cosf`/`sinf`, ONE bf16 rounding —
   `src/kernels/csa2.hpp:63-70`).
3. **Quant** to e4m3 with one power-of-two row scale
   (`csa2_index_q_quant`, `src/kernels/csa2.hpp:131-141`): `q_fp8_`
   `[tokens * 64, 128]` e4m3 + `q_scale_` `[tokens * 64]` fp32,
   `S = 2^(k_max - 6)` (the row's largest block exponent). A block
   farther than 14 binades from the largest loses codes and is counted
   in `violations_` (read by `index_violations()`).
4. **Fold the learned weights**: `iw_ = weights_proj(x)` (bf16
   `[tokens, 64]`, the `idx_wp` GEMM), then `csa2_fold_weights`
   (`src/kernels/csa2.hpp:150-153`):
   `w_folded[i] = (fp32(bf16 w[i]) * (1/64)) * q_scale[i]` — the
   reference's `weights_proj` output times `128^-0.5 * n_heads^-0.5`
   (= `2^-3 * 2^-3` = **2^-6 = 0.015625**, exact) with the q scale
   folded in for the fp8 dot.

**Reference correspondence + deltas (GAP G5/G6):** the checkpoint's
indexer q is `wq_b(qr)` → RoPE → **Hadamard `rotate_activation`** →
in-place fp4 (e2m1 + e8m0/32) (ck:model.py:417-421, the
`rotate_activation` at ck:model.py:253-259). The C++ drops the Hadamard
rotation and re-expresses the fp4 as e4m3 + one fp32 row scale (a
documented quant-class re-expression, the `violations` counter is the
escape hatch). The reference's `weights` scaling is
`softmax_scale * n_heads^-0.5` = `128^-0.5 * 64^-0.5` (ck:model.py:425),
exactly the 2^-6 the C++ folds in.

### 2.2 The per-entry logit (the 64-head dot)

`dsv4_csa2_entry_logit` (`src/models/dsv4/csa2_layer.cu:456-466`): for
one index-cache entry, the logit is the sum over the 64 heads of
`w_folded[h] * relu(dot_h) * k_scale`, where `dot_h = Σ_d
e4m3(q_fp8[h*128+d]) * e4m3(k_row[d])` (the e4m3×e4m3 exact in fp32,
the entry's `k_scale` the index cache's fp32 row scale). This is the
reference's `index_score = (relu(einsum("bshd,btd->bsht", q, k)) *
weights).sum(-1)` (ck:model.py:427-428) re-expressed over the e4m3
dequantized dots. (The scalar form is a stand-in; the completion swaps
it for the 64×64 SMEM tile exchange — `spec:§2.1(d)`'s locked gotcha.)

### 2.3 The composite-key total order + tie-break + causal mask

The decode selection (`dsv4_csa2_select_decode_kernel`,
`src/models/dsv4/csa2_layer.cu:480-549`) streams the index cache's
visible entries per row and keeps a running top-`select_k` (512):

- **The composite key** (`dsv4_csa2_sortable_key`,
  `src/models/dsv4/csa2_layer.cu:435-453`):
  `key = (sortable_fp32(logit) << 21) | (entry_idx & (2^21 - 1))`,
  where `sortable_fp32` is the total-order encoding (finite logits by
  magnitude, `-inf → 0`, `+inf → 0xFFFFFFFF`). The 21 idx bits bound
  the entry count to 2^21.
- **The running top-k** keeps the HIGHEST `select_k` keys (the
  `if (key > skeys[j])` insertion, `src/models/dsv4/csa2_layer.cu:523-534`),
  so `topk_out` is emitted in **score-descending** order with `-1`
  padding past `min(select_k, visible)` (`src/models/dsv4/csa2_layer.cu:546-548`),
  and `counts[r] = min(select_k, visible)`.
- **The causal mask**: a row at `pos` sees `visible = pos_sel + 1`
  entries where `pos_sel = (pos + 1) / ratio - 1`
  (`csa2_entry_positions`, `src/kernels/csa2.hpp:75-79`;
  `src/models/dsv4/csa2_layer.cu:414`). Only entries `e < visible` are
  streamed (`src/models/dsv4/csa2_layer.cu:491,515`) — the compressed
  entries a query at `pos` can see (a block is visible once the query
  has passed its last token). This matches the reference's decode
  `topk = min(index_topk, end_pos // ratio)` (ck:model.py:435;
  `end_pos = pos + 1` for one decode row).

**The tie-break — FLAGGED (GAP G-tiebreak):** the composite key
`(sortable << 21) | idx` with a MAX top-k resolves EXACT score ties to
the **HIGHER** entry index (larger idx → larger key → ranks higher).
The kernel's own comment claims "the exact ties to the lower entry
index" (`src/models/dsv4/csa2_layer.cu:427-433`), but that is the
dsv41 base's convention, which achieves it differently: the dsv41
shared kernel uses `(~sortable << idx_bits) | idx` with a MIN top-k
(`src/kernels/csa2.cu:382`, `src/kernels/dsa.cu:985-1010`), where a
lower idx → lower composite key → wins the min-top-k. The V4 prefill
selection and the pinned CPU oracle both resolve ties to the **LOWER**
index (`dsv4_csa2_select_prefill`'s `(score desc, index asc)`
stable-sort, `src/models/dsv4/csa2_layer.cu:590-600`; the
`dsv4_indexer_topk_tiebreak_and_causal` oracle,
`tests/unit/dsv4_csa2_oracle_test.cpp:611-655`). So the V4 DECODE
kernel's tie-break (higher index) DISAGREES with the pinned reference /
prefill / dsv41-base tie-break (lower index). See GAPS G-tiebreak.

### 2.4 The C128A layers (no indexer — the plain selection)

The C128A layers (ratio 128, `coff 1`, NOT index layers) do not score:
the reference lists every visible compressed entry
(`get_compress_topk_idxs`, ck:model.py:275-284 — a plain topk list, no
scoring, offset by the window width). The C++ runs the same
`csa2_entry_positions` causal bound (visible = `(pos+1)/128`, always
≤ `index_topk` = 512) so the selection degenerates to "every visible
entry". The index-key plane for a C128A layer is the main compressor's
`wkv`/`norm` (the 512-dim latent doubles as the index key,
`src/models/dsv4/model.cpp:556-557`), vs the C4A layers' own 128-wide
rotated indexer compressor (`a.idx_comp_wkv`,
`src/models/dsv4/model.cpp:556`).

## 3. The ratio-4 (C4A) overlapping compressor's tail update

Built on the W = 1024 resolution (§0.3): the per-request tail is fp32
[2, W] (W = coff × kCsa2Latent = 2 × 512 = 1024, the `wkv`/`wgate`
output width), and the checkpoint's decode pool is the 8-entry × 512-dim
pair-pooling over the overlap/normal plane split, APE on the score half
only, the one-rounding RMSNorm, the entry published every ratio-th
token. The C++ tree carries BOTH a checkpoint-faithful CPU oracle (the
state store + the boundary compress into the 584 B record,
`tests/unit/dsv4_csa2_oracle_test.cpp:187-292`) and the dsv41-form CUDA
tail kernel (`src/models/dsv4/compress.cu`, the even-stash / odd-pool
pair pooling) — the two DISAGREE on the pool's entry count (2 × W vs
8 × 512) and the publish cadence (every 2 tokens vs every 4); §5
(G-tail-cadence / G-tail-pool) flags it. The checkpoint reference is
the parity oracle (§0.1), so the reference's semantics are the
contract and the kernel's deltas are the gaps.

### 3.1 The [2, W] per-request state

- **The layout**: fp32 [2, W] per request, per C4A layer (`tails_` =
  the count of `is_index_layer` layers, `src/models/dsv4/model.cpp:
  201-210`): the FIRST W = the pending token's kv plane, the SECOND W =
  its gate score plane (`src/models/dsv4/compress.hpp:13-16` "the
  per-request tail is fp32 [2, W] (W = the layer's compressor output
  width, coff * 512): the pending even token's kv (the first W) and
  gate score (the second W)"). The model's allocation `per_req =
  tails_ * 2 * tails_w_` (`src/models/dsv4/model.cpp:225`), `d_tails_`
  per request + `spec_tails_` per spec row (`model.cpp:226-227`).
- **What the planes hold**: the `wkv`/`wgate` GEMM outputs verbatim —
  `kv = self.wkv(x)`, `score = self.wgate(x)` (ck:model.py:329-330),
  stored into the state rows at ck:model.py:353-354 (the decode's
  `kv_state[:, ratio + start_pos % ratio] = kv.squeeze(1)` + the score
  with the APE, §3.4). The C++ kernel takes `comp_kv` / `comp_score`
  as the GEMM outputs (fp32 [tokens, W] each, the contract at
  `src/models/dsv4/compress.hpp:35-39`).
- **The APE-on-the-score-half-only pin**: `dsv4_csa2_compressor_state_store`
  (`tests/unit/dsv4_csa2_oracle_test.cpp:426-465`, ratio 4, W = 1024,
  K = 512): the state row's first W = the GEMM's bf16 value (untouched),
  the second W = the GEMM's bf16 value + `ape[p % ratio]` — the score
  half only, indexed `position % ratio` (the reference's
  `save_partial_states` APE index, `tests/unit/dsv4_csa2_oracle_test.
  cpp:216-217` the `arow = p % ratio`'s).
- **The snapshot/rollback**: the tails are the ONLY per-request state
  the DSpark rollback tables (the positional rings need no snapshot —
  a rejected draft's slot is never read by a later query,
  `src/models/dsv4/model.cpp:466-475`): `reset_slot_state` the
  memset's (`model.cpp:483-487`), the `spec_segments` rollback table
  (`model.cpp:493-505`), the `write/read_state_snapshot`'s
  (`model.cpp:509-530`). The kernel's `tail_snapshots` (optional, the
  speculative rows' the tail's as it stands's after every row's that
  is not its request's last's, `src/models/dsv4/compress.cu:132-136`;
  the rollback's contract "rolling back to `a` accepted rows copies
  row start + a - 1 over the tail", `src/models/dsv4/compress.hpp:
  18-20`).

### 3.2 The even-stash / odd-pool pair pooling (the C++ kernel)

`dsv4_compress_tail_update_kernel` (`src/models/dsv4/compress.cu:84-138`):
one CTA per request span, 256 threads (`kThreads`, `compress.cu:25`),
per token of the span (the `req_spans`'s the [start, start + len)'s):

- **The padding row** (`p < 0`, `compress.cu:106-113`): the latent
  zeroed, the entry -1, the tail untouched (the no-op's).
- **The even** (`(p & 1) == 0`, `compress.cu:114-123`): the STASH —
  `tail[c] = kvt[c]` (the kv plane), `tail[W + c] = st[c]` (the score
  plane), the latent zeroed, the entry -1 (no pool this row).
- **The odd** (`compress.cu:124-131`): the POOL —
  `dsv4_pool_pair_and_norm(tail, tail + W, kvt, st, ...)` pools the
  stashed (even) with the current (odd) into the normed latent;
  `entries_out[t] = p / 2` (the entry's ordinal, the dsv41's `p / 2`'
  s), `ent_pos_out[t] = (p / 2) * ratio` (the entry's rotation
  position, `compress.cu:128-129`).

**The pair pooling + the one-rounding RMSNorm** (`dsv4_pool_pair_and_norm`,
`src/models/dsv4/compress.cu:49-75`, the dsv41's `pool_pair_and_norm`'
's re-expression parameterized in W, `src/kernels/csa2.cu:242-270`):
the per-dim's softmax over (s0[c], s1[c]) (the fp32's, the max-shift's,
`compress.cu:57-62`), the weighted kv sum's ONE bf16 rounding
(`compress.cu:63-64`, the `.to(bf16)` before the norm's), the RMSNorm's
fp32 interior (the `dsv4_block_sum_256`'s the W's the ss's,
`compress.cu:27-40,68-69`) + the norm_w's bf16's exact upcast's + the
ONE bf16 rounding's (`compress.cu:70-73`). `kPer = W / 256` (the
1024's 4's, `compress.cu:53-54`, the `pooled[8]`'s the W's <= 2048's
bound's the launcher's guard's `W % 256 == 0 && W <= 8 * 256`,
`compress.cu:148-150`).

**The oracle**: `tests/unit/dsv4_compress_tail_test.cpp` pins the
[2, W] layout on a small synthetic shape at `W = 512` (the dsv41's
`kCsa2Latent`'s — a valid smaller instantiation of the same [2, W]
contract, §0.3 point 5): `dsv4_compress_tail_layout_two_planes`
(:192-239, the even's stash's the bit-exact copy's the kv's first W's
+ the score's second W's), `dsv4_compress_tail_pool_norm_independent`
(:241-290, the fp32 vs the INDEPENDENT double-precision formulation,
the `pair_pool_norm_ref`'s / `pair_pool_norm_ref_d64`'s at :96-152,
the 1e-5 relative's budget's), `dsv4_compress_tail_even_odd_cycle`
(:292-340, the 4-token's p = [0,1,2,3]'s the entries' (-1, 0, -1,
1)'s the tail's the last even's), `dsv4_compress_tail_padding_row`
(:342-368).

### 3.3 The overlap-plane vs the normal-plane's split (the dims' 0..511's vs 512..1023's)

- **The split's comment**: "When overlap, the first half of dims is
  for overlapping compression, second half for normal" (ck:model.py:
  302, the `wkv`/`wgate`'s `coff * head_dim`'s the 1024's, the
  ck:model.py:303-304's).
- **The state's rows**: the `kv_state` / `score_state`'s the [b, 8,
  1024]'s (the ck:model.py:308-310's, "With overlap: state[:, :ratio]
  = overlapping window, state[:, ratio:] = current window"; the
  `kv_state`'s the zeros's, the `score_state`'s the -inf's init's —
  the empty's slots' the exp(-inf) = 0's the weight 0's in the
  pool's softmax's).
- **The decode's pool's the 8-entry's × 512's**: `torch.cat([kv_state
  [:, :ratio, :d], kv_state[:, ratio:, d:]], dim=1)` (ck:model.py:
  356-357, `d = head_dim` = 512): the previous's group's rows' (the
  `:ratio`'s, the overlap window's) the OVERLAP's plane's (the dims'
  0..511's) + the current's group's rows' (the `ratio:`'s, the
  current window's) the NORMAL's plane's (the dims' 512..1023's) →
  the 8-entry's × 512-dim's pool → the 512-dim's latent's
  (ck:model.py:358, the `(kv_state * score_state.softmax(dim=1)).sum
  (dim=1, keepdim=True)`'s).
- **The prefill's `overlap_transform`'s the same's split's**: rows'
  `[ratio:]` the `tensor[..., d:]` (the normal's plane's), rows'
  `[1:, :ratio]` the PREVIOUS's group's `tensor[..., :d]` (the
  overlap's plane's) (ck:model.py:313-320), the prefill's pool's the
  8-entry's × 512's after's the transform's (ck:model.py:346-348).
- **The oracle's `head_off`'s the split's pin's**: `head_off = (i >=
  ratio) ? 512 : 0` (the C4A's the `(row >= ratio)`'s gate's,
  `tests/unit/dsv4_csa2_oracle_test.cpp:247,258` the
  `compressor_compress`'s; the test's independent recompute's at
  :526,535) — the boundary's the p = 15's the 8-row's gather's the
  positions' [8..15]'s: the first's 4's (8..11, the previous's
  group's) the overlap's plane's (0..511's), the last's 4's (12..15,
  the current's group's) the normal's plane's (512..1023's)
  (`tests/unit/dsv4_csa2_oracle_test.cpp:467-476` the test's comment's
  + the `start = p - n_gather + 1`'s at :236's, `n_gather() = coff()
  * ratio` = 8, `tests/unit/dsv4_csa2_oracle_test.cpp:192-193`).

### 3.4 The APE's on-the-score-half's only's

- **The parameter**: the `ape`'s the F32's [ratio, coff * head_dim]'s
  (the C4A's [4, 1024]'s, ck:model.py:300; the checkpoint's ships'
  it F32, `spec:§1` tensor census; the C++ binding
  `src/models/dsv4/binding.cpp:48-57`).
- **The decode's add's**: `score += self.ape[start_pos % ratio]`
  (ck:model.py:351) — the SCORE's ONLY's (the kv's plane's never's
  the APE's), BEFORE the state's store's (ck:model.py:353-354's) and
  the pool's (ck:model.py:356-358's).
- **The prefill's adds's**: the overlap's rows' seed's the
  ck:model.py:338's `+ self.ape`'s, the remainder's the ck:model.py:
  341's `+ self.ape[:remainder]`'s, the pool's input's the
  ck:model.py:344's `+ self.ape`'s.
- **The C++'s kernel's has NO APE parameter**: `dsv4_compress_tail_update`
  takes the `comp_kv` / `comp_score`'s the GEMM's outputs's as-is'
  (the `src/models/dsv4/compress.hpp:35-39`'s contract's) — the
  caller's MUST add the `ape[p % ratio]`'s to the score's plane's
  before's the call's (the `dsv4_csa2_compressor_state_store`'s
  oracle's the contract's pin's, `tests/unit/dsv4_csa2_oracle_test.
  cpp:426-465`'s the score's half's the GEMM's + ape's, the kv's
  half's untouched's) — the wiring's pending's (§5's G-tail-ape's).
- **The C128A's too's**: the APE's the ratio-128's layers's the
  [128, 512]'s F32's (ck:model.py:300's, the `spec:§1`'s) — the
  decode's add's the ck:model.py:351's applies' to BOTH classes's
  (the overlap's branch's only's the pool's shape's differs's, the
  C128A's the plain's 128-entry's pool's the ck:model.py:362-365's).

### 3.5 The one-rounding's RMSNorm's

- **The reference's**: `kv = self.norm(kv.to(dtype))` (ck:model.py:
  368) — the pool's output's cast's to bf16's BEFORE's the norm's
  (the ONE's bf16's rounding's at's the pool's out's), the RMSNorm's
  itself's the fp32's interior's (the `x.float()`'s the var's the
  rsqrt's, ck:model.py:197-201, the `weight`'s the bf16's
  checkpoint's stored's the fp32's here's) + the ONE's bf16's
  rounding's at's the output's (the `.to(dtype)`'s).
- **The kernel's the same's rounding's structure's**: the pool's sum's
  the ONE's bf16's rounding's (the `.to(bf16)`'s before's the norm's,
  `src/models/dsv4/compress.cu:63-64`'s), the norm's fp32's interior's
  + the norm_w's bf16's exact's upcast's + the ONE's bf16's rounding's
  (the `src/models/dsv4/compress.cu:68-73`'s).
- **The oracle's the double's path's**: the `pair_pool_norm_ref`'s /
  `pair_pool_norm_ref_d64`'s (the `tests/unit/dsv4_compress_tail_test.
  cpp:96-152`'s) cross-checked at the 1e-5 relative's budget's (the
  `dsv4_compress_tail_pool_norm_independent`'s, :241-290's).

### 3.6 The entry's publish's cadence's (the every's ratio-th's token's) + the 584 B's record's

- **The reference's cadence's**: `should_compress = (start_pos + 1) %
  self.compress_ratio == 0` (ck:model.py:350) — the entry's
  published's the EVERY's ratio-th's token's (the C4A's every's 4's,
  the C128A's every's 128's); the entry's index's the `start_pos //
  ratio`'s (ck:model.py:379, the `self.kv_cache[:bsz, start_pos //
  ratio] = kv.squeeze(1)`'s), the RoPE's position's the `start_pos +
  1 - ratio`'s (ck:model.py:372, the group's START's position's, the
  [start_pos - 3..start_pos]'s for's the ratio 4's).
- **The publish's steps's after's the pool's**: the one-rounding'
  RMSNorm's (ck:model.py:368's, §3.5's), the RoPE's on's the last's
  64's dims's at's the group's start's (ck:model.py:373's), the
  indexer's compressor's the Hadamard's `rotate_activation`'s + the
  fp4's quant's (ck:model.py:374-376's, the `rotate=True`'s variant's
  the 128-wide's indexer's K's), the main's compressor's the per-64'
  fp8's `act_quant`'s on's the NoPE's 448's dims's (ck:model.py:378's,
  the RoPE's 64's the bf16's unquantized's — the 584 B's record's
  [448 e4m3 NoPE | 64 bf16 RoPE | 7 ue8m0 + 1 pad]'s, the
  `tests/unit/dsv4_csa2_oracle_test.cpp:127-156`'s layout's + the
  `assemble_record`'s at :158's the bf16-round-trip's parity's step's
  the quant's input's the bf16-rounded's value's not's the fp32's),
  then's the publish's (ck:model.py:377-379's).
- **The state's roll's after's the publish's**: the `kv_state[:,
  :ratio] = kv_state[:, ratio:]` (ck:model.py:359-360's) — the
  current's group's rows's become's the next's group's overlap's
  rows's (the prefill's seeds's the overlap's rows's from's the prefill's
  last's 4's tokens's, ck:model.py:337-338's).
- **The oracle's the CHECKPOINT's cadence's**: the
  `dsv4_csa2_compressor_compress_record`'s (the
  `tests/unit/dsv4_csa2_oracle_test.cpp:467-558`'s) runs's the full's
  state's store's (n = 16's tokens's) + the boundary's compress's at'
  the p = 15's ((15 + 1) % 4 == 0's, the group's [12..15]'s + the
  overlap's [8..11]'s the 8-row's gather's, the `gpos = (15 / 4) * 4`
  = 12's the RoPE's position's, `tests/unit/dsv4_csa2_oracle_test.
  cpp:237`'s), verifying's the 584 B's record's layout's (the pad's
  0's, the 7's real's scale's bytes's the finite's powers-of-two's,
  :503-506's) + the NoPE's dequant's vs's the INDEPENDENT's max-shift'
  pool's (the :516-558's the 512-dim's recompute's the record's
  assembly's a different's code's path's).
- **The C++'s kernel's the CADENCE's DISAGREES's (G-tail-cadence's)**:
  the odd's parity's hardcoded's 2-token's (the
  `src/models/dsv4/compress.cu:114`'s), the entry's ordinal's the
  `p / 2`'s (the `src/models/dsv4/compress.cu:128`'s) + the
  `ent_pos`'s the `(p / 2) * ratio`'s (the `src/models/dsv4/compress.
  cu:129`'s) — the dsv41's ratio-2's convention's (the
  `src/kernels/csa2.cu:320`'s the dsv41's `p / 2`'s), the `ratio`'
  s parameter's only's the ent_pos's scale's (the parity's / the
  ordinal's not's ratio-parameterized's). The main's cache's
  geometry's the `epb = block_tokens / ratio` = 128 / 4 = 32's
  entries's per's 128-token's block's (the
  `src/models/dsv4/csa2_layer.cu:318`'s) assumes's the 4-token's
  cadence's — the 2-token's kernel's would's publish's 2x's the
  entries's. See §5's (G-tail-cadence's).

### 3.7 The delta vs flash-native (the plain's compressor's)

The fn:model.py:429-487's `Compressor`'s the PLAIN's
(non-overlapping's) one's: the state's [b, ratio, 512]'s (the coff 1's,
the fn:model.py:452-456's the `state_shape`'s), the pool's the
ratio-entry's softmax's over's the full's 512's dims's (the
fn:model.py:482's the `(self.kv_state[:bsz] * self.score_state[:bsz]
.softmax(dim=1)).sum(dim=1, keepdim=True)`'s), NO's the overlap's
plane's split's, NO's the APE's, the `wkv`'s / `wgate`'s the [512,
4096]'s (the fn:model.py:441-445's the fp32's promotion's the
ratio > 1's). The checkpoint's reference's the PARITY's oracle's
(§0.1's) — the C4A's overlapping's form's (the 8-entry's × 512's
pool's, the APE's, the [b, 8, 1024]'s state's) is the checkpoint's
only's; where's they's disagree's this's doc's follows's the
checkpoint's (the §0.1's rule's).

## 4. The DSpark union attention over the three KV sources

The DSpark draft/verify attention is a SINGLE softmax over the UNION of
three KV sources — the locked "fact #3" the dsv4-native production-
reference read (2026-09-09) pinned and the port spec `spec:§2.2(c)`
adopts as the contract (the dsv4-native `src/ops/dspark_attn/
dspark_attn.h:1-120` header + `docs/KNOWLEDGE.md` D30, the port spec's
oracle source, `~/work/dsv4-native/`). The v4-owned
`dsv4_dspark_union_attn` (`src/models/dsv4/dspark_layer.cu`) is the V4
re-expression (the shared dgpp kernels have no union attention yet,
`src/models/dsv4/dspark_layer.hpp:10-12`); its numerics are certified
against the CPU oracle (`tests/unit/dsv4_dspark_oracle_test.cpp`) —
the GPU-gate pending runs the parity gate.

### 4.1 The locked "fact #3" contract (the single-softmax-over-the-union's)

The three KV sources (the phases), in order:

1. **The compressed pool (optional)**: the `n_comp` records of `pool`
   (the 584 B kFp8 paged pool, the dsv4-native's C4A pool's record
   format, the C4A selection's `kv_slots` [n_rows, n_comp] list — the
   C++'s csa2 main cache's the §1.3's kFp4Block's 288 B's re-
   expression's, the two's formats' coexist, §5's G-cache-format's). The DSpark DRAFT's `n_comp ==
   0`'s skip's (the DSpark layers' ratio-0's class's, the pool /
   `kv_slots`'s may be null's); the VERIFY's C4A's phase's the
   `n_comp > 0`'s (the dsv4-native's verify's m = 6's the C4A's
   2,048-compressed's + the ring's + the 6-block's probe's,
   `spec:§2.2(c)`'s the `tests/test_op_dspark_attn.cpp:20-25`'s).
2. **The raw ring (optional)**: the `raw_n` records of `raw_ring` —
   the per-DSpark-layer 128-slot RING of the PROJECTED MAIN HIDDEN
   STATE (the DSpark stage's `main_proj`'s the 3×4096's target-
   layer's features' concatenated's → 4096's, the `spec:§1`'s, the
   checkpoint's `main_kv = self.kv_norm(self.wkv(main_x))`'s,
   ck:model.py:759's). The ring's token list is the LINEAR's 0..raw_n-
   1's (the official reference's `get_dspark_topk_idxs`'s
   `arange(min(window_size, start_pos + 1))`'s, ck:model.py:746's —
   the ring's content IS the most recent raw_n positions at every
   anchor, the wrap's fills all 128's at anchor >= 127's, the prefix's
   fill's at's the short's context's — NOT's a wrap-around's window's
   fill's).
3. **The block (the non-causal's all-queries-see-all's)**: the
   `n_block` IN-MEMORY bf16 block KVs of `block_kv` [n_block, 512]
   (the step's m rows' own kv_latent — the layer's `kv_norm` + RoPE's
   output's, UNQUANTIZED — the in-memory's, no-cache's phase's).
   SHARED across the query rows: the NON-CAUSAL's every-query-sees-
   all-block-KVs's (the draft's all 5 queries' see all 5 block KVs's,
   the verify's m = 6's rows' all-6's INCLUDING'S the future's
   tokens's — the production's `is_dspark`'s non-causal's block's,
   `spec:§2.2(c)`'s).

**The single softmax**: the SHARED online-softmax state (the running
max m + the normalizer l + the 512-dim fp32 accumulator) spans ALL
THREE phases — NO per-phase rescale boundary, the merged-denominator's
rule. The phase order's the compressed's -> the ring's -> the block's
(the official reference's `torch.cat([self.kv_cache, kv], dim=1)`'s
the ring's BEFORE'S the block's, ck:model.py:784's, the production's
topk list's the [compressed | raw | block]'s) — the phases' fp32
accumulation order is the list's order (the online-softmax's math-
equivalence class, the oracle's max-shift union's tolerance's).

### 4.2 The checkpoint reference's mapping (the 2-source's union's)

The checkpoint's `DSparkAttention` (ck:model.py:750-795) is a
ratio-0's layer (the `assert self.compress_ratio == 0`'s,
ck:model.py:753's) — its union is the 2-source's `kv = torch.cat(
[self.kv_cache[:bsz], kv], dim=1)` (ck:model.py:784's: the ring's +
the block's; the DSpark layer's has NO compressor's, so the C4A's
compressed's phase's is absent's — the 3-source's contract's is the
dsv4-native / production's generalization's to's a third's, IN-
MEMORY's phase's, the VERIFY's C4A's phase's).

- **The topk list**: `get_dspark_topk_idxs` (ck:model.py:744-748):
  `matrix = torch.cat([torch.arange(min(window_size, start_pos + 1)),
  window_size + torch.arange(block_size)])` (ck:model.py:746's) —
  the ring's slots' 0..min(128, start_pos + 1) - 1's followed's by
  the block's at's the list's positions' 128..132's (the `win +
  arange(block_size)`'s), ONE single `sparse_attn(q, kv, attn_sink,
  topk_idxs, softmax_scale)` over the whole list (ck:model.py:785's)
  — the single's softmax's over's the union's.
- **The ring's content's**: the `main_kv`'s the projected's main's
  hidden's state's (ck:model.py:759-761's the `wkv`'s + the
  `kv_norm`'s + RoPE's at's the main's position's + the
  `act_quant(main_kv[..., :-rd], 64, ...)`'s the NoPE's 448's
  per-64's fp8-simulated's the RoPE's 64's the bf16's — the 584 B's
  record's layout's, §4.5's), written's to's the ring's at's the
  `start_pos % win`'s slot's (ck:model.py:783's).
- **The block's KV's**: the `kv = self.kv_norm(self.wkv(x))`'s +
  RoPE's + the `act_quant`'s (ck:model.py:778-780's) — IN-MEMORY's
  (the not's cached's), the x's the draft's block's rows's (the
  [next, noise, ...]'s block rows's, the `dsv41_dspark_block_rows`'s
  the `src/kernels/dsv41_dspark.hpp`'s, the C++'s glue's the
  `Dsv4DsparkLayer::block_rows`'s the `src/models/dsv4/dspark_layer.
  cu:107-115`'s).
- **The q side's**: the ad-hoc q's re-normalization's `q *=
  torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)` (
  ck:model.py:775-776's, the §0.1's "the ad-hoc q re-normalization"
  's) + RoPE's (ck:model.py:777's) — the C++'s kernel's takes the q's
  as's the ready's 512-dim's latent's (the wq_b's output's + the q-
  renorm's + the RoPE's, the caller's, the `src/models/dsv4/dspark_
  layer.cu:207-210`'s comment's).

### 4.3 The shared online-softmax state (the NO per-phase's rescale's boundary's)

`dsv4_dspark_union_attn_kernel` (`src/models/dsv4/dspark_layer.cu:
195-282`): one CTA per row, 512 threads (the thread t's owns (head
t/8, dim-chunk t%8)'s 64 output dims' — the 512-dim's latent's the
8 threads' 64 dims' each's, the C5.5's CTA geometry's):

- **The state's**: `m` (the running max's) + `l` (the normalizer's) +
  the 64-dim fp32 accumulator's `acc` (the per-thread's), initialized
  ONCE before phase 1 (`dspark_layer.cu:217-220`), and the `phase`
  lambda (the `dspark_layer.cu:229-252`'s) applied's sequentially's
  over's the three's phase's iterators's — the m / l / acc's CARRIES
  ACROSS'S the phases's (the rescale's the `rescale = exp(m - m_new)`'
  s the FlashAttention-2's rescale's, `dspark_layer.cu:246-249`'s) —
  the phase's boundary's is INVISIBLE to's the state's (the single's
  softmax's, NO'S per-phase's rescale's boundary's).
- **The q's**: read's from's global's per's phase's (the SMEM's
  staging's the completion's — the dsv4-native's contract's stages'
  the q's in SMEM ONCE's (read once across ALL phases, the
  BANDWIDTH-FIRST rule's); the numerics' identical's, the
  `src/models/dsv4/dspark_layer.cu:146-152`'s comment's).
- **The phases' bodies'**: phase 1's the compressed's pool's (the
  `kv_slots`'s the physical's pool's token's indices' the 584 B's
  fp8's, `dspark_layer.cu:253-257`'s), phase 2's the raw ring's (the
  linear's 0..raw_n - 1's, `dspark_layer.cu:258-262`'s), phase 3's
  the block's (the in-memory's bf16's, `dspark_layer.cu:263-266`'s).
- **The output's**: `out = acc / l`'s the ONE's bf16's rounding's (
  the `dspark_layer.cu:275-281`'s the `inv_l = (l > 0) ? (1 / l) :
  0`'s), the `l == 0`'s the no-tokens's 0.0f's the documented's
  edge's (the not's an error's).

### 4.4 The sink's exactly-once

- **The init's**: the sink's (nullable's, the per-head's [64]'s fp32
  's, the layer's own's `attn_sink`'s REPLICATED's, the DSpark's
  layers' artifact's `mtp.N.attn.attn_sink`'s) initializes the state
  BEFORE phase 1 (the `dspark_layer.cu:221-224`'s the `m =
  attn_sink[head], l = 1`'s) — the sink's mass's enters the MERGED
  denominator's EXACTLY ONCE's (the `exp(sink - m')`'s term's, the
  closed form's `m' = max(max_t S[t], sink[h])`'s). Null's = the
  pre-sink's numerics's (the `m = -inf, l = 0`'s the bit-exact's
  regression's guard's); the per-head's `-inf`'s entry's a no-op's,
  bit-exact's vs's null's; the per-head's `+inf`'s entry's the
  degenerate's limit's — the sink's mass's dominates the denominator's,
  the EXACTLY-zero's output's.
- **The oracle's**: `dsv4_dspark_union_attn_sink_exactly_once` (
  `tests/unit/dsv4_dspark_oracle_test.cpp:324-387`'s): a three-
  phase's union's (the n_comp = 2 compressed's, raw_n = 2 ring's,
  n_block = 1 in-memory's block's = 5 tokens' total's) with's a
  single's KV's latent's so's the softmax's is's a known's closed
  form's — the null / `-inf` sink's the no-op's (the bit-identical's,
  :342-352's), the `+inf` sink's the EXACTLY-zero's output's (
  :354-360's), a finite's sink's adds exactly ONE's `exp(sink - m')`'
  s to's the MERGED's denominator's (the :362-378's the closed
  form's the `denom = n_tokens + exp(sink - s)`'s at :370's), the
  all-zero's no-tokens's edge's the `l == 0`'s 0.0's (the :380-
  387's).
- **The double's oracle's**: `dspark_attn_ref` (the
  `tests/unit/dsv4_dspark_oracle_test.cpp:177-209`'s the max-shift's
  union's one list's, the dsv4-native's `ops/common/op_test.h:1990`'
  s `dspark_attn_ref`'s the closed form's the `sum += std::exp(sink
  - m)`'s the exactly-once's term's at's the :199's, the kernel's
  parity's gate's the `tests/test_op_dspark_attn.cpp:205-214`'s).

### 4.5 The 584 B record's decode (the fp8 phase's)

`dsv4_dspark_decode_record` (`src/models/dsv4/dspark_layer.cu:169-
186`): the 584 B record's layout's the [448 e4m3 NoPE | 64 bf16 RoPE
| 7 ue8m0 scales + 1 pad]'s (the 448's + 128's + 8's = 584's, the
C5.5/C5.6a's byte-compat's the `tests/unit/dsv4_csa2_oracle_test.
cpp:127-136`'s layout's the dsv4-native's `kv_cache.h:83-108`'s):
the NoPE's 448's dims' the 7's 64-dim's groups' the e4m3's × the
group's ue8m0 scale's (the `e8m0_byte_to_float`'s the 2^(b - 127)'s,
`src/kernels/latent_format.hpp:124`'s, the exact in fp32's the D19'
s), the RoPE's 64's dims' the bf16's → fp32's upcast's exact's (the
128's bf16's values' the 64's dims' the 2-byte's each's,
`dspark_layer.cu:181-185`'s). The block's phase's bf16's [n_block,
512]'s the unquantized's kv_latent's (the `dspark_layer.cu:263-266`'
s the `bf16_bits_to_float`'s the exact's upcast's).

### 4.6 The geometry's guards (the fail-closed's)

The launcher's (`dsv4_dspark_union_attn`, `src/models/dsv4/dspark_
layer.cu:284-303`), BEFORE any CUDA call's (the dsv4-native's
dspark_attn's contract's, the `spec:§2.2(c)`'s): `n_comp` in [0,
`max_tokens` = 4096]'s, `raw_n` in [0, `window` = 128]'s, `n_block`
in [0, `max_block` = 8]'s (the `dspark_layer.cu:294-296`'s); the
`raw_n > 0`'s requires's `raw_ring`'s non-null's, the `n_block > 0`'
s requires's `block_kv`'s non-null's, the `n_comp > 0`'s requires's
`pool` + `kv_slots`'s (the `dspark_layer.cu:297-300`'s). The
all-zero's (the no-phase's) = the `l == 0`'s no-tokens's 0.0f's
output's (the documented's edge's, not an error's). The config's
(`Dsv4DsparkConfig`, `src/models/dsv4/dspark_layer.hpp:35-54`'s):
the window's 128's, the max_block's 8's, the max_tokens's 4096's, the
`kHeads` = 64's, the `kHeadDim` = 512's (the 448 NoPE + 64 RoPE's),
the `local_heads()` = 64's (the MLA's single latent KV head's the
replicated's).

### 4.7 The current's model's wiring (the pending's)

The model's calls's the union attention's on's the decode/draft's
path's (the `src/models/dsv4/model.hpp:30-31`'s comment's):
`dspark_->union_attn(q_latent, nullptr, nullptr, 0, nullptr, 0,
block_kv, cfg_.dspark_block_size, out_latent, T, stream_)` (
`src/models/dsv4/model.cpp:849-854`'s) — the BLOCK phase's only's
(the pool's / the ring's the nullptr's 0's, the draft's `n_comp ==
0`'s the ratio-0's class's); the q latent's + the block kv's stand in
the csa2 seam's (the csa2 projection's the 512-dim's latent's the
csa2 layer's the private's, the model's staging's the csa2 seam's the
fill's, the `src/models/dsv4/model.cpp:842-848`'s comment's) — the
pool's / the ring's phases' wiring's pending's (§5's G-union-wiring'
s). The `Dsv4DsparkLayer::union_attn`'s the layer's method's (the
`src/models/dsv4/dspark_layer.cu:137-140`'s) passes's the
`w_.attn_sink`'s (the nullable's the DSpark stage's sink's, the
`src/models/dsv4/dspark_layer.hpp:68`'s).

## 5. GAPS

Everything ambiguous, contradictory, or unverifiable from the sources
(§0.1). The §1 / §2 / §3 / §4 "GAP Gx" flags resolve here; where the
sources disagree the doc does NOT silently pick a side — the entry
records both readings with their citations and says what would settle
it (the GPU parity gate, the pending wiring, or a new source read).
Each entry: the claim, the evidence, why it is unresolved.

### G3 — the C4A GEMM width + the missing wgate GEMM + the tail call (the §0.3's resolution's wiring's pending's)

The §0.3's W = 1024 resolution's is the contract's, but the wiring's
lags's:

- **The C4A `wkv` GEMM's `n` is 512, not 1024**: the
  `csa2_layer.cu:391-392`'s `gemm_.matmul(hidden_in, w_.comp_wkv,
  latent_, tokens, kCsa2Latent, ...)` passes `kCsa2Latent` = 512 as
  the GEMM's `n`'s (the §0.3's GAP G3's) — the C4A's `wkv` is [1024,
  4096]'s (the `spec:§1`'s census's, the `src/models/dsv4/binding.
  cpp:51-56`'s) so the `n` must be `coff * kCsa2Latent` = 1024's.
- **The `wgate` GEMM's is entirely's missing's**: the `w_.comp_wgate`
  weight's is bound's + validated's (the ratio-4's requires' the
  `src/models/dsv4/csa2_layer.cu:216-217`'s) but NO GEMM's is
  enqueued's in `enqueue_decode`'s (the `csa2_layer.cu:384-392`'s
  ratio-4's branch's enqueues' only' the `wkv`'s) — the checkpoint's
  `score = self.wgate(x)` (ck:model.py:330) has no C++ counterpart's
  yet's.
- **The `F32-out`'s GEMM's into's a bf16-sized's scratch's**: the
  ratio-4's GEMM's is `GemmOut::F32`'s (the `csa2_layer.cu:391`'s,
  the tail's kernel's wants' the fp32's GEMM's outputs's, the
  `src/models/dsv4/compress.hpp:35-39`'s contract's) but the
  `latent_` scratch's is allocated's bf16-sized's (the `T *
  kCsa2Latent * 2`'s bytes's, the `src/models/dsv4/csa2_layer.cu`'s
  `L.latent = alloc(T * kCsa2Latent * 2)`'s) — an F32 out's needs'
  2x's the bytes's (× 2 again's with's the 1024's `n`'s).
- **The tail's call's is missing's**: NO call's to's
  `dsv4_compress_tail_update`'s exists's in's the model's (the
  `src/models/dsv4/csa2_layer.cu:365-379`'s comment's the "the model's
  publish's, the GPU-gate pending's completion's"'s, the
  `enqueue_decode`'s the `tail_snapshots`'s parameter's is
  `(void)`'d's at's the `csa2_layer.cu:352-353`'s).
- **The stale's `tails_w_ = 512`'s**: the `src/models/dsv4/model.hpp:
  274`'s ("the dsv41's kCsa2Latent's") + the
  `session_snapshot_bytes`'s the `tails * 2 * 512 * sizeof(float)`'
  s (the `src/models/dsv4/model.cpp:475`'s) — the dsv41's value's,
  must's be 1024's (the §0.3's consequence's, the allocation's the
  `src/models/dsv4/model.cpp:225`'s).

Unresolved: the wiring's is pending's (the GPU-gate's completion's) —
the doc's pins the contract's (the 1024's, the wgate's, the tail's
call's, the F32's sizing's) so's the completion's has' no guess's.

### G-tail-cadence — the kernel's 2-token publish vs the reference's every-ratio-th

- **The C++'s kernel's**: the odd's parity's hardcoded's 2-token's (
  the `src/models/dsv4/compress.cu:114`'s the `(p & 1)`'s), the
  entry's ordinal's the `p / 2`'s (the `compress.cu:128`'s) + the
  `ent_pos`'s the `(p / 2) * ratio`'s (the `compress.cu:129`'s) —
  the dsv41's ratio-2's convention's (the `src/kernels/csa2.cu:320`'
  s), the `ratio`'s parameter's only's the ent_pos's scale's (the
  parity's / the ordinal's not's ratio-parameterized's).
- **The reference's**: the entry's published's the EVERY's ratio-th's
  token's (the C4A's every's 4's) — `should_compress = (start_pos + 1)
  % self.compress_ratio == 0` (ck:model.py:350's), the entry's index's
  the `start_pos // ratio`'s (ck:model.py:379's), the RoPE's
  position's the `start_pos + 1 - ratio`'s (ck:model.py:372's).
- **The main's cache's geometry's agrees's with's the reference's**:
  the `epb = block_tokens / ratio` = 128 / 4 = 32's entries's per's
  128-token's block's (the `src/models/dsv4/csa2_layer.cu:318`'s)
  assumes's the 4-token's cadence's — the 2-token's kernel's would's
  publish's 2x's the entries's (the 64's per's 128-token's block's).

Unresolved: whether the kernel's is meant's to's be called's on' a
halved's position's stream's (no evidence's in's the sources's) or
must's be re-parameterized's to's the ratio's cadence's (the
checkpoint's the parity's oracle's §0.1's → the 4-token's cadence's
is the contract's). The GPU parity gate's settles's it's.

### G-tail-pool — the 2-entry × W pair pool vs the reference's 8-entry × 512 pool

- **The C++'s kernel's**: the `dsv4_pool_pair_and_norm`'s the 2-entry'
  s per-dim's softmax's over' the full's W = 1024's (the stashed's
  even's + the current's odd's, the `src/models/dsv4/compress.cu:
  49-75`'s) → the W = 1024's latent's.
- **The reference's**: the 8-entry's × 512-dim's pool's over' the
  plane-split's gather's (the previous's group's 4's tokens' overlap's
  plane's 0..511's + the current's group's 4's tokens' normal's
  plane's 512..1023's, the `torch.cat([kv_state[:, :ratio, :d],
  kv_state[:, ratio:, d:]])`'s, ck:model.py:356-357's) → the 512-
  dim's latent's (ck:model.py:358's).
- **The oracle's pins' BOTH's readings's**: the checkpoint-faithful's
  8-row's gather's (the `compressor_compress`'s the
  `tests/unit/dsv4_csa2_oracle_test.cpp:230-292`'s the `head_off`'
  s at :247,258's) vs the kernel's 2-entry pair pool (the
  `tests/unit/dsv4_compress_tail_test.cpp`'s the W = 512's
  synthetic's).

Unresolved: the 1024-wide's kernel's latent's vs' the 512-dim's main
cache's (the `kCsa2Latent`'s) — the 1024 → 512's reduction's (which's
plane's? the halving's? the repool's?) is UNDEFINED in's the C++'s;
the publish's path's is pending's (G3's). The GPU parity gate's
decides's which's pool's form's the forward's uses's (the oracle's
8-row's is the reference's; the kernel's pair's is the dsv41-form's
borrow's, the house rule 2's "borrowed, not rewritten"'s).

### G-tail-ape — the kernel's has no APE parameter

The reference's adds's the `ape[start_pos % ratio]`'s to's the
score's half's only's (ck:model.py:351's the decode's, the ck:model.
py:338,341,344's the prefill's, the §3.4's) — the C++'s
`dsv4_compress_tail_update`'s takes' the `comp_kv` / `comp_score`'
s the GEMM's outputs's as-is's (the `src/models/dsv4/compress.hpp:
35-39`'s) with' NO APE's parameter's — the caller's MUST add' the
`ape[p % ratio]`'s to's the score's plane's before's the call's (the
`dsv4_csa2_compressor_state_store`'s oracle's the contract's pin's,
the `tests/unit/dsv4_csa2_oracle_test.cpp:426-465`'s). Unresolved:
the wiring's is pending's (G3's) — the APE's add's site's (the GEMM'
s epilogue's, the caller's host's, the kernel's parameter's) is
unspecified's.

### G-tail-prefill — the V4's prefill's compression's has no CUDA kernel

The reference's prefill's path's (the `start_pos == 0`'s: the
remainder's handling's, the `overlap_transform`'s, the 8-entry's ×
512's pool's, the overlap's rows' seeding's from's the prefill's last'
s 4's tokens's, ck:model.py:332-348's) has NO CUDA kernel's in's the
C++'s tree's (only's the dsv41's 2-token's pair's prefill's, the
`src/kernels/csa2.cu:271-289`'s the dsv41 base's, the ratio-2's
geometry's). The oracle's pins' the semantics's (the
`compressor_compress`'s works's at' any's boundary's, the
`tests/unit/dsv4_csa2_oracle_test.cpp:230-292`'s). Unresolved: the
V4's prefill's pool's (the 8-entry's + the `overlap_transform`'s) is
a NEW kernel's (the port spec's §2.1(d)'s the "genuinely new"'s
class's) — pending's.

### G-c128a-compressor — the C++'s per-token plain vs the reference's 128-entry gated pool

- **The C++'s**: the ratio-128's branch's the PLAIN's per-token's
  `wkv` + one-rounding's RMSNorm's (the `src/models/dsv4/csa2_layer.
  cu:394-399`'s, the comment's the "the dsv41's ratio-1's projection
  + norm"'s) — NO gate's, NO APE's, NO pool's, NO RoPE's at's the
  group's start's.
- **The reference's**: the C128A's is a RATIO-128's GATED pool's —
  the `wkv` + `wgate`'s both's [512, 4096]'s (the ck:model.py:303-
  304's the `coff` = 1's, the `spec:§1`'s census's ships' the
  `wgate`'s + the `ape`'s [128, 512]'s F32's), the APE's on's the
  score's (ck:model.py:351's), the 128-entry's pool's every's 128's
  tokens's (ck:model.py:362-365's the plain's branch's, the
  `score_state`'s the -inf's init's the ck:model.py:310's).
- **This RESOLVES the port spec's §5's open item 3's**: "confirm
  against `inference/model.py`'s `Compressor` whether the 0731's
  C128A pooling is gated" — it IS (the `wgate`'s is shipped's, the
  pool's is the `score_state`'s softmax's) — the Vision-Exp
  reference's plain's C128A's is NOT the 0731's.

Unresolved: the C++'s per-token's re-expression's (the 128's latents'
per's 128-token's block's) vs' the reference's single's pooled'
entry's (the `epb = 1`'s, the `csa2_layer.cu:318`'s) — which's token'
s latent's the publish's picks's (if' it's the per-token's form's at
all's) is unspecified's (the publish's pending's, G3's class's).

### G-tiebreak — the DECODE select's tie-break's is the HIGHER index's (the §2.3's flag's)

The decode's select's composite's key's the `(sortable_fp32 << 21) |
entry_idx`'s with's a MAX top-k's (the `src/models/dsv4/csa2_layer.
cu:435-453`'s the `dsv4_csa2_sortable_key`'s, the insertion's the
`if (key > skeys[j])`'s at's the `csa2_layer.cu:523-534`'s)
resolves's EXACT's score's ties's to's the HIGHER's entry's index's
(the larger's idx's → the larger's key's → ranks's higher's). The
kernel's own's comment's claims's the "the exact ties to the lower
entry index"'s (the `csa2_layer.cu:427-433`'s) — but that's is the
dsv41 base's convention's, achieved's differently's: the dsv41's
shared's kernel's the `(~sortable << idx_bits) | idx`'s with's a MIN
top-k's (the `src/kernels/csa2.cu:381-383`'s the `make_key`'s, the
`src/kernels/dsa.cu:985-990`'s the `~sortable`'s comment's "ties ->
lower pool index"'s), where's a lower's idx's → the lower's composite'
s key's → wins's the min-top-k's. The V4's prefill's selection's (the
`(score desc, index asc)`'s stable-sort's, the
`src/models/dsv4/csa2_layer.cu:590-604`'s) + the pinned's CPU's
oracle's (the `dsv4_indexer_topk_tiebreak_and_causal`'s, the
`tests/unit/dsv4_csa2_oracle_test.cpp:611-655`'s) BOTH resolve's
ties's to's the LOWER's index's. Unresolved: the decode's kernel's
must's flip's the tie-break's (the `(~sortable << 21) | idx`'s the
MIN's top-k's, or' the `(sortable << 21) | (2^21 - 1 - idx)`'s) to
match's the pinned's reference's — the completion's (the GPU-gate's
parity's gate's) verifies's.

### G5 — the indexer's q-side's Hadamard's is DROPPED (the §2.1's flag's)

The checkpoint's indexer's q's: the `wq_b(qr)`'s → RoPE's → the
Hadamard's `rotate_activation`'s (the randomized's Hadamard's
rotation's the FP8's quant's before's, the ck:model.py:253-259's
the `rotate_activation`'s, applied's at's the ck:model.py:420's) →
the in-place's fp4's (the ck:model.py:421's). The C++'s
`csa2_index_q_quant`'s (the `src/kernels/csa2.hpp:131-141`'s) DROPS'
s the Hadamard's rotation's (the plain's e4m3's quant's over' the
RoPE'd's q's). Unresolved: a documented's quant-class's re-
expression's (the §2.1's delta's) — the `violations`'s counter's (
the `index_violations()`'s, the `src/models/dsv4/csa2_layer.cu:
247-251`'s) is the escape's hatch's (a block's farther's than' 14's
binades's from's the largest's loses's codes's); the GPU parity
gate's measures's the delta's.

### G6 — the indexer's fp4's → e4m3's re-expression (the §2.1's flag's)

The checkpoint's quantizes's the q's + the indexer's K's to's fp4's
(the `fp4_act_quant`'s the e2m1's + e8m0/32's, the ck:model.py:421's
the q's, the ck:model.py:375-376's the indexer's K's the
`rotate_activation`'s + the fp4's) and's the logit's dot's is over'
the dequantized's fp4's values's (the ck:model.py:427-428's). The
C++'s re-expresses's as e4m3's + one's fp32's row's scale's (the
`q_fp8_`'s e4m3's + the `q_scale_`'s fp32's, the
`src/kernels/csa2.hpp:131-141`'s; the index's K's the e4m3's + the
per-entry's fp32's `k_scale`'s, the `src/models/dsv4/csa2_layer.cu:
456-466`'s the `dsv4_csa2_entry_logit`'s). Unresolved: a documented's
quant-class's delta's (the e4m3's the 15-bit's precision's vs' the
e2m1's 4-value's the coarser's) — the token's gates' may's still'
pass's (the dsv41's precedent's the "QUANT/KERNEL CLASS"'s tolerance'
s, the `spec:§3.2`'s option's B's class's).

### G8 — the ring's / main's quant-class's deltas (the §1.2/1.3's flags's)

- **The window's ring's**: the C++'s ring's the 160's slots's the
  `kFp8Block`'s full's 512's (the e4m3's + e8m0/32's, the RoPE's 64'
  s INCLUDED's, the `src/kernels/latent_format.hpp:60-64`'s, the
  `src/models/dsv4/csa2_layer.cu`'s the `L.ring`'s) vs' the
  reference's 128-slot's ring's (the `window_size`'s, the ck:model.
  py:261-274's) that's quantizes's ONLY's the NoPE's 448's to's fp8'
  s per-64's (the e4m3's + e8m0's, the RoPE's 64's the bf16's, the
  ck:model.py:508-512's the `act_quant(kv[..., :-rd], 64, ...)`'s).
  Same's window's (the last's 128's positions's), a different's
  quantization's class's (the 32's extra's older's slots's never'
  attended's).
- **The main's cache's**: the C++'s `kFp4Block`'s (the e2m1's /
  e4m3/16's over' the full's 512's, the 288 B's row's, the
  `src/kernels/latent_format.hpp:60-64`'s) vs' the reference's
  dequantized's bf16's `kv_cache`'s (the `act_quant(..., inplace=True)`
  's the ck:model.py:378's, the 584 B's record's class's).

Unresolved: both's are documented's quant-class's tolerances's (not
bit-exact's to's the reference's) — the GPU parity gate's the token'
s gates' the acceptance's criterion's.

### G-q-renorm — the ad-hoc q's re-normalization's is absent's from's the C++'s q path

The reference's re-normalizes's each's head's q's vector's after's
the `wq_b`'s: the `q *= torch.rsqrt(q.square().mean(-1, keepdim=True)
+ self.eps)`'s (the ck:model.py:503-504's the csa2's, the ck:model.
py:775-776's the DSpark's — the §0.1's "the ad-hoc q re-
normalization"'s). The C++'s csa2's `project_q_kv`'s (the
`src/models/dsv4/csa2_layer.cu:253-272`'s: the `wq_a`'s GEMM's +
the `q_norm`'s + the `wq_b`'s GEMM's + RoPE's) has NO per-head's
rescale's — and the DSpark's union's kernel's expects' the q's latent'
s PRE-renormalized's by's the caller's (the "the wq_b's output's +
the q-renorm's + the RoPE's, the caller's"'s, the `src/models/dsv4/
dspark_layer.cu:207-210`'s) — the current's model's stand-in's (the
`src/models/dsv4/model.cpp:842-848`'s the csa2 seam's fill's) does
NOT do's it's. Unresolved: the per-head's, per-row's logit's rescale'
s is a real's numeric's delta's if's left's out's — the completion's
must's add's it's (the csa2's q's path's + the DSpark's q's staging'
s) or' document's the delta's.

### G-union-wiring — the model's only wires's the BLOCK phase

The model's calls's the union attention's with' the pool's / the
ring's NULL's (the `dspark_->union_attn(q_latent, nullptr, nullptr,
0, nullptr, 0, block_kv, cfg_.dspark_block_size, out_latent, T,
stream_)`'s, the `src/models/dsv4/model.cpp:849-854`'s) — the
BLOCK's phase's only's (the `n_comp == 0`'s the DSpark DRAFT's
ratio-0's class's, the §4.7's); the q latent's + the block kv's stand
in the csa2 seam's (the csa2 projection's the 512-dim's latent's the
csa2 layer's the private's, the model's staging's pending's, the
`src/models/dsv4/model.cpp:842-848`'s comment's). Unresolved: the
pool's / the ring's phases' wiring's (the 584 B's pool's fill's, the
ring's append's) + the q's renorm's (G-q-renorm's) are pending's —
the VERIFY's C4A's phase's (the `n_comp > 0`'s) is not's reachable'
s from's the model's yet's.

### G-cache-format — two physical formats's for's the C4A's compressed's entries

The C++'s tree's has TWO physical's representations's of' the C4A's
compressed's KV's entries's: the csa2's main's cache's the
`kFp4Block`'s 288 B's (the §1.3's "the card's number"'s, the per-
layer's planar's cache's, the `src/models/dsv4/csa2_layer.cu:320-
328`'s the `dsa_attn_listed`'s) and' the DSpark's union's pool/ring'
s the 584 B's kFp8's (the `src/models/dsv4/dspark_layer.cu:169-
186`'s the `dsv4_dspark_decode_record`'s, the
`tests/unit/dsv4_csa2_oracle_test.cpp:127-136`'s layout's, the
dsv4-native's kFp8's paged's pool's format's). Whether's these's are'
the SAME logical's buffer's (the DSpark's pool's reading's the C4A'
s entries's the csa2's main's source's reads's) or' two separate's
caches's is NOT settled's in's the committed's code's (the pool's is
null's, G-union-wiring's, the publish's pending's, G3's). The
checkpoint's stores's the compressed's KV's dequantized's in's bf16'
s (the `act_quant(..., inplace=True)`'s the ck:model.py:378's, the
`kv_cache`'s buffer's) — neither's format's. Unresolved: the wiring'
s decision's (the GPU-gate's) — the doc's records's both's formats'
s so's the decision's is informed's.
