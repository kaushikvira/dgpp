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
