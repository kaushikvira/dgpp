# dsv4 vs dsv4-native diff — cross-check of the V4-Flash numerics

Independent cross-check of OUR `dsv4` implementation against a second,
independent C++/CUDA engine for the same model family: `~/work/dsv4-native`
(from-scratch, oracle-qualified, drove a live serving lane on this kit).
Where the three disagree, the **Python checkpoint reference** is the
authority:
`/data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local/inference/model.py`
(cited `ref:model.py:N`; `ref:kernel.py:N` for the attention/sink kernel).

Line shorthands:
- `us:` = this repo (`/home/kv/work/dgpp`, branch `dsv4-flash` @ 9aaa2ae).
- `nat:` = `~/work/dsv4-native`.
- `ref:` = the Python checkpoint reference above.

Classification of every difference:
- **(a) a real bug in ours** — changes the model's output, not a documented
  quant-class tolerance.
- **(b) a documented quant-class tolerance** — a coarser/different
  quantization or re-association that is expected to be a small drift, not a
  wiring error.
- **(c) cosmetic** — no effect on the output.

**Scope note (do-not-duplicate):** the two q-path gaps the other agent owns
— `G-q-renorm` (the reference's per-head unit-RMS q re-normalization) and
`G5` (the indexer's Hadamard `rotate_activation`) — are **not** re-derived
here; this doc only records where `dsv4-native` **agrees** or **disagrees**
with that framing (it agrees on both; see §1.4, §3).

---

## 1. The RoPE

### 1.1 Frequency construction (the inv_freq table)

| | construction |
|---|---|
| **us** | Two host tables, one per layer class, built by `yarn_rope_inv_freq_host` (`us:src/kernels/rope_scaling.cpp:17-49`, the YaRN mix: `freqs = 1/theta^(2i/64)`, and with `correction_max_position > 0` the `low/high` correction range `dim*ln(max/(rot*2π))/(2 ln theta)` + the linear ramp `freqs = fr/f*(1-smooth) + fr*smooth`). Built at `us:src/models/dsv4/model.cpp:300-303`: **window (SWA, ratio 0)** = `theta 10000, original_seq_len 0` (no correction → plain); **compressed (ratio > 0)** = `theta compress_rope_theta 160000, original_seq_len = original_max_position_embeddings (65536)` (YaRN-corrected, factor 16, β_fast 32, β_slow 1). Table picked per layer at `us:src/models/dsv4/model.cpp:665` (`ratio > 0 ? inv_freq_compressed_ : inv_freq_window_`). |
| **nat** | `dsv4::rope::yarn_inv_freq` (`nat:src/ops/rope/rope.cpp:36-104`) — the identical YaRN mix (same `correction_dim`, `floor/ceil` bounds clamped to `[0, dim-1]`, same ramp `inv = interp*(1-mask) + extrap*mask`). Built per layer at `nat:src/targets/deepseek_v4/model_forward_tp2.cpp:1222-1238`: **SWA (ratio 0)** = the plain `1/theta^(2j/64)` (θ 10000, no correction); **compressor (ratio > 0)** = `yarn_inv_freq(θ=160000, factor 16, 64, 32, 1, 65536)`. |
| **ref** | `precompute_freqs_cis` (`ref:model.py:206-237`) — `freqs = 1/base^(arange(0,dim,2)/dim)`, and with `original_seq_len > 0` the same `find_correction_range` + `linear_ramp_factor` blend `freqs/factor*(1-smooth) + freqs*smooth`. Table chosen per layer at `ref:model.py:480-487`: `compress_ratio > 0` → `original_seq_len=65536, theta=compress_rope_theta (160000)`; else (`== 0`) → `original_seq_len=0, theta=rope_theta (10000)` ("disable YaRN and use base rope_theta in pure sliding-window attention"). |

**Verdict: (c) cosmetic / agreement.** All three build the *same* table:
SWA layers get the plain θ 10000 table, compressor layers get the
YaRN-corrected θ 160000 table with the identical low/high/ramp math. The
`dsv4-native` comment at `nat:…model_forward_tp2.cpp:1215-1221` even names
the rank-4 fix that made SWA use the plain table — ours already does
(`us:…model.cpp:300` passes `0` → no correction). No output difference.

### 1.2 Where it is applied (q? k? head split, interleaving)

| | application |
|---|---|
| **us** | `csa2_rope_apply` (`us:src/kernels/csa2.hpp:63-70`, the kernel `us:src/kernels/csa2.cu`) rotates **adjacent GPT-J pairs** (`x[2i], x[2i+1]`) of the **last `rope_dim`=64 dims** of each head; angle = fp32 `pos * inv_freq[i]`, `cosf/sinf`, one bf16 rounding; `inverse` conjugates. Applied to: **kv** (`us:src/models/dsv4/csa2_layer.cu:304`), **q** (`csa2_layer.cu:308`), the **indexer q** (`csa2_layer.cu:321`), the **index key** (`csa2_layer.cu:384`), and the **main latent on publish** (`csa2_layer.cu:395`). The output's inverse-rotation is in the finish (`us:src/kernels/csa2.hpp:238-245`, step 5). |
| **nat** | `dsv4::rope::rope_device` (`nat:src/ops/rope/rope.cu`, GPT-J on the last `rope_dim`=64 of the 512-dim head; `rope.h:24` `f = p*inv_freq[j]`, `out[2j]=bf16(x[2j]c - x[2j+1]s)`, `out[2j+1]=bf16(x[2j+1]c + x[2j]s)`). Applied to **q_latent** and **kv_latent** at `nat:…model_forward_tp2.cpp:1260-1262` (both, after q-renorm), and the inverse on `attn_out` at `nat:…model_forward_tp2.cpp:1591`. |
| **ref** | `apply_rotary_emb` (`ref:model.py:238-251`) — `view_as_complex(x.float().unflatten(-1,(-1,2)))` = **adjacent GPT-J pairs**, `inverse` → conjugate. Applied to **q** (`ref:model.py:505`), **kv** (`ref:model.py:510`), the **indexer q** (`ref:model.py:419`), the **compressed kv** (`ref:model.py:373`), and inverse on **o** (`ref:model.py:539`). |

**Verdict: (c) agreement.** All three: GPT-J (non-NeoX) adjacent pairs on
the last 64 dims of each 512-dim head, applied to **both** q and k (kv
latent), fp32 angle `pos*inv_freq`, inverse-rotate the attention output.
No output difference. (The `is_neox_style=False` convention is shared.)

### 1.3 Per-row RoPE positions (the MTP/verify case)

`nat` had a real bug here (fixed): the shared `m=1` decode generalization
gave **all** `r>1` verify rows the **same** RoPE position, which was "the
garbage's symptom" — fixed to per-row `pos + i`
(`nat:…model_forward_tp2.cpp:1240-1258`, the comment at `:1241-1246`).
**Ours** builds per-row position arrays (`us:src/models/dsv4/model.cpp:776-779`
`pos[i] = first_pos + i`; `rows.pos` flows into `csa2_rope_apply` as a
per-row array, `us:src/kernels/csa2.hpp:63-70`).

**Verdict: (c) agreement** — we already do the per-row positions `nat`
required. (Worth re-verifying on the GPU gate that `rows.pos` is per-row in
every decode/MTP path, since this is exactly the class of bug that produced
garbage in `nat`.)

### 1.4 The q-side re-normalization (the `G-q-renorm` gap — the other agent's)

| | behaviour |
|---|---|
| **us** | **ABSENT.** `project_q_kv` (`us:src/models/dsv4/csa2_layer.cu:292-310`) = `wq_a → q_norm → wq_b → RoPE`. There is **no** per-head unit-RMS rescale of `q` between `wq_b` and RoPE. (Documented `GAP G-q-renorm`; the DSpark union kernel also expects the q pre-renormalized by the caller, `us:src/models/dsv4/dspark_layer.cu:207-210`.) |
| **nat** | **PRESENT** — the C6.4 rank-1 fix: a weightless RMSNorm (all-1.0 weight) over the 512-dim head, AFTER `wq_b`, BEFORE the RoPE, at `nat:…model_forward_tp2.cpp:1156-1201` (`rmsnorm_bf16_device(q_latent, qrenorm_ones, …)`), the ref's `model.py:539`/`dspark.py:464` `q *= rsqrt(mean(q²)+eps)`. The comment: *"the engine's MISSING it was the C6.4 diff report's rank-1 candidate."* |
| **ref** | **PRESENT** — `ref:model.py:504` `q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)` (and the DSpark q at `ref:model.py:775-776`). |

**Verdict: (a) a real bug in ours** (the missing step). `dsv4-native`
**corroborates** the q-path agent's framing — it is the *rank-1* candidate
for the output divergence, and `nat`'s `DSV4_QRENORM_DUMP` diagnostic
(`nat:…model_forward_tp2.cpp:1158-1190`) reports the **pre-renorm per-head
RMS is O(2–4)**, i.e. the renorm rescales each head's q by **~1/2–1/4,
non-uniformly per head**. That is a large, head-varying logit-scale
distortion (not a uniform temperature change) — see §6.

---

## 2. The attention KV path

### 2.1 The window ring (slots, positions, per-64 quant, which dims)

| | behaviour |
|---|---|
| **us** | Per-layer ring of **`ring_slots` = 160** rows (`us:src/models/dsv4/csa2_layer.hpp:61`), row = the **`kFp8Block`** latent format: **512 e4m3 codes + 16 e8m0 (power-of-two) scales, one per 32 elements = 528 B/row** (`us:src/kernels/latent_format.hpp:44-64`, `kLatentFp8BlockGroup = 32`). **The FULL 512 is quantized — the RoPE 64 included.** Ring slot for position `p` = `p % 160` (`us:src/kernels/csa2.hpp:85-87`); the window slot list = the ascending positions `[max(0, p-127), p]` mapped to ring slots (`us:src/kernels/csa2.hpp:97-110`, `csa2_window_slots_decode`). The ring is read **after** the append so a row's own token is visible (`us:src/models/dsv4/csa2_layer.cu:380-382`). |
| **nat** | A **128-slot** ring (2-page **584 B kFp8** paged layout, the C5.6a `raw_ring`, byte-compat with the record below), slot `pos % 128` (`nat:src/targets/deepseek_v4/model_forward_tp2.cpp:1367-1372`; `nat:src/ops/sparse_attn/sparse_attn.h:146-210`). The record keeps **NoPE 448 in fp8 (7×64, ue8m0) and the RoPE 64 in bf16 (unquantized)** — `nat:src/ops/kv_cache/kv_cache.h:18-26,83-100` (`kNoPEEls 448`, `kRoPEBytes 128`, `kRecordBytes 584`). |
| **ref** | A **`window_size`=128**-slot ring (`ref:model.py:261-274` `get_window_topk_idxs`, the `kv_cache` width 128); the window KV **quantizes ONLY the non-RoPE 448 dims to fp8 (per-64, e8m0) and leaves the RoPE 64 in bf16** — `ref:model.py:512` `act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)` ("FP8-simulate non-rope dims to match QAT; rope dims stay bf16 for positional precision"). |

**Verdict: (b) a documented quant-class tolerance (the spec's `G8`), with one
sub-point that is a real (a)-class delta:**
- **Which dims are quantized** — we quantize **all 512** (e4m3 + e8m0/32);
  `nat`/`ref` quantize **NoPE 448 only** (per-64) and keep **RoPE 64 in
  bf16**. This is the `G8` "first bullet." It is a quant-class re-expression
  (documented), **but** quantizing the RoPE 64 to e4m3 (3-bit mantissa,
  ~6–12 % relative error) corrupts the **positional** signal that the
  reference deliberately keeps exact — so it is a *larger* drift than the
  NoPE re-quant, and it is the **dominant** KV source on a short prompt
  (§6). Classified (b) per the spec, flagged as the top drift.
- **The 160 vs 128 slot count** — the 32 extra older slots are never
  attended (the window set, the last 128 positions, is identical); (c)
  cosmetic. (Both rings hold the same 128 most-recent positions; the wrap
  modulus differs but the window content is identical for all `p`.)
- **e8m0/32 vs per-64** — our per-32 scale is *finer* than the ref's
  per-64 for the NoPE dims, so the NoPE re-quant is at least as good; the
  net drift is dominated by the RoPE-64 quantization above.

### 2.2 The main / compressed cache format

| | behaviour |
|---|---|
| **us** | The planar **`kFp4Block`** main cache: **256 e2m1 nibbles + 32 e4m3 block-scales (one per 16) = 288 B/row**, over the **full 512** (`us:src/kernels/latent_format.hpp:48-52`), the identity block table (`us:src/models/dsv4/csa2_layer.cu:95-99`), `epb = block_tokens/ratio` = 32 (C4A) / 1 (C128A) (`us:src/models/dsv4/csa2_layer.cu:318`). The DSpark union's pool/ring is a separate **584 B kFp8** record (`us:src/models/dsv4/dspark_layer.cu:169-186`) — two physical formats for the same logical entries (`GAP G-cache-format`). |
| **nat** | A single **584 B kFp8** paged pool (NoPE 448 fp8 + RoPE 64 bf16 + 8 scale B) for **both** the compressed C4A/C128A records and (byte-compat) the raw ring — `nat:src/ops/kv_cache/kv_cache.h:83-100`, `nat:src/ops/sparse_attn/sparse_attn.h`. |
| **ref** | The compressed KV is stored **dequantized in bf16** in the single `kv_cache` buffer (the `act_quant(..., inplace=True)` at `ref:model.py:378`, the 584 B record class), and the window + compressed share **one** `kv_cache` buffer (`ref:model.py:474,538`). |

**Verdict: (b) a documented quant-class tolerance.** Our `kFp4Block`
(e2m1/e4m3/16 over the full 512) vs `nat`/`ref`'s 584 B fp8-NoPE/bf16-RoPE
(or the ref's bf16-dequantized buffer) is a coarser, different quantization
class (the spec's `G8` second bullet / `G-cache-format`). It affects the
compressed entries only — **nearly empty on a 15-token prompt** (§6).

### 2.3 The two-source merge (window + compressed) and the sink

| | behaviour |
|---|---|
| **us** | A **two-source** attention: the window ring (`dsa_attn_partial`, `us:src/models/dsv4/csa2_layer.cu:420-422`) and the selected main rows (`dsa_attn_listed` / `dsa_attn_partial`, `csa2_layer.cu:424-438`), each a split-KV partial (`m/l/c` × main/window, the `[r*n_split+s]` layout, `csa2_layer.cu:104`), merged in one finish (`csa2_attn_finish`, `us:src/kernels/csa2.hpp:238-245`): merge the splits, merge the two sources into one softmax state, the **sink exactly-once** (`exp(sink_h - m)` in the merged denominator, `csa2.hpp:240`), normalize, one bf16 round, inverse-rotate. |
| **nat** | A **single online-softmax over the union** (compressed → ring → block), no per-phase rescale boundary, the sink exactly-once (`nat:src/ops/sparse_attn/sparse_attn.h`, `nat:src/ops/dspark_attn/`, `docs/spec/attention.md` "The attention sink (D25)"). |
| **ref** | A **single** `sparse_attn` over the concatenated `[window | compressed]` topk list, the learned per-head `attn_sink` in the denominator, `softmax_scale = head_dim**-0.5` — `ref:model.py:520,533,538` (the sink term at `ref:kernel.py:346`). |

**Verdict: (c) agreement.** Our split-partial + finish merge is
algebraically the *same single softmax* as `nat`/`ref` (within the fp32
accumulation budget); the sink's exactly-once + the `+inf`/no-tokens
degenerate limits are pinned by the CPU oracle
(`us:tests/unit/dsv4_csa2_oracle_test.cpp:657-720`). The softmax scale is
`512**-0.5` in all three (`us:src/models/dsv4/csa2_layer.cu:181`,
`nat`/`ref` `head_dim**-0.5`). No output difference.

---

## 3. The indexer / selection (C4A only)

### 3.1 The q-side transform chain

| | chain |
|---|---|
| **us** | `indexer_query` (`us:src/models/dsv4/csa2_layer.cu:312-324`): `idx_wq_b` (fp8 `[8192,1024]`) over the q-latent → **GPT-J RoPE on the last 64** (`csa2_layer.cu:321`) → **`csa2_index_q_quant` (e4m3 + ONE fp32 row scale, `us:src/kernels/csa2.hpp:131-141`) — the Hadamard `rotate_activation` is DROPPED** (`GAP G5`, the other agent's) → `weights_proj` (bf16 `[T,64]`) → `csa2_fold_weights` (`us:src/kernels/csa2.hpp:150-153`, `w_folded = bf16(w)*(1/64)*q_scale`, the ref's `128^-0.5 * 64^-0.5 = 2^-6` folded in). |
| **nat** | `indexer_q_fused_device` (`nat:src/ops/indexer_token/indexer_token.h:110-160`, `.cu:156-237`): `q [r,64,128]` bf16 (the `wq_b` output) → GPT-J RoPE on the last 64 → **the 64-head learned-weight FOLD into a single 128-dim vector** `qsum[r,d] = (128^-0.5 * 64^-0.5) * Σ_h w[r,h]*q_rope[h,d]` → **per-row fp8 quant (amax/448, e4m3)** → `q_fp8 [r,128] + q_scale [r]`. **No Hadamard, no fp4, and the fold is BEFORE the dot (see 3.2).** |
| **ref** | `ref:model.py:417-422`: `q = wq_b(qr)` → RoPE → **`rotate_activation(q)` (the Hadamard)** → **`fp4_act_quant(q)` (e2m1 + e8m0/32)**; `weights = weights_proj(x) * (128^-0.5 * 64^-0.5)` (`ref:model.py:424`). |

**Verdict: (a) two real deltas, both flagged; (b) one quant-class delta:**
- **G5 — the Hadamard is dropped** in ours (and in `nat`): a documented
  quant-class re-expression, but the Hadamard *spreads* the q before the
  low-precision quant; dropping it changes which dims carry the mass.
  (The other agent's `G5`.)
- **G6 — fp4 → e4m3 re-expression**: we quantize the q to e4m3 + one fp32
  row scale; `ref` uses fp4 (e2m1 + e8m0/32). A coarser/finer quant-class
  delta → (b).
- **`nat`'s pre-dot head-fold** (3.2) is a *structural* difference from
  the `ref` (see below) → (a)-class for the indexer *score* (not for the
  selection on a short prompt).

### 3.2 The K-side key, the logit, top-k ordering + ties, the causal bound

| | behaviour |
|---|---|
| **us** | K-side: the index key is the **512-dim main latent projected** — `publish_entries` runs `gemm(lm, idx_wk, ik_)` with `lm` = the 512-dim main latent and `idx_wk = idx_comp_wkv` (`us:src/models/dsv4/model.cpp:662`, `us:src/models/dsv4/csa2_layer.cu:380`) — **not** the `ref`'s separate 128-wide indexer compressor on the *hidden* (the `ref`'s `Indexer.compressor` has `wkv [256,4096]`, `ref:model.py:394-399`). The per-entry logit = `Σ_h w_folded[h]*relu(dot_h)*k_scale` over the 64-head e4m3 dots (`us:src/models/dsv4/csa2_layer.cu:456-466`). The decode select (`dsv4_csa2_select_decode_kernel`, `csa2_layer.cu:658-735`) keeps a running top-512 on the composite key **`(~sortable << 21) | idx` with a MIN top-k → exact ties to the LOWER entry index** (`csa2_layer.cu:575-604`); causal bound `visible = (pos+1)/ratio` (`csa2_entry_positions`, `us:src/kernels/csa2.hpp:75-79`, `csa2_layer.cu:414,491,515`). |
| **nat** | K-side: the **132 B/token** indexer cache (128 e4m3 + ONE fp32 scale, the V3.2 layout) produced by the **indexer's own 128-wide compressor** (the `nat:src/ops/compressor/` with `head_dim=128, rotate=True`), `nat:src/ops/indexer_token/indexer_token.h:1-60,160-200`. The score = `q_scale[r]*k_scale[j]*Σ_d e4m3(q_fp8[r,d])*e4m3(k_code[j,d])` — a **single 128-dim dot, no relu** (`nat:src/ops/indexer_token/indexer_token.h:53-56,294`). Top-k: top-512 in **(score desc, index asc)** (the torch.topk stable tie-break), candidates = the first `valid_lens[r]` (the causal bound `(pos+1)//4`), `-1` padding (`nat:src/ops/indexer_token/indexer_token.h:58-70`). |
| **ref** | K-side: the **indexer's own 128-wide rotated compressor** (`ref:model.py:394-399`), the `kv_cache [bsz, max_seq//ratio, 128]`. Logit = `einsum("bshd,btd->bsht", q, kv)` → **`relu_()`** → `* weights` → `.sum(dim=2)` (`ref:model.py:426-427`). Top-k = `index_score.topk(min(index_topk, end_pos//ratio))[1]` (`ref:model.py:433`) — the causal bound `end_pos//ratio`, the `torch.topk` ordering. |

**Verdict:**
- **The K-side key source is a real (a) delta in ours**: we project the
  **512-dim main latent** through `idx_comp_wkv`; the `ref`/`nat` use a
  **separate 128-wide compressor on the hidden** (its own `wkv [256,4096]`).
  This changes *which* 128-dim vector is the index key. (It only feeds the
  selection score, so on a short prompt where `visible < topk` every entry
  is kept regardless of score — §6. But for a long context it is a real
  selection error.) **Also note the shape mismatch**: `idx_comp_wkv` is
  `[256, 4096]` (`us:src/models/dsv4/loader.hpp:87`) but the GEMM in
  `publish_entries` consumes it as `[128, 512]` (`us:src/models/dsv4/csa2_layer.cu:380`, `k` = `kCsa2Latent` 512) — it reads a *slice* of the real weight, not the intended
  projection. This is a latent (a) bug — and it is part of the known-incomplete
  C4A publish path (the spec's `G-tail-pool` / `G3` placeholder: the 1024 → 512
  plane reduction is explicitly "undefined, the GPU parity gate settles the
  plane choice", `us:src/models/dsv4/csa2_layer.cu:347-360`) — inert on a 15-token prompt.
- **The logit's relu**: the `ref` clamps each head's dot to ≥ 0 (`relu_`);
  **`nat` drops the relu** and folds the heads *before* the dot
  (`Σ_h w[h] q[h,d]` then one dot). These are algebraically equal **only if
  every per-head dot is ≥ 0**; otherwise they differ. **This is where
  `dsv4-native` contradicts the Python reference** (§7). Ours *keeps* the
  relu (matches the `ref`).
- **top-k ordering + ties**: all three resolve to **(score desc, index
  asc)** (the lower index wins exact ties) — `us` MIN-top-k on
  `(~sortable<<21)|idx` (`csa2_layer.cu:575-604`), `nat` "(score desc,
  index asc)", `ref` `torch.topk`. **Agreement (c).** (The spec doc's
  `G-tiebreak` is stale — the decode kernel is now the MIN/lower-index
  form.)
- **causal/visible bound**: all three use `visible = (pos+1)//ratio`
  (`us:csa2.hpp:75-79`, `nat` `(pos+1)//4`, `ref` `end_pos//ratio`).
  **Agreement (c).**

---

## 4. The compressor (the ratio-4 overlapping form)

| | behaviour |
|---|---|
| **us** | `dsv4_compress_tail_update` (`us:src/models/dsv4/compress.cu:84-138`): a **dsv41-form 2-token even-stash / odd-pool** — the even token STASHES its `(kv, score)` into the `[2, W]` tail, the odd token POOLS the stashed even with the current odd via `dsv4_pool_pair_and_norm` (a **2-entry per-dim softmax** over the full `W=1024`, `us:src/models/dsv4/compress.cu:49-75`), publishing an entry **every 2 tokens** (`entries_out = p/2`, `ent_pos = (p/2)*ratio`, `compress.cu:124-131`). **No APE parameter** (the caller must add `ape[p%ratio]` to the score plane, `us:src/models/dsv4/compress.hpp:35-39`). The 1024 → 512 reduction is a placeholder slice of the first 512 (the overlap plane) (`us:src/models/dsv4/csa2_layer.cu:347-400`). The C128A branch is a **plain per-token** `wkv` + one-rounding RMSNorm, **no gate/pool/APE/RoPE** (`us:src/models/dsv4/csa2_layer.cu:533-536`). |
| **nat** | `compressor_state_store_device` (EVERY token) + `compressor_compress_device` (the boundary) (`nat:src/ops/compressor/compressor.h:26-60`, `.cu:136-170`): the **fp32 state cache** `[blk, off, 0:W]=kv` (untouched), `[blk,off, W:2W]=score + ape[p%ratio]` (**APE on the score half only**); the boundary gate `(position+1)%ratio==0` (**publish every 4 tokens for C4A**); the **8-row × 512-dim pool over the overlap/normal plane split** (`head_off = (r>=ratio)?512:0`, `nat:…compressor.cu:146-170`, `start = position - 8 + 1`); the one-rounding RMSNorm; the GPT-J RoPE at the group start; the fp8 quant of the **NoPE 448 only** (the 584 B record). |
| **ref** | `Compressor.forward` (`ref:model.py:322-383`): the **fp32** `kv_state`/`score_state` `[b, 8, 1024]`; the decode `score += ape[start_pos%ratio]` (**APE on the score half only**, `ref:model.py:351`); the state write `kv_state[:, ratio + start_pos%ratio]` (`ref:model.py:353-354`); `should_compress = (start_pos+1)%ratio==0` (**every 4 tokens**, `ref:model.py:350`); the **8-entry × 512-dim pool** over the plane split `torch.cat([kv_state[:,:ratio,:d], kv_state[:,ratio:,d:]])` (`ref:model.py:356-357`); the one-rounding RMSNorm (`ref:model.py:368`); the RoPE at the group start (`ref:model.py:373`); the fp8 quant of the NoPE 448 (`ref:model.py:378`); the state roll (`ref:model.py:359-360`). |

**Verdict: (a) a real structural delta in ours (the `G-tail-*` gaps), NOT a
quant-class tolerance:**
- **Cadence**: we publish **every 2 tokens** (`odd` parity, `compress.cu:114`);
  `nat`/`ref` publish **every 4 tokens** (`(position+1)%4==0`). We publish
  **2× the entries** with the wrong ordinals/positions. (`G-tail-cadence`.)
- **The pool**: we do a **2-entry pair pool over the full 1024** (`compress.cu:49-75`);
  `nat`/`ref` do the **8-entry × 512 pool over the overlap/normal plane
  split** (`nat:compressor.cu:146-170`, `ref:model.py:356-357`). Different
  pooled value. (`G-tail-pool`.)
- **The APE**: we have **no APE** in the kernel; `nat`/`ref` add
  `ape[p%ratio]` to the **score half only** (`nat:compressor.h:37-40`,
  `ref:model.py:351`). (`G-tail-ape`.)
- **The C128A**: we do a **plain per-token** projection; `nat`/`ref` do a
  **ratio-128 gated pool** (`ref:model.py:362-365`). (`G-c128a-compressor`.)

On a 15-token prompt the C4A compressed cache holds only ~3–4 entries, so
these structural deltas move a *few* of ~18 attended keys — a **drift, not
the total degeneration** (§6). But they are real (a) bugs, not tolerances.

---

## 5. The MoE / mHC / lm head (only as far as they could globally degenerate)

| | behaviour |
|---|---|
| **us** | mHC: the shared GLM 4-copy mHC (`us:src/models/dsv4/model.cpp:199-215,837,908-925,940-943,1028-1030` — `hc_pre`/`hc_post` sites + the `hc_head` collapse before the final norm), oracle-qualified. MoE: the fused router `route` + the MXFP4 `expert` (`us:src/models/dsv4/model.cpp:1036-1039`), the hash/`tid2eid` layers via the table (`us:src/models/dsv4/binding.cpp:150-158`). lm head: the `globals_.lm_head` GEMM (F32 out) (`us:src/models/dsv4/model.cpp:839,1077`). |
| **nat** | mHC: `mhc_pre`/`mhc_post`/`mhc_head` (the 4-copy flatten-RMS + `[24,4h]` GEMM + the 20-iter Sinkhorn comb, `nat:src/ops/mhc/`, `docs/spec/mhc.md`). MoE: `router_fused_topk` (sqrtsoftplus + topk-bias + renorm ×1.5) + `fused_moe_mxfp4` (W4A16) + `moe_layer_tp2` (the TP=2 sharded block, one allreduce, `nat:src/ops/moe/`, `docs/spec/moe.md`). lm head: the vocab-parallel column GEMM (`nat:docs/spec/linear.md`). |
| **ref** | mHC: the `Block.hc_pre`/`hc_post`/`hc_head` + `hc_split_sinkhorn` (`ref:model.py:680-718`). MoE: the `Gate` (sqrtsoftplus, noaux_tc bias, renorm, ×1.5) + the `Expert`/`MoE` (top-6 routed + 1 shared, swiglu clamp) (`ref:model.py:551-651`). lm head: the `ParallelHead` (the 4-copy collapse + final norm + the vocab GEMM, `ref:model.py:719-743`). |

**Verdict: (c) agreement / no known global-degeneration delta.** All three
carry the same 4-copy learned-mixing mHC, the same top-6+1 MoE with the
sqrtsoftplus/renorm/×1.5 router, and the same lm head. Ours reuses the
shared, oracle-qualified GLM mHC/MoE kernels (not a V4-specific rewrite), so
there is no *V4-specific* mHC/MoE wiring that would globally degenerate the
output. **None of these is a plausible cause of the total degeneration** —
they are structurally present and match the reference. (If the degeneration
survives the q-path + G8 fixes, a per-layer mHC/MoE *numerics* drift would
be the next place to look, but it would be a small drift, not total.)

---

## 6. PRIORITISED list — what explains TOTAL degeneration on a 15-token prompt

On a 15-token prompt the **window ring dominates** (15 of ~18 attended keys
for a C4A layer, 15 of 15 for C128A/SWA) and the **compressed cache is
nearly empty** (C4A ≈ 3–4 entries, C128A/SWA 0). So the cause must live in
the **dominant window-ring path** or in something **global to every layer**.
Ranked by plausibility of *total* (not drift) degeneration:

1. **THE MISSING q-re-normalization (`G-q-renorm`) — the prime suspect.**
   - **Why total, not drift:** it is the only *global, systematic,
     non-quant-class* delta in the dominant path. It rescales **each head's
     q by a different factor** (the pre-renorm per-head RMS is O(2–4), so
     the renorm divides each head's q by ~2–4, **non-uniformly**). That is a
     head-varying **logit-scale distortion**, not a uniform temperature
     change — it sharpens some heads' attention to near-argmax while leaving
     others soft, collapsing the attention into a self-referential
     repetition loop. A uniform 2–4× scale would be a mild temperature
     shift; the *per-head non-uniformity* is what makes it degenerate.
   - **Evidence it is the cause (independent source):** `dsv4-native` lists
     the missing q-renorm as its **C6.4 rank-1** candidate and its
     `DSV4_QRENORM_DUMP` diagnostic (`nat:…model_forward_tp2.cpp:1158-1190`)
     measures the pre-renorm per-head RMS as **O(2–4)** — exactly the
     magnitude that would collapse attention. Adding it moved `nat`'s
     token-gate from 3749 → 11932 (the largest single fix).
   - **Caveat:** this is the *other agent's* gap. This report only
     corroborates it from `dsv4-native`. **If the q-path agent's fix does
     not stop the degeneration, the next suspect is #2.**

2. **The window-ring RoPE-64 quantization (`G8`, the `G8` first bullet) —
   the top *drift*, and the most plausible cause if #1 is fixed.**
   - We quantize the **full 512** (e4m3 + e8m0/32, the RoPE 64 included);
     `nat`/`ref` keep the **RoPE 64 in bf16** and quantize only the NoPE 448
     (per-64). The RoPE dims carry the **positional** signal the reference
     deliberately keeps exact; e4m3 (3-bit mantissa, ~6–12 % relative error)
     corrupts them. On a short prompt the window ring is the *dominant* KV
     source, so a ~10 % error on 64 of the 512 positional dims in **every**
     attended key is a large, systematic positional drift — the class of
     bug that turns coherent text into repetition. Classified (b) per the
     spec, but it is the single largest *numerical* delta in the dominant
     path and the first thing to try after the q-renorm.

3. **The DSpark raw-ring / compressed-pool not wired (`G-union-wiring`) —
   only if the MTP draft path is what is actually emitting the tokens.**
   The DSpark stages (layers 43–45) attend to **only the 5 in-memory block
   KVs** (`us:src/models/dsv4/model.cpp:1021` `union_attn(q_latent, nullptr, nullptr, 0,
   nullptr, 0, block_kv, …)`); the 128-slot main ring + the C4A compressed
   pool are still null. On a 15-token prompt the ring would hold the whole
   context; without it the *draft* is garbage. This does **not** break the
   main 43-layer path (the verify re-runs it), so it degrades MTP
   efficiency, not the final token — **unless** the deployment is emitting
   the draft tokens directly. Lower priority than #1/#2 for the *main*
   output.

4. **The compressor structural deltas (`G-tail-cadence` / `G-tail-pool` /
   `G-tail-ape` / `G-c128a`) and the indexer K-side key (§3.2) — real (a)
   bugs, but inert on a 15-token prompt.** The compressed cache is nearly
   empty and the indexer keeps *all* visible entries (`visible < topk`), so
   neither changes the output at 15 tokens. They will matter at longer
   context. **Fix them, but they are not the degeneration.**

**Bottom line:** the single difference that most plausibly explains *total*
degeneration on a 15-token prompt is **#1, the missing per-head q
re-normalization** (the other agent's `G-q-renorm`, corroborated by
`dsv4-native` as its rank-1 fix with an O(2–4) measured distortion). The
single most plausible *drift* that would remain after that is **#2, the
window-ring RoPE-64 quantization** (`G8`). Everything else (the two-source
merge, the sink, the tie-break, the causal bound, the mHC/MoE/lm head)
matches the reference and is not a degeneration source.

---

## 7. Where dsv4-native contradicts the Python reference (the most interesting signal)

These are the places where `dsv4-native` (which mirrors the *production*
vLLM DSpark fork) **disagrees with the checkpoint reference** — the
reference is the authority, so each is a candidate "the production deviates
from the shipped reference" flag:

1. **The indexer logit drops the `relu` and folds the heads before the dot
   (the biggest one).** `nat` computes
   `qsum[r,d] = (128^-0.5·64^-0.5)·Σ_h w[r,h]·q_rope[h,d]` then a **single
   128-dim dot with no relu** (`nat:src/ops/indexer_token/indexer_token.h:53-56,199-237`);
   the `ref` does a **per-head dot, `relu_()`, per-head weight, sum**
   (`ref:model.py:426-427`). Equal only when every per-head dot ≥ 0;
   otherwise the scores differ. **Ours keeps the relu** (matches the `ref`).
   If the production lane's selection is measured to differ from the
   reference's, this is the likely cause — but it only affects the
   *selection* (long context), not a short prompt.
2. **The indexer q-side is fp8 (V3.2 `amax/448`), not the `ref`'s fp4
   (e2m1 + e8m0/32).** `nat` re-expresses the `ref`'s fp4 as e4m3 + a fp32
   scale (`nat:src/ops/indexer_token/indexer_token.h:34-38`); a quant-class
   re-expression (the `ref`'s own comment: "We performed QAT here, kv could
   also use fp8").
3. **The KV record is a 584 B fp8-NoPE/bf16-RoPE envelope; the `ref` stores
   the compressed KV dequantized in bf16 in a single buffer.** `nat` keeps
   the 584 B record (a byte-identical-to-fp8 interior, the "true 4-bit"
   abandoned, `nat:docs/spec/kv-cache.md`); the `ref` uses the bf16
   `kv_cache` (`ref:model.py:378`). A quant-class delta, not a wiring one.
4. **The `wq_b` engine deviation (B4).** `nat` materializes the `wq_b`
   column shard to a full **replicated** plane at load (a spec deviation —
   the spec's `wq_b` is column-parallel 32 heads/rank); the B4
   de-replication is a perf flag, not a numerics change
   (`nat:docs/spec/linear.md` "the wq_b's engine's deviation").

**The signal to act on:** #1 (the dropped indexer relu + pre-dot fold) is
the only place `dsv4-native` changes the *math* (not just the quant class)
relative to the reference. It is inert on a 15-token prompt, so it is not
the degeneration — but if the long-context selection is later found to
diverge from the reference, this is the first thing to reconcile.

---

## Appendix — the file:line index (quick reference)

**RoPE:** `us` `model.cpp:300-303,665`, `csa2_layer.cu:304,308,321,384,395`,
`rope_scaling.cpp:17-49`, `csa2.hpp:63-70` · `nat` `rope.cpp:36-104`,
`model_forward_tp2.cpp:1215-1262` · `ref` `model.py:206-237,238-251,480-487,505,510`.
**q-renorm:** `us` (absent) `csa2_layer.cu:292-310` · `nat`
`model_forward_tp2.cpp:1156-1201` · `ref` `model.py:504,775-776`.
**Window ring:** `us` `csa2_layer.hpp:61`, `latent_format.hpp:44-64`,
`csa2.hpp:85-110`, `csa2_layer.cu:380-382` · `nat` `kv_cache.h:83-100`,
`model_forward_tp2.cpp:1367-1372`, `sparse_attn.h:146-210` · `ref`
`model.py:261-274,512`.
**Main cache:** `us` `latent_format.hpp:48-52`, `csa2_layer.cu:318,95-99` ·
`nat` `kv_cache.h:83-100` · `ref` `model.py:378,474,538`.
**Merge + sink:** `us` `csa2.hpp:238-245`, `csa2_layer.cu:401-445,181` ·
`nat` `sparse_attn.h`, `dspark_attn/` · `ref` `model.py:520,533,538`,
`kernel.py:346`.
**Indexer:** `us` `csa2_layer.cu:312-324,380,456-466,575-735`,
`csa2.hpp:131-153,75-79`, `model.cpp:662`, `loader.hpp:87` · `nat` `indexer_token.h:110-237,294`,
`indexer_token.cu:156-237` · `ref` `model.py:394-399,408-439`.
**Compressor:** `us` `compress.cu:49-138` (odd parity `:114`, entries `:128-129`), `csa2_layer.cu:347-400,511-536` ·
`nat` `compressor.h:26-60`, `compressor.cu:136-170` · `ref` `model.py:322-383`.
**MoE/mHC/lm:** `us` `model.cpp:199-215,837,908-943,1028-1039,839,1077`,
`binding.cpp:150-158` · `nat` `ops/mhc/`, `ops/moe/`, `spec/{mhc,moe,linear}.md`
· `ref` `model.py:551-651,680-743`.
