# C.1.0 — MXFP8 SF-fragment layout spec (sm_121a, `scale_vec::1X`)

Pinned by `tests/cuda/qwen_fp8_mxfp8_sf_test.cu` (the C.1.0 oracle in
`docs/qwen_fp8_phase_c_plan.md` §3). Instruction:

```
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
  {d0..d3}, {a0..a3}, {b0,b1}, {c0..c3}, {sfa}, {sfa_sel0,sfa_sel1}, {sfb}, {sfb_sel0,sfb_sel1};
```

Note the operand order (found by ptxas trial, matching the in-repo mxf4nvf4
4X form in `moe_w4a4.cu`): each scale register is followed by a **pair of
`.b16` selector registers**. All four selectors are 0 for `scale_vec::1X`.
The trailing-selectors order (`sfa, sfb, sel, sel`) is rejected by ptxas with
"Arguments mismatch".

## The layout (the governed outcome)

One mma covers one 32-wide k-block, so it carries **one ue8m0 scale per A row
(16) and one per B column (8)**. Lane = `4*g + t` (`g = lane/4`, `t = lane%4`):

| fragment | register | byte | supplies the scale for |
|---|---|---|---|
| SFA (1 .b32) | `sfa` | byte 0 (`& 0xFF`) | A row `g`, this k-block |
| SFA (1 .b32) | `sfa` | byte 1 (`>> 8 & 0xFF`) | A row `g+8`, this k-block |
| SFB (1 .b32) | `sfb` | byte 0 (`& 0xFF`) | B col `2t`, this k-block |
| SFB (1 .b32) | `sfb` | byte 1 (`>> 8 & 0xFF`) | B col `2t+1`, this k-block |

- The 4 threads of a group (`t = 0..3`) carry the **same two SFA bytes**
  (redundant); SFB bytes are partitioned (2 cols per thread × 4 threads = 8).
- Selectors: `sfa_sel = sfb_sel = 0` (`.b16`) for `scale_vec::1X`.
- A/B/C data-fragment packing is the plain e4m3 m16n8k32 layout (pinned by
  `qwen_fp8_mma_a_fragment_oracle`); the C fragment is
  `c0..c3 = D[g][2t], D[g][2t+1], D[g+8][2t], D[g+8][2t+1]`.
- Semantics: `D[m][n] += 2^(sfa[m]-127) · 2^(sfb[n]-127) · Σ_k a[m][k]·b[k][n]`
  (the pair product applies per k-block; verified multiplicatively by the
  all-ones probe: 0x7f/0x7f → 32.0 per element, 0x80/0x80 → 128.0).

## Per-k-step update rule (the 256-dim, 8-scales-per-row design)

The 256-dim head reduction is 8 sequential m16n8k32 k-steps into the same C
accumulator. At step `ks` (k-block `ks·32..ks·32+31`):

- `sfa` reg = `sfa_row[ks][g] | sfa_row[ks][g+8] << 8` — **8 distinct scales
  per A row are fully supported**: each k-step carries its own SFA/SFB
  registers, so no k-step shares a byte and no two rows share a scale.
- `sfb` reg = `sfb_col[ks][2t] | sfb_col[ks][2t+1] << 8`.
- `sfa_sel = sfb_sel = 0` every step.

Consequence for C.1a: Q (A) gets 8 E8M0 scales per 256-dim row and K (B) 8 per
row, exactly as the plan wants; the `α·βᵏ` rank-1 epilogue is deleted.

## Verified vs inferred

- **Verified (this box, sm_121a)**: the instruction compiles and runs
  (all-ones probe 132.0/516.0 with C=1, i.e. 32.0/128.0 per element at C=0);
  rejected on plain `sm_121` (the `121a` arch flag stays).
- **Verified (by the test, once run in the GPU window)**: the lane/byte map
  above against the exact host oracle with 16/8 *distinct* scale bytes
  (max diff 0), plus sensitivity — byte-swapped SFA/SFB and uniform-A
  mappings must all FAIL the oracle (if uniform-A passed, the A side would be
  uniform across the 8-row pair and the per-row design is a NO-GO with the
  fallback of K-side-only block scales).
- **Verified (ptxas trial on this box)**: the exact operand order — the
  selectors are *pairs* of `.b16` registers interleaved after each scale
  register (`{sfa} {sel,sel} {sfb} {sel,sel}`), matching the in-repo mxf4nvf4
  4X form in `moe_w4a4.cu`. The trailing-selectors order (`sfa, sfb, sel, sel`)
  is rejected with "Arguments mismatch". The SASS emits `QMMA.SF.16832.F32.E4M3.E4M3.E8`
  (a real block-scaled MMA, not the BRA-self-loop stub the plan warns about).
- **Inferred (PTX ISA 9.7.14.6.1, `.kind::mxf8f6f4` `.scale_vec::1X`
  fragment table, not re-derived here)**: the specific byte positions in the
  table above and the zero-selector rule. The test is the arbiter; if it
  fails, the table — not the test — is wrong.

## Status

Test written and compiling (`qwen_fp8_mxfp8_sf_test`); **not yet run** —
needs a GPU window: `ctest --test-dir build-ci -R qwen_fp8_mxfp8_sf` (or the
binary `build-ci/qwen_fp8_mxfp8_sf_test`).
