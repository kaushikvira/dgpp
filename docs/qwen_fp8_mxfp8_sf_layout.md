# C.1.0 — MXFP8 SF-fragment layout spec (sm_121a, `scale_vec::1X`)

Pinned by `tests/cuda/qwen_fp8_mxfp8_sf_test.cu` (the C.1.0 oracle in
`docs/qwen_fp8_phase_c_plan.md` §3). Instruction:

```
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
  {d0..d3}, {a0..a3}, {b0,b1}, {c0..c3}, {sfa}, {sfa_sel0,sfa_sel1}, {sfb}, {sfb_sel0,sfb_sel1};
```

Note the operand order (found by ptxas trial, matching the in-repo mxf4nvf4
4X form in `moe_w4a4.cu`): each scale register is followed by a **pair of
`.b16` selector registers**. All four selectors are 0 for `scale_vec::1X`
(byte 0 of each register is the active scale byte). The trailing-selectors
order (`sfa, sfb, sel, sel`) is rejected by ptxas with "Arguments mismatch".

## The layout (the governed outcome) — PROBE-VERIFIED

One mma covers one 32-wide k-block, so it carries **one ue8m0 scale per A row
(16) and one per B column (8)**. Lane = `4*g + t` (`g = lane/4`, `t = lane%4`).
The map below was **established empirically with a map-independent
single-byte-boost probe on this box** (see "Probe evidence"), *not* read off
the PTX ISA table — the table's layout was falsified by the oracle (see
"Changelog").

| fragment | register | byte | lane (thread) | supplies the scale for |
|---|---|---|---|---|
| SFA (1 .b32) | `sfa` | byte 0 | `4g+0` (t=0) | A row `g`, this k-block |
| SFA (1 .b32) | `sfa` | byte 0 | `4g+1` (t=1) | A row `g+8`, this k-block |
| SFB (1 .b32) | `sfb` | byte 0 | `4g+0` (t=0) | B col `g`, this k-block |

- **A side is PER-ROW**: row `g` and row `g+8` live in *different* threads of
  the same group (t=0 and t=1), so they hold **distinct** scales. Threads
  t=2, t=3 of a group carry no A scale (their `sfa` register is ignored).
- **B side is PER-COLUMN**: col `g`'s scale is in that column's group t=0
  thread, byte 0. Threads t=1, t=2, t=3 of a group carry no B scale (their
  `sfb` register is ignored).
- **Bytes 1–3 of every register are ignored at `sel=0`** (fill with `0x7f` =
  2^0, neutral). The `.b16` selectors can select bytes 1–3, but `scale_vec::1X`
  uses byte 0, so `sfa_sel = sfb_sel = 0` every step.
- A/B/C data-fragment packing is the plain e4m3 m16n8k32 layout (pinned by
  `qwen_fp8_mma_a_fragment_oracle`); the C fragment is
  `c0..c3 = D[g][2t], D[g][2t+1], D[g+8][2t], D[g+8][2t+1]`.
- Semantics: `D[m][n] += 2^(sfa[m]-127) · 2^(sfb[n]-127) · Σ_k a[m][k]·b[k][n]`
  (the pair product applies per k-block; verified multiplicatively by the
  all-ones probe: 0x7f/0x7f → 32.0 per element, 0x80/0x80 → 128.0).

## Probe evidence (map-independent, this box, sm_121a)

Method: A codes all `0x38` (1.0), B codes all `0x38` (1.0), all SF bytes
`0x7f` (2^0) → every `D[m][n] = 32`. Boost **exactly one** SF byte to `0x84`
(ue8m0 2^5) in a single thread's register and read all 128 outputs; the
boosted rows/cols read `32·32 = 1024`. This localizes each byte to a
row/col independent of any assumed map.

Single-byte-boost localization (byte0 → `0x84`; "no change" = all 128 cells
stay 32):

```
SFA thread (g=0,t=0) byte0 -> D[0..15][0..7] row 0 only   = 1024   (row g   <- thread(g,0) byte0)
SFA thread (g=0,t=1) byte0 -> row 8 only                   = 1024   (row g+8 <- thread(g,1) byte0)
SFA thread (g=0,t=2) byte0 -> no change
SFA thread (g=0,t=3) byte0 -> no change
SFB thread (g=0,t=0) byte0 -> col 0 (all 16 rows)          = 1024   (col g   <- thread(g,0) byte0)
SFB thread (g=3,t=0) byte0 -> col 3 (all 16 rows)          = 1024
SFB thread (g=0,t=1) byte0 -> no change
SFB thread (g=0,t=2) byte0 -> no change
SFB thread (g=0,t=3) byte0 -> no change
```

Cross-check with **distinct** per-row / per-col scales (A row `m` → 2^(m%5),
B col `n` → 2^(3+n%4), `sel=0`), the exact host oracle
`D[m][n] = 32 · 2^(sfa[m]-127) · 2^(sfb[n]-127)`:

```
TRUE-map distinct-scale oracle: max |mma - oracle| = 0
  D[0][0]=256  D[0][1]=512  D[0][3]=2048  D[0][7]=2048
  D[1][0]=512  D[1][1]=1024 D[1][3]=4096  D[1][7]=4096
  D[8][0]=2048 D[8][1]=4096 D[8][3]=16384 D[8][7]=16384
  D[9][0]=4096 D[9][1]=8192 D[9][3]=32768 D[9][7]=32768
  D[15][0]=256 D[15][1]=512 D[15][3]=2048 D[15][7]=2048
```

The per-row (rows 0,1,8,9,15 scale independently) and per-col (cols 0,1,3,7
scale independently) structure is unambiguous, and the oracle is bitwise
exact (max diff 0).

## Per-k-step update rule (the 256-dim, 8-scales-per-row design)

The 256-dim head reduction is 8 sequential m16n8k32 k-steps into the same C
accumulator. At step `ks` (k-block `ks·32..ks·32+31`):

- `sfa` reg of thread `(g,0)` = `sfa_row[ks][g]` (byte 0); thread `(g,1)` =
  `sfa_row[ks][g+8]`; threads `(g,2),(g,3)` = `0x7f` (unused).
- `sfb` reg of thread `(g,0)` = `sfb_col[ks][g]` (byte 0); threads
  `(g,1),(g,2),(g,3)` = `0x7f` (unused).
- `sfa_sel = sfb_sel = 0` every step.

Consequence for C.1a: Q (A) gets 8 E8M0 scales per 256-dim row and K (B) 8 per
row, exactly as the plan wants; the `α·βᵏ` rank-1 epilogue is deleted. The
packing is **per-thread byte 0** (not the same-thread two-byte / two-col-per-
thread form the old doc described) — see the changelog.

## Verified vs inferred

- **Verified (this box, sm_121a)**: the instruction compiles and runs
  (all-ones probe 32.0/128.0 per element at C=0; the plan's 132.0/516.0 were
  with C=1); rejected on plain `sm_121` (the `121a` arch flag stays).
- **Verified (probe, this box)**: the lane/byte map in the table above, by the
  map-independent single-byte-boost localization plus the distinct-scale
  oracle (max diff 0). The map is credited to the probe, **not** to the PTX
  ISA table.
- **Verified (by the test, run in the GPU window)**: the true map against the
  exact host oracle with 9/7 *distinct* scale bytes (max diff 0), plus
  sensitivity *relative to the true map* — sfa row-swap, sfb col-misplace,
  uniform-A, and the **falsified old-spec map** all FAIL (non-zero). If
  uniform-A had passed, the A side would be uniform across the 8-row pair and
  the per-row design a NO-GO (it does not pass).
- **Verified (ptxas trial on this box)**: the exact operand order — the
  selectors are *pairs* of `.b16` registers interleaved after each scale
  register (`{sfa} {sel,sel} {sfb} {sel,sel}`), matching the in-repo mxf4nvf4
  4X form in `moe_w4a4.cu`. The trailing-selectors order is rejected with
  "Arguments mismatch". The SASS emits `QMMA.SF.16832.F32.E4M3.E4M3.E8`
  (a real block-scaled MMA, not the BRA-self-loop stub the plan warns about).
- **Not re-derived / out of scope**: the exact semantics of *non-zero*
  selector values (which byte they select on the B side in particular was
  irregular in the sweep). The kernel only ever uses `sel=0` (byte 0), which
  is fully pinned by the probe; non-zero selectors are not used by C.1a.

## Changelog — what the old doc claimed vs what the hardware does

The previous revision of this doc (and the first revision of the test)
asserted the **PTX ISA 9.7.14.6.1** `.kind::mxf8f6f4` `.scale_vec::1X`
fragment table: SFA `byte0 = row g`, `byte1 = row g+8` **in the same thread's
register** (all 4 threads of a group carrying the same two bytes), and SFB
`byte0 = col 2t`, `byte1 = col 2t+1` (two columns per thread, partitioned
across the 4 threads). That mapping was **falsified by running the oracle on
the GPU** (max `|mma - oracle| = 6.06e+04`, non-zero). The probe-verified
hardware behaviour is different in both fragments:

- **SFA**: row `g` is in thread `(g,0)` byte 0 and row `g+8` is in thread
  `(g,1)` byte 0 — *different* threads, so the two rows hold *distinct*
  scales (per-row). The old doc's same-thread `byte0/byte1` pairing is wrong;
  bytes 1–3 are ignored at `sel=0`.
- **SFB**: col `g` is in thread `(g,0)` byte 0 — *one* column per group, not
  two columns per thread. The old doc's `col 2t / col 2t+1` two-col-per-thread
  partition is wrong.

**Impact on C.1a (`src/kernels/qsa_warp.cu`)**: the SF-register packing must
change from the same-thread two-byte SFA (`byte0=g, byte1=g+8`) + two-col-per-
thread SFB (`byte0=2t, byte1=2t+1`) to the **per-thread byte-0** form:
thread `(g,0)` `sfa` byte0 = row `g`, thread `(g,1)` `sfa` byte0 = row `g+8`,
thread `(g,0)` `sfb` byte0 = col `g`, all other threads' SF registers and all
bytes 1–3 = `0x7f`. (This file's test `map 4` encodes the old-spec packing as
a regression case that must FAIL.)

## Status

Test written, compiling, and **run in the GPU window**: true map → max diff 0;
sensitivity (sfa-swap / sfb-swap / sfa-uniform / old-spec) all non-zero.
`ctest --test-dir build-ci -R qwen_fp8_mxfp8_sf` (or the binary
`build-ci/qwen_fp8_mxfp8_sf_test`).
