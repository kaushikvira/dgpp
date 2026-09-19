import json, numpy as np, sys, math
idx = json.load(open('st_index.json'))  # built by the snippet in README.md
def load(name):
    f, off, nb, dt, shape = idx[name]
    assert dt == 'BF16'
    a = np.memmap(f, dtype=np.uint16, mode='r', offset=off, shape=tuple(shape))
    return a
def stats(name, tp_rows=None):
    a = np.asarray(load(name))
    n, k = a.shape[0], int(np.prod(a.shape[1:]))
    a = a.reshape(n, k)
    e = ((a >> 7) & 0xFF).astype(np.uint8)
    hist = np.bincount(e.ravel(), minlength=256).astype(np.float64)
    tot = hist.sum()
    p = hist[hist > 0] / tot
    H = -(p * np.log2(p)).sum()
    nz = np.nonzero(hist)[0]
    order = np.argsort(-hist)
    top15 = hist[order[:15]].sum() / tot
    top16 = hist[order[:16]].sum() / tot
    # contiguous 16-window per tensor (best)
    cs = np.concatenate([[0], np.cumsum(hist)])
    bestwin = max((cs[i+16] - cs[i], i) for i in range(0, 256-16))
    # per-row window: [rowmax-15, rowmax] (codes 0..15), escapes = below window
    rmax = e.max(axis=1, keepdims=True)
    esc_row16 = (e < (rmax.astype(np.int16) - 15)).sum(axis=1)
    esc_row15 = (e < (rmax.astype(np.int16) - 14)).sum(axis=1)
    # per-tensor window anchored at tensor max
    tmax = int(e.max())
    esc_t16 = (e < tmax - 15).sum()
    above = 0
    # per 16-element chunk windows (chunk max anchored)
    kk = (k // 16) * 16
    ec = e[:, :kk].reshape(n, kk // 16, 16)
    cmax = ec.max(axis=2, keepdims=True).astype(np.int16)
    esc_chunk16 = (ec < cmax - 15).sum()
    esc_chunk8 = (ec < cmax - 7).sum()
    zeros = ((a & 0x7FFF) == 0).sum()
    print(f"{name}\n  shape {a.shape} distinct exps {len(nz)} range [{nz.min()},{nz.max()}] entropy {H:.3f} bits  zeros {zeros}")
    print(f"  top15 {top15:.6f} top16 {top16:.6f} best contiguous16 {bestwin[0]/tot:.6f} (start {bestwin[1]})")
    print(f"  tensor-anchored 16-window escapes {esc_t16} ({esc_t16/tot:.2e})")
    print(f"  row-anchored 16-window: escapes {esc_row16.sum()} ({esc_row16.sum()/tot:.2e}), rows with escapes {np.count_nonzero(esc_row16)}/{n} ({np.count_nonzero(esc_row16)/n:.3f})")
    print(f"  row-anchored 15-window: escapes {esc_row15.sum()} ({esc_row15.sum()/tot:.2e}), rows with escapes {np.count_nonzero(esc_row15)}/{n} ({np.count_nonzero(esc_row15)/n:.3f})")
    print(f"  chunk16-anchored 16-window escapes {esc_chunk16} ({esc_chunk16/tot:.2e}); 8-window (3 bits) escapes {esc_chunk8/tot:.2e}")
    print("  hist:", {int(i): int(hist[i]) for i in nz})
L = 'model.language_model.layers.'
names = [L+'0.self_attn.q_proj.weight', L+'0.self_attn.o_proj.weight', L+'17.self_attn.k_proj.weight', L+'17.self_attn.v_proj.weight', L+'40.self_attn.o_proj.weight', 'lm_head.weight']
for nm in names:
    if nm in idx: stats(nm)
    else: print('missing', nm)
