import json, numpy as np
idx = json.load(open('st_index.json'))
def load(name):
    f, off, nb, dt, shape = idx[name]
    return np.memmap(f, dtype=np.uint16, mode='r', offset=off, shape=tuple(shape))
def row_escapes(a):
    e = ((np.asarray(a) >> 7) & 0xFF).astype(np.int64)
    n, k = e.shape
    out = np.zeros(n, dtype=np.int64)
    # histogram per row via bincount on offset codes
    h = np.zeros((n, 256), dtype=np.int64)
    rows = np.repeat(np.arange(n), k)
    np.add.at(h, (rows, e.ravel()), 1)
    cs = np.concatenate([np.zeros((n,1),dtype=np.int64), np.cumsum(h, axis=1)], axis=1)
    win = cs[:, 15:] - cs[:, :-15]
    return k - win.max(axis=1)
bad_layers = []
for L in range(45):
    p = f'model.language_model.layers.{L}.self_attn.'
    if p + 'q_proj.weight' not in idx: continue
    rep = {}
    for nm, sl in [('f_a_proj', slice(None)), ('g_a_proj', slice(None)), ('q_proj', slice(0, 2048)), ('k_proj', slice(0, 2048)), ('v_proj', slice(0, 2048)), ('b_proj', slice(0, 16))]:
        a = load(p + nm + '.weight')[sl]
        r = row_escapes(a)
        rep[nm] = (int(r.max()), int((r > 64).sum()), int(r.sum()))
    worst = max(v[0] for v in rep.values())
    if worst > 64:
        bad_layers.append(L)
        print(L, {k: v for k, v in rep.items() if v[0] > 64})
print('layers over the bound:', bad_layers)
