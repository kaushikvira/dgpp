import json, struct, glob, os, sys, collections
snap = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--HawkBearPig--GLM-5.3-Flash-NVFP4-FP8/snapshots/*/'))[0]
by_dtype = collections.Counter(); bytes_by = collections.Counter()
idx = {}
for f in sorted(glob.glob(snap + '*.safetensors')):
    with open(f, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for k, v in hdr.items():
        if k == '__metadata__': continue
        idx[k] = (f, 8 + n + v['data_offsets'][0], v['data_offsets'][1] - v['data_offsets'][0], v['dtype'], v['shape'])
json.dump(idx, open('st_index.json', 'w'))
# summarize BF16 tensors by name pattern (layer index stripped)
import re
pat = collections.defaultdict(lambda: [0, 0, None])
for k, (f, off, nb, dt, shape) in idx.items():
    key = re.sub(r'\.\d+\.', '.N.', k)
    key = re.sub(r'experts\.N\.', 'experts.E.', key)
    p = pat[(key, dt)]
    p[0] += 1; p[1] += nb; p[2] = shape
rows = sorted(pat.items(), key=lambda kv: -kv[1][1])
for (key, dt), (cnt, nb, shape) in rows[:70]:
    print(f"{nb/2**30:9.3f} GiB  x{cnt:6d}  {dt:8s} {str(shape):22s} {key}")
