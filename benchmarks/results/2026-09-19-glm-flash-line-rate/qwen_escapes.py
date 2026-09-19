import json, numpy as np
idx=json.load(open('st_index.json'))
def load(name):
    f,off,nb,dt,shape=idx[name]
    return np.memmap(f,dtype=np.uint16,mode='r',offset=off,shape=tuple(shape))
def stats(a):
    e=((np.asarray(a)>>7)&0xFF).astype(np.int64); n,k=e.shape
    h=np.zeros((n,256),dtype=np.int64); np.add.at(h,(np.repeat(np.arange(n),k),e.ravel()),1)
    cs=np.concatenate([np.zeros((n,1),dtype=np.int64),np.cumsum(h,axis=1)],axis=1)
    esc=k-(cs[:,15:]-cs[:,:-15]).max(axis=1)
    raw=int((esc>64).sum()); packed_esc=int(esc[esc<=64].sum())
    return n,k,packed_esc/(n*k),raw,int(esc.max())
L='model.language_model.layers.'
for l in (0,1,2,3,23,47):
    for nm,sl in [('linear_attn.in_proj_qkv.weight',None),('linear_attn.in_proj_z.weight',None),('linear_attn.out_proj.weight',(slice(None),slice(0,1536))),
                  ('self_attn.q_proj.weight',(slice(0,3072),slice(None))),('self_attn.o_proj.weight',(slice(None),slice(0,1536))),
                  ('attn_hyper_connection.input_mix_weight_down.weight',None),('attn_hyper_connection.input_mix_weight_up.weight',None),
                  ('mlp_hyper_connection.input_mix_weight_up.weight',None),
                  ('mlp.shared_expert.gate_proj.weight',(slice(0,160),slice(None))),('mlp.shared_expert.down_proj.weight',(slice(None),slice(0,160))),
                  ('mlp.gate.weight',None)]:
        name=f'{L}{l}.{nm}'
        if name not in idx or (l in (1,2,3) and 'hyper' not in nm and 'self_attn' not in nm): continue
        a=load(name); a=a if sl is None else a[sl]
        n,k,fr,raw,mx=stats(a)
        print(f'layer {l:2d} {nm:52s} [{n:6d} x {k:5d}] escapes {fr*1e4:6.2f} per 10k, raw rows {raw:5d} ({raw/n*100:5.2f} %), widest {mx}', flush=True)
a=load('lm_head.weight')[0:62080]
n,k,fr,raw,mx=stats(a); print(f'lm_head slice [{n} x {k}] escapes {fr*1e4:.2f} per 10k, raw rows {raw} ({raw/n*100:.2f} %), widest {mx}')
