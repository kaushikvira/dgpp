import json, numpy as np
idx=json.load(open('st_index.json'))
def load(name):
    f,off,nb,dt,shape=idx[name]
    return np.memmap(f,dtype=np.uint16,mode='r',offset=off,shape=tuple(shape))
L='model.language_model.layers.0.'
out={
 'qkv_2560x2560': np.concatenate([load(L+'linear_attn.in_proj_qkv.weight')[0:512], load(L+'linear_attn.in_proj_qkv.weight')[2048:2560], load(L+'linear_attn.in_proj_qkv.weight')[4096:5632]]),
 'out_2560x1536': np.ascontiguousarray(load(L+'linear_attn.out_proj.weight')[:, 0:1536]),
 'grdown_320x10240': np.asarray(load(L+'attn_hyper_connection.input_mix_weight_down.weight')),
 'grup_10240x320': np.asarray(load(L+'attn_hyper_connection.input_mix_weight_up.weight')),
 'head_62080x2560': np.asarray(load('lm_head.weight')[0:62080]),
}
for k,v in out.items():
    v=np.ascontiguousarray(v); v.tofile(f'w_{k}.bin'); print(k, v.shape, v.nbytes/1e6, 'MB')
