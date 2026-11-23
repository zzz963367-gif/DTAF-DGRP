"""Fuse cell similarities before neighborhood selection, not graph adjacencies."""
import numpy as np
import build_modelad_bipartite_graph_diffusion_prior as graph

def prepare(z,mask,reg,available):
    features=graph.l2_normalize(graph.response_profile_features(z,mask,1).astype(np.float32))
    response=features@features.T
    inds=np.flatnonzero(available);block=np.ix_(inds,inds)
    off=~np.eye(len(inds),dtype=bool)
    sr=response[block];sg=reg[block]
    rm=float(sr[off].mean());rs=float(sr[off].std())
    gm=float(sg[off].mean());gs=float(sg[off].std())
    if rs<1e-12 or gs<1e-12:raise ValueError('Degenerate similarity distribution')
    mapped=np.clip((sg-gm)*(rs/gs)+rm,-1,1).astype(np.float32)
    stats=dict(response_mean=rm,response_std=rs,regulatory_mean=gm,regulatory_std=gs,
               mapped_mean=float(mapped[off].mean()),mapped_std=float(mapped[off].std()))
    return response,inds,mapped,stats

def construct(prepared,base,cfg,sw,eta):
    if not 0<=eta<=1:raise ValueError('Invalid mixture weight')
    if eta==0:return base.copy()
    response,inds,mapped,_=prepared
    sim=response.copy();block=np.ix_(inds,inds)
    sim[block]=(1-eta)*sim[block]+eta*mapped
    # All pairs lacking either expression profile keep their original similarity.
    np.fill_diagonal(sim,-np.inf)
    k=min(cfg['k_cell'],len(sim)-1)
    ix=np.argpartition(-sim,kth=k-1,axis=1)[:,:k];rows=np.arange(len(sim))[:,None]
    scores=np.maximum(sim[rows,ix],-.5)
    w=np.exp(scores/max(cfg['temperature'],1e-6));w/=np.maximum(w.sum(1,keepdims=True),1e-8)
    a=np.zeros_like(sim,dtype=np.float32);a[rows,ix]=w.astype(np.float32)
    if sw>0:a+=np.eye(len(a),dtype=np.float32)*sw
    a/=np.maximum(a.sum(1,keepdims=True),1e-8)
    a=graph.mix_graphs(a,a,1)
    available=np.zeros(len(a),bool);available[inds]=True
    assert np.array_equal(a[~available],base[~available])
    assert np.isfinite(a).all() and np.allclose(a.sum(1),1)
    return a

def load_similarity(path,ci):
    f=np.load(path);cells=f['cells'].astype(str);x=f['features'].astype(np.float32)
    assert x.shape==(618,573) and len(set(cells))==618 and np.isfinite(x).all()
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    inds=np.array([ci[c] for c in cells]);reg=np.zeros((len(ci),len(ci)),np.float32)
    reg[np.ix_(inds,inds)]=x@x.T
    available=np.zeros(len(ci),bool);available[inds]=True
    return reg,available,cells
