"""Evaluate only the requested miRNA/target-regulatory_residual CAAN, without new controls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix,save_npz
from threadpoolctl import threadpool_limits
threadpool_limits(2)
from paths import DATA,ROOT
R=ROOT;D=DATA/"Dataset";A=DATA/"regulatory"
O=ROOT/"outputs/regulatory";O.mkdir(parents=True,exist_ok=True)
from normalize_mirna import normalize_mirna

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def standardize(x):
    mu=x.mean(0);sd=x.std(0);sd=np.where(sd>1e-8,sd,1)
    return (x-mu)/sd,mu,sd
def knn(sim,k,t):
    sim=sim.copy();np.fill_diagonal(sim,-np.inf)
    ix=np.argsort(-sim,axis=1,kind='stable')[:,:k]
    scores=np.maximum(np.take_along_axis(sim,ix,axis=1),-.5)/t
    w=np.exp(scores-scores.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
    g=csr_matrix((w.ravel(),(np.repeat(np.arange(len(sim)),k),ix.ravel())),shape=sim.shape)
    assert np.allclose(g.sum(1),1) and np.all(g.diagonal()==0)
    return g
def metrics(y,p,drugs):
    vals=[]
    for d in np.unique(drugs):
        mask=drugs==d
        if mask.sum()>=10 and np.std(y[mask])>1e-10 and np.std(p[mask])>1e-10:
            vals.append(float(np.corrcoef(y[mask],p[mask])[0,1]))
    return dict(pcc=float(np.corrcoef(y,p)[0,1]),rmse=float(np.sqrt(np.mean((y-p)**2))),
                drug_macro_pcc=float(np.mean(vals)),drug_macro_n=len(vals))

total=pd.read_csv(D/'lenient_split_fold_0_dataset.csv',dtype={'Cell':str})
test=pd.read_csv(D/'lenient_split_test.csv',usecols=['Cell','Drug'],dtype=str)
test_pairs=set(zip(test.Cell,test.Drug))
rma={str(k):np.asarray(v,float) for k,v in joblib.load(D/'RMA_dict.pkl').items()}
cells=pd.read_csv(A/'cohort.csv',dtype=str).Cell.tolist();assert len(cells)==618
ci={c:i for i,c in enumerate(cells)};drugs=sorted(total.Drug.unique());di={d:i for i,d in enumerate(drugs)}
mapping=pd.read_csv(A/'mapping_recheck.csv',dtype=str).fillna('')
sample_for={r.old_cosmic:r.GCT_sample for _,r in mapping.iterrows() if r.old_cosmic}
gct=pd.read_csv(D/'CCLE_miRNA_20181103.gct',sep='\t',skiprows=2)
gct.index=gct.Description.map(normalize_mirna)
edges=pd.read_csv(A/'human_strong_candidate_pairs.csv').drop_duplicates()
mirnas=sorted(edges.miRNA.unique());assert len(mirnas)==220 and len(edges)==947
genes=(A/'gene_list_sel.txt').read_text().splitlines();gi={g:i for i,g in enumerate(genes)}
X=np.stack([gct.loc[mirnas,sample_for[c]].to_numpy(float) for c in cells])
Y=np.stack([rma[c] for c in cells])
assert np.isfinite(X).all() and np.isfinite(Y).all()
xz,xmu,xsd=standardize(X);yz,ymu,ysd=standardize(Y)
module_rows=[];target=np.zeros((len(cells),len(mirnas)))
for j,m in enumerate(mirnas):
    targets=sorted(edges.loc[edges.miRNA.eq(m),'GeneSymbol'].unique())
    target[:,j]=yz[:,[gi[g] for g in targets]].mean(1)
    module_rows.append(dict(miRNA=m,target_count=len(targets),genes=';'.join(targets)))
modules=pd.DataFrame(module_rows);modules.to_csv(O/'module_definitions.csv',index=False)
# Gene-wise ridge models restricted to curated miRNA regulators.
from scipy.stats import rankdata
rx,_,_=standardize(rankdata(X,axis=0))
ry,_,_=standardize(rankdata(Y,axis=0))
mi={m:i for i,m in enumerate(mirnas)}
target_genes=sorted(edges.GeneSymbol.unique())
residual=[];fit_records=[]
alpha=10.0
for gene in target_genes:
    regulators=sorted(edges.loc[edges.GeneSymbol.eq(gene),'miRNA'].unique())
    a=rx[:,[mi[m] for m in regulators]]
    y=ry[:,gi[gene]]
    inv=np.linalg.inv(a.T@a+alpha*np.eye(a.shape[1]))
    coef=inv@a.T@y
    fitted=a@coef
    leverage=np.sum((a@inv)*a,axis=1)+1.0/len(cells)
    # Exact leave-one-cell-out residual for ridge with an unpenalized intercept,
    # conditional on the fixed training-reference rank transformation.
    loo=(y-fitted)/np.maximum(1-leverage,1e-8)
    residual.append(loo)
    fit_records.append(dict(gene=gene,regulators=len(regulators),
        loo_mse=float(np.mean(loo**2)),positive_coefficients=int((coef>0).sum()),
        negative_coefficients=int((coef<0).sum())))
features=unit(standardize(np.stack(residual,axis=1))[0])
pd.DataFrame(fit_records).to_csv(O/'regulatory_fit_diagnostics.csv',index=False)
assert features.shape==(618,573) and np.isfinite(features).all()
np.savez_compressed(O/"regulatory_features.npz",cells=np.array(cells),features=features)
reference=np.load(A/"regulatory_features.npz")
assert np.array_equal(reference["cells"],np.array(cells))
delta=float(np.max(np.abs(reference["features"]-features)))
assert delta<1e-7,delta
print("Residual features reproduced; maximum absolute difference:",delta)
