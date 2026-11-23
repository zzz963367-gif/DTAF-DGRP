import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import joblib
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
rma={str(k):np.asarray(v,float) for k,v in joblib.load(D/'RMA_dict.pkl').items()}
cells=pd.read_csv(A/'cohort.csv',dtype=str).Cell.tolist();assert len(cells)==618
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
module_rows=[]
for m in mirnas:
    targets=sorted(edges.loc[edges.miRNA.eq(m),'GeneSymbol'].unique())
    module_rows.append(dict(miRNA=m,target_count=len(targets),genes=';'.join(targets)))
modules=pd.DataFrame(module_rows);modules.to_csv(O/'module_definitions.csv',index=False)
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
