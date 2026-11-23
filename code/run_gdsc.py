import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'2')

ROOT=Path(__file__).resolve().parents[1]

def frames(fold):
    import pandas as pd
    from paths import DATA
    from build_modelad_bipartite_graph_diffusion_prior import pair_frame
    d=DATA/'Dataset'
    total=pair_frame(pd.read_csv(d/'lenient_split_fold_0_dataset.csv'))
    val=pair_frame(pd.read_csv(d/f'lenient_split_fold_{fold}_validation.csv'))
    test=pair_frame(pd.read_csv(d/'lenient_split_test.csv'))
    excluded=set(zip(val.Cell,val.Drug))|set(zip(test.Cell,test.Drug))
    train=total.loc[[(c,d) not in excluded for c,d in zip(total.Cell,total.Drug)]].reset_index(drop=True)
    result=dict(train=train,val=val,test=test)
    keys={s:set(zip(f.Cell,f.Drug)) for s,f in result.items()}
    for s,f in result.items():
        assert len(keys[s])==len(f),f'Duplicate pairs in {s}'
        assert f.IC50.notna().all()
    assert not keys['train']&keys['val'] and not keys['train']&keys['test'] and not keys['val']&keys['test']
    return result

def train(fold,cfg,out):
    env=os.environ.copy();env.update(cfg['environment'])
    from paths import DATA
    env.update(DTAF_FOLD=str(fold),DTAF_DATA_DIR=str(DATA),MODELADV_EXPERIMENT_NAME=f'FinalMLP_hybrid_fold{fold}',
       MODELADV_TOTAL_CSV=str(DATA/'Dataset/lenient_split_fold_0_dataset.csv'),MODELADV_VAL_CSV=str(DATA/f'Dataset/lenient_split_fold_{fold}_validation.csv'),MODELADV_TEST_CSV=str(DATA/'Dataset/lenient_split_test.csv'),MODELADV_TRAIN_CSV='',PYTHONPATH=str(ROOT/'code'))
    (out/'result').mkdir(parents=True,exist_ok=True)
    if any((out/'result').iterdir()):raise FileExistsError('Choose a new output directory; training will not overwrite results.')
    subprocess.run([sys.executable,'-u',str(ROOT/'code/Train.py')],cwd=out,env=env,check=True)

def dgrp(fold,cfg,out):
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.model_selection import KFold
    from paths import DATA
    import build_modelad_bipartite_graph_diffusion_prior as g
    from dual_graph_laplacian import complete
    from regulatory_similarity_caan import prepare,construct,load_similarity
    from calibration import fit_calibrator,apply_calibrator
    fs=frames(fold);ci,di=g.build_maps(fs)
    reg,available,paired=load_similarity(DATA/'regulatory/regulatory_features.npz',ci)
    def solve(training):
        _,z,m,mu,sd=g.build_matrix({'train':training},ci,di)
        c=cfg['graph'];sw=cfg['self_weight']
        ac=g.knn_graph_from_features(g.response_profile_features(z,m,1),c['k_cell'],c['temperature'],sw)
        ad=g.knn_graph_from_features(g.response_profile_features(z,m,0),c['k_drug'],c['temperature'],sw)
        ac=g.mix_graphs(ac,ac,1);ad=g.mix_graphs(ad,ad,1)
        ac=construct(prepare(z,m,reg,available),ac,c,sw,cfg['eta'])
        r,diagnostic=complete(z,m,ac,ad,cfg['ratio'])
        return r*sd[None,:]+mu[None,:],m,diagnostic
    def frame(f,p):
        result=f.rename(columns={'IC50':'real'}).copy();result['pred']=p;return result
    training=fs['train'];oof=np.full(len(training),np.nan);diagnostics=[]
    for inner,(keep,held) in enumerate(KFold(cfg['inner_folds'],shuffle=True,random_state=cfg['inner_seed']).split(training)):
        assert set(paired)<=set(training.iloc[keep].Cell)
        r,m,diag=solve(training.iloc[keep]);rr,dd,_=g.split_indices(training.iloc[held],ci,di)
        assert not m[rr,dd].any();oof[held]=r[rr,dd];diagnostics.append(diag)
        print('DGRP inner fold completed:',inner+1,flush=True)
    assert np.isfinite(oof).all()
    cal=fit_calibrator(frame(training,oof),**cfg['calibration'])
    r,m,diag=solve(training);diagnostics.append(diag)
    frame(training,oof).to_csv(out/'calibration_oof.csv',index=False)
    for s in ['val','test']:
        rr,dd,_=g.split_indices(fs[s],ci,di);assert not m[rr,dd].any()
        f=frame(fs[s],r[rr,dd]);f['pred']=apply_calibrator(f,cal);f.to_csv(out/f'dgrp_{s}.csv',index=False)
    joblib.dump(cal,out/'calibrator.pkl')
    (out/'solver_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))

def evaluate(fold,out):
    import numpy as np
    import pandas as pd
    import joblib
    from fusion import metrics,oof_predictions,Ridge,StandardScaler
    def read(p):return pd.read_csv(p,dtype={'Cell':str,'Drug':str}).rename(columns={'IC50':'real'})
    features={};neural={}
    for s in ['val','test']:
        neural[s]=read(out/f'result/FinalMLP_hybrid_fold{fold}_{s}_predictions.csv')
        prior=read(out/f'dgrp_{s}.csv')
        col='pred'
        n=neural[s];m=n[['Cell','Drug','real']].merge(prior[['Cell','Drug','real',col]],on=['Cell','Drug'],how='left',validate='one_to_one',suffixes=('','_prior'),sort=False)
        assert len(m)==len(prior) and np.allclose(m.real,m.real_prior,atol=2e-6,rtol=0)
        features[s]=np.column_stack([n.pred,m[col]])
        assert np.isfinite(features[s]).all()
    y=neural['val'].real.to_numpy(float)
    alphas=[.001,.003,.01,.03,.1,.3,1,3,10,30,100,300,1000,3000,10000]
    cv=[dict(alpha=a,mse=float(np.mean((oof_predictions(features['val'],y,a,5,14946934)-y)**2))) for a in alphas]
    alpha=min(cv,key=lambda x:x['mse'])['alpha']
    scaler=StandardScaler().fit(features['val']);ridge=Ridge(alpha=alpha).fit(scaler.transform(features['val']),y)
    pred=ridge.predict(scaler.transform(features['test']))
    report=dict(fold=fold,alpha=alpha,**metrics(neural['test'].real,pred))
    output=neural['test'][['Cell','Drug','real']].copy();output['prediction']=pred
    output.to_csv(out/'full_test.csv',index=False)
    pd.DataFrame(cv).to_csv(out/'fusion_cv.csv',index=False)
    joblib.dump(dict(scaler=scaler,ridge=ridge),out/'fusion.pkl')
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['check','train','dgrp','fuse'])
    p.add_argument('--fold',type=int,choices=range(5),default=0)
    p.add_argument('--output',type=Path)
    args=p.parse_args();fold=args.fold
    cfg=json.loads((ROOT/f'code/configs/fold{fold}.json').read_text())
    out=(args.output or ROOT/f'outputs/fold{fold}').resolve();out.mkdir(parents=True,exist_ok=True)
    if args.stage=='check':
        from dual_graph_laplacian import self_check
        print(json.dumps(dict(splits={s:len(f) for s,f in frames(fold).items()},solver=self_check()),indent=2))
    elif args.stage=='train':train(fold,cfg,out)
    elif args.stage=='dgrp':dgrp(fold,cfg,out)
    else:evaluate(fold,out)

if __name__=='__main__':main()
