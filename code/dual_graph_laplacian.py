import inspect
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, cg

def laplacian(adjacency):
    a=csr_matrix(adjacency,dtype=np.float64)
    if not np.isfinite(a.data).all() or (a.data<0).any():
        raise ValueError('Adjacency must be finite and nonnegative')
    w=(a+a.T)*.5
    return (diags(np.asarray(w.sum(1)).ravel())-w).tocsr(),w

def complete(z,mask,ac,ad,ratio=1.0,tolerance=1e-8,maxiter=3000):
    if ratio<=0:raise ValueError('Both graph coefficients must be positive')
    z=np.asarray(z,dtype=np.float64);mask=np.asarray(mask,dtype=bool)
    lc,wc=laplacian(ac);ld,wd=laplacian(ad)
    if z.shape!=mask.shape or z.shape!=(lc.shape[0],ld.shape[0]):raise ValueError('Shape mismatch')
    nc,cc=connected_components(wc,directed=False);nd,cd=connected_components(wd,directed=False)
    product_labels=cc[:,None]*nd+cd[None,:]
    anchors=np.bincount(product_labels[mask],minlength=nc*nd)
    if (anchors==0).any():raise ValueError('Unanchored product-graph components: '+str(int((anchors==0).sum())))
    fixed=np.where(mask,z,0.0);free=np.flatnonzero(~mask.ravel())
    def energy_gradient(r):return ratio*(lc@r)+(ld@r.T).T
    if not len(free):return fixed,dict(iterations=0,relative_residual=0.0,product_components=int(nc*nd),anchor_counts=anchors.tolist())
    rhs=-energy_gradient(fixed).ravel()[free]
    def mv(x):
        r=np.zeros(z.shape,dtype=np.float64);r.ravel()[free]=x
        return energy_gradient(r).ravel()[free]
    diag=(ratio*lc.diagonal()[:,None]+ld.diagonal()[None,:]).ravel()[free]
    if (diag<=0).any():raise ValueError('Nonpositive free diagonal')
    op=LinearOperator((len(free),len(free)),matvec=mv,dtype=np.float64)
    pre=LinearOperator(op.shape,matvec=lambda x:x/diag,dtype=np.float64)
    iterations=[0]
    def callback(x):iterations[0]+=1
    kwargs=dict(M=pre,maxiter=maxiter,callback=callback,atol=0)
    kwargs['rtol' if 'rtol' in inspect.signature(cg).parameters else 'tol']=tolerance
    x,info=cg(op,rhs,**kwargs)
    residual=float(np.linalg.norm(mv(x)-rhs)/max(np.linalg.norm(rhs),1e-15))
    if info!=0 or not np.isfinite(x).all() or residual>max(10*tolerance,1e-7):raise RuntimeError('CG did not converge: '+str((info,residual)))
    out=fixed.copy();out.ravel()[free]=x
    assert np.array_equal(out[mask],z[mask])
    return out,dict(iterations=iterations[0],relative_residual=residual,product_components=int(nc*nd),anchor_counts=anchors.tolist(),unknowns=len(free),ratio=float(ratio))

def self_check():
    from scipy.sparse import kron,eye
    ac=np.array([[.2,.8,0],[.3,.2,.5],[0,.9,.1]])
    ad=np.array([[.1,.9],[.4,.6]])
    z=np.array([[1.,0.],[0.,-1.],[.5,0.]]);mask=np.array([[1,0],[0,1],[1,0]],bool)
    r,d=complete(z,mask,ac,ad,ratio=2)
    lc,_=laplacian(ac);ld,_=laplacian(ad)
    h=(2*kron(lc,eye(2))+kron(eye(3),ld)).toarray()
    u=np.flatnonzero(~mask.ravel());o=np.flatnonzero(mask.ravel())
    expected=np.linalg.solve(h[np.ix_(u,u)],-h[np.ix_(u,o)]@z.ravel()[o])
    assert np.allclose(r.ravel()[u],expected,atol=1e-8)
    try:complete(z,mask,np.eye(3),np.eye(2))
    except ValueError:pass
    else:raise AssertionError('Unanchored components not rejected')
    return dict(dense_solution_max_error=float(np.max(np.abs(r.ravel()[u]-expected))),unanchored_rejection=True)
