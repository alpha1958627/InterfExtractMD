"""Select actual density roots; never interpolate missing surface heights."""
import numpy as np

def density_roots(rho,dx,threshold,z_origin=-4.):
    z=z_origin+(np.arange(rho.shape[-1])+.5)*dx
    a,b=rho[:,:,:-1],rho[:,:,1:]
    cross=(a>=threshold)&(b<threshold)
    roots=np.full(a.shape,np.nan)
    roots[cross]=np.broadcast_to(z[:-1],a.shape)[cross]+dx*(a[cross]-threshold)/(a[cross]-b[cross])
    return roots,cross

def energy(h,guide,lam):
    ok=np.isfinite(h);e=np.sum((h[ok]-guide[ok])**2)
    for ax in [0,1]:
        other=np.roll(h,1,axis=ax);both=ok&np.isfinite(other)
        e+=(lam/4)*np.sum((h[both]-other[both])**2)
    return float(e)

def _choose_dense(roots,guide,width,lam,maxiter=50):
    allowed=np.isfinite(roots)&(np.abs(roots-guide[:,:,None])<=width)
    valid=allowed.any(axis=-1)
    def nearest(target):
        return np.argmin(np.where(allowed,np.abs(roots-target[:,:,None]),np.inf),axis=-1)
    idx=nearest(guide)
    h=np.take_along_axis(roots,idx[:,:,None],axis=-1)[:,:,0];h[~valid]=np.nan
    if all(n%2==0 for n in h.shape):
        parity=np.indices(h.shape).sum(axis=0)%2;ncolors=2
    else:
        cols=[];sizes=[]
        for n in h.shape:
            c=np.arange(n)%2
            if n%2:c[-1]=2
            cols.append(c);sizes.append(int(c.max()+1))
        parity=cols[0][:,None]*sizes[1]+cols[1][None,:];ncolors=sizes[0]*sizes[1]
        assert all(np.all(parity!=np.roll(parity,1,axis=ax)) for ax in [0,1])
    hist=[energy(h,guide,lam)];flips=[]
    for sweep in range(maxiter):
        changed=0
        for colour in range(ncolors):
            total=np.zeros(h.shape);count=np.zeros(h.shape)
            for axis in [0,1]:
                for step in [-1,1]:
                    other=np.roll(h,step,axis=axis);ok=np.isfinite(other)
                    total+=np.where(ok,other,0);count+=ok
            target=(guide+(lam/4)*total)/(1+(lam/4)*count)
            new=nearest(target);update=valid&(parity==colour)
            changed+=int(np.sum((new!=idx)&update));idx[update]=new[update]
            selected=np.take_along_axis(roots,idx[:,:,None],axis=-1)[:,:,0]
            h[update]=selected[update]
        value=energy(h,guide,lam)
        assert value<=hist[-1]+1e-8*max(1,hist[-1]),'Branch energy increased'
        hist.append(value);flips.append(changed)
        if changed==0:break
    idx[~valid]=-1
    jumps=[]
    for ax in [0,1]:
        other=np.roll(h,1,axis=ax);both=np.isfinite(h)&np.isfinite(other)
        jumps.extend(np.abs(h[both]-other[both]).tolist())
    v=h[valid]
    stats={'missing':int((~valid).sum()),'raw_missing':int(np.sum(~np.isfinite(roots).any(axis=-1))),
           'rejected_existing_roots':int(np.sum(np.isfinite(roots).any(axis=-1)&~valid)),
           'stationary':flips[-1]==0,'sweeps':len(flips),'root_switches':int(sum(flips)),
           'energy_history':hist,'edge_jump_p95':float(np.quantile(jumps,.95)) if jumps else None,
           'edge_jump_max':float(max(jumps)) if jumps else None,
           'height_mean':float(v.mean()) if len(v) else None,'height_std':float(v.std()) if len(v) else None,
           'height_minmax':[float(v.min()),float(v.max())] if len(v) else [],
           'columns_gt20':int(np.sum(v>20)),'columns_lt4':int(np.sum(v<4)),
           'finite_FFT':bool(valid.all() and flips[-1]==0)}
    return h,idx,stats

def choose(roots,guide,width,lam,maxiter=50):
    # All callers, including the mesh guide vote, retain every finite candidate.
    if roots.shape[-1]>32:
        from ._packed import pack,unpack_index
        compact,source=pack(roots)
        h,j,stats=_choose_dense(compact,guide,width,lam,maxiter)
        return h,unpack_index(source,j),stats
    return _choose_dense(roots,guide,width,lam,maxiter)

