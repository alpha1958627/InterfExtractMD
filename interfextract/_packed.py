"""Lossless storage of finite density crossings, retaining original z indices."""
import numpy as np
def pack(roots):
 valid=np.isfinite(roots);count=valid.sum(axis=-1);k=max(1,int(count.max()));nz=roots.shape[-1];idx=np.flatnonzero(valid.ravel());del valid
 col=idx//nz;offset=np.r_[0,np.cumsum(count.ravel())];rank=np.arange(len(idx))-offset[col]
 packed=np.full(roots.shape[:2]+(k,),np.nan);source=np.full(packed.shape,-1,dtype=np.int32)
 packed.reshape(-1,k)[col,rank]=roots.ravel()[idx];source.reshape(-1,k)[col,rank]=idx%nz
 assert np.isfinite(packed).sum()==count.sum()
 return packed,source
def unpack_index(source,selected):
 result=np.take_along_axis(source,np.maximum(selected,0)[:,:,None],axis=-1)[:,:,0];result[selected<0]=-1;return result
