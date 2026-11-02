"""Periodic main-sheet identification; actual heights remain float64 linear roots."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage import __version__ as skimage_version
from skimage.measure import marching_cubes
from ._branch import density_roots, choose

def surface_components(rho,threshold,dx,guide,z_origin=-4.):
    ns=rho.shape[:2]
    # skimage itself casts to contiguous float32. Build that identical work
    # array directly, without retaining a second full float64 extension.
    ext=np.empty((rho.shape[0]+1,rho.shape[1]+1,rho.shape[2]),dtype=np.float32)
    ext[:-1,:-1]=rho
    ext[-1,:-1]=rho[0]
    ext[:,-1]=ext[:,0]
    vertices,faces,_,_=marching_cubes(ext,level=threshold,method='lewiner',allow_degenerate=False)
    rows=[faces[:,0],faces[:,1]];cols=[faces[:,1],faces[:,2]];boundary=[];unmatched=[];unmatched_low=[]
    for axis in [0,1]:
        other=[i for i in [0,1,2] if i!=axis]
        lo=np.flatnonzero(np.abs(vertices[:,axis])<1e-6)
        hi=np.flatnonzero(np.abs(vertices[:,axis]-ns[axis])<1e-6)
        if len(hi) and len(lo):
            dist,idx=cKDTree(vertices[lo][:,other]).query(vertices[hi][:,other])
            match=dist<1e-4
            unmatched.append(hi[~match])
            lowdist,_=cKDTree(vertices[hi][:,other]).query(vertices[lo][:,other]);unmatched_low.append(lo[lowdist>=1e-4])
            rows.append(hi[match]);cols.append(lo[idx[match]])
            boundary.append({'axis':axis,'low_vertices':len(lo),'high_vertices':len(hi),'unmatched_high':int((~match).sum()),'max_distance':float(dist.max())})
        else:boundary.append({'axis':axis,'low_vertices':len(lo),'high_vertices':len(hi),'unmatched_high':int(len(hi)),'max_distance':None})
    row=np.concatenate(rows);col=np.concatenate(cols)
    graph=coo_matrix((np.ones(len(row),dtype=np.uint8),(row,col)),shape=(len(vertices),len(vertices))).tocsr()
    ncomp,labels=connected_components(graph,directed=False)
    vertex_count=len(vertices);triangle_count=len(faces)
    del ext,graph,row,col,rows,cols,faces
    roots,cross=density_roots(rho,dx,threshold,z_origin);points=np.argwhere(cross).astype(float)
    points[:,2]=(roots[cross]-z_origin)/dx-.5
    dist,near=cKDTree(vertices).query(points)
    assert np.all(dist<1e-3),'Root cannot be located on reconstructed isosurface'
    rootlabel=np.full(roots.shape,-1,dtype=np.int32);rootlabel[cross]=labels[near]
    root_distance_max=float(dist.max())
    del points,near,dist,vertices
    initial,idx,_=choose(roots,guide,np.inf,0.)
    safe=np.maximum(idx,0);initial_labels=np.take_along_axis(rootlabel,safe[:,:,None],axis=-1)[:,:,0]
    good=idx>=0;values,counts=np.unique(initial_labels[good],return_counts=True)
    main=int(values[np.argmax(counts)])
    roots[rootlabel!=main]=np.nan
    restricted=roots
    stats={'vertices':int(vertex_count),'triangles':int(triangle_count),'components':int(ncomp),
           'periodic_boundaries':boundary,'periodic_gate':all(b['unmatched_high']==0 for b in boundary),
           'root_to_mesh_max_distance_cells':root_distance_max,'main_component':main,
           'main_votes':int(counts.max()),'initial_guide_roots':int(good.sum()),
           'main_vote_fraction':float(counts.max()/good.sum()),
           'raw_missing_columns':int((~cross.any(axis=-1)).sum()),
           'main_component_missing_columns':int((~np.isfinite(restricted).any(axis=-1)).sum()),
           'skimage_version':skimage_version,
           'unmatched_vertex_component_labels':[np.unique(labels[v]).astype(int).tolist() for v in unmatched],
           'unmatched_vertices_in_main':[int(np.sum(labels[v]==main)) for v in unmatched],
           'unmatched_low_vertices_in_main':[int(np.sum(labels[v]==main)) for v in unmatched_low],
           'main_periodic_gate':all(np.all(labels[v]!=main) for v in unmatched+unmatched_low),
           'selection':'Largest vote of nearest-guide roots; no spectral information used. Component IDs are local identifiers.'}
    return restricted,rootlabel,stats

