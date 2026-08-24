from __future__ import annotations
import copy,itertools,time
import numpy as np
from pathlib import Path
from .common import fisherz_p

def matlab_find(mask):
    """Return (row,col) pairs in MATLAB column-major find order."""
    mask=np.asarray(mask)
    out=[]
    nr,nc=mask.shape
    for c in range(nc):
        for r in range(nr):
            if mask[r,c]: out.append((r,c))
    return out

def assign_iden(G):
    # MATLAB: matr=tril(G); idx=find(matr==1); matr(matr==1)=1:length(idx); matr=matr+tril(matr)'
    d=G.shape[0]; lower=np.tril((G==1).astype(int)); IDs=np.zeros((d,d),dtype=int); k=0
    for r,c in matlab_find(lower==1):
        k+=1; IDs[r,c]=k
    IDs=IDs+np.tril(IDs).T
    return IDs

def get_skeleton_stable(C,n,alpha,kmax=None):
    d=C.shape[0]; G=np.ones((d,d),dtype=int);np.fill_diagonal(G,0)
    sep=[[set() for _ in range(d)] for __ in range(d)]
    cell=[[[] for _ in range(d)] for __ in range(d)]
    ord_=0
    while True:
        done=True
        pairs=matlab_find(G!=0)
        nbrs2=[list(np.flatnonzero(G[i])) for i in range(d)]
        for x,y in pairs:
            nbrs=[v for v in nbrs2[y] if v!=x]
            if len(nbrs)>=ord_ and G[x,y]!=0:
                done=False
                for S in itertools.combinations(nbrs,ord_):
                    p=fisherz_p(C,n,x,y,S)
                    if p<=alpha:
                        cell[x][y].append(p);cell[y][x]=list(cell[x][y])
                    if p>alpha:
                        G[x,y]=G[y,x]=0;sep[x][y]|=set(S);sep[y][x]|=set(S);break
        ord_+=1
        if kmax is None:
            if ord_>int(np.max(np.sum(G,axis=1),initial=0)):break
        elif ord_>kmax:break
        if done:break
    for r,c in matlab_find(np.array([[bool(z) for z in row] for row in cell])):
        if G[r,c]==1: cell[r][c]=[max(cell[r][c])]
        else: cell[r][c]=[]
    return G,sep,cell,assign_iden(G)

def get_v_structures(C,n,G,sep,cell_in,IDs_in,kmax=None):
    cell=copy.deepcopy(cell_in); IDs=IDs_in.copy(); pdag=G.copy();d=G.shape[0]
    cell2=[[[] for _ in range(d)] for __ in range(d)];maxid=int(IDs.max())
    for x,y in matlab_find(G!=0):
        Z=[z for z in np.flatnonzero(G[y]) if z!=x]
        for z in Z:
            if G[x,z]==0 and y not in sep[x][z]:
                if not (pdag[x,y]==-1 and pdag[z,y]==-1):
                    pdag[x,y]=-1;pdag[z,y]=-1
                    ii1=list(np.flatnonzero(G[x]>0));ii2=list(np.flatnonzero(G[z]>0))
                    k1=max(len(ii1),len(ii2)) if kmax is None else kmax
                    rows=[]
                    for ii in (ii1,ii2):
                        for kk in range(1,k1+1):
                            if len(ii)>1:
                                if kk<=len(ii): rows.extend(itertools.combinations(ii,kk))
                            else:
                                if ii: rows.append(tuple(ii))
                                break
                    # MATLAB SS(t), t=1:size(SS,1), is first column under linear indexing.
                    p2=[]
                    for subset in rows:
                        cond=[subset[0]] if subset else []
                        p2.append(fisherz_p(C,n,x,z,tuple([z]+cond)))
                    p2m=max(p2) if p2 else 0.0
                    cxy=cell[x][y][0] if cell[x][y] else None
                    czy=cell[z][y][0] if cell[z][y] else None
                    vals1=[v for v in (cxy,p2m) if v is not None]
                    vals2=[v for v in (czy,p2m) if v is not None]
                    cell2[z][y].append(max(vals1) if vals1 else 0.0)
                    cell2[x][y].append(max(vals2) if vals2 else 0.0)
                    if len(cell2[z][y])==1 and len(cell2[x][y])==1:
                        IDs[x,y]=maxid+1;IDs[z,y]=maxid+1
                    else:
                        IDs[x,y]=maxid+1;IDs[z,y]=maxid+2
                    maxid=int(IDs.max())
    for rr,cc in matlab_find(pdag==-1):
        if pdag[cc,rr]==1:
            pdag[cc,rr]=0;IDs[cc,rr]=0
    for rr,cc in matlab_find(pdag==0): cell[rr][cc]=[]
    nonempty=np.array([[bool(z) for z in row] for row in cell2])
    for r,c in matlab_find(nonempty):
        base=cell[r][c] if cell[r][c] else []
        cell[r][c]=[max(base+[sum(cell2[r][c])])]
    return pdag,cell,IDs

def clamp_edges(pdag,G,cell2,cell_orig,IDs):
    pdag=pdag.copy();G=G.copy();cell2=copy.deepcopy(cell2);IDs=IDs.copy();d=G.shape[0]
    for i,j in matlab_find(pdag==-1):
        if pdag[i,j]==-1 and pdag[j,i]==-1:
            pdag[i,j]=pdag[j,i]=2;G[i,j]=G[j,i]=2
            cell2[i][j]=list(cell_orig[i][j]);cell2[j][i]=list(cell_orig[j][i])
    for i,j in list(matlab_find(pdag==2)):
        idx1=list(np.where(pdag[:,j]==-1)[0]);idx2=list(np.where(pdag[:,i]==-1)[0])
        for t in idx1:
            G[t,j]=G[j,t]=2;cell2[t][j]=list(cell_orig[t][j]);cell2[j][t]=list(cell2[t][j]);pdag[t,j]=pdag[j,t]=2
        for t in idx2:
            G[i,t]=G[t,i]=2;cell2[i][t]=list(cell_orig[i][t]);cell2[t][i]=list(cell2[i][t]);pdag[i,t]=pdag[t,i]=2
    maxid=int(IDs.max())
    for r,c in matlab_find(pdag==2):
        maxid+=1;IDs[r,c]=maxid;IDs[c,r]=maxid
    return pdag,G,cell2,IDs

def _set_literal_linear(M, node_values, value):
    flat=M.ravel(order='F').copy()
    for idx0 in node_values:
        if 0<=idx0<flat.size:flat[idx0]=value
    M[:]=flat.reshape(M.shape,order='F')

def orientation_rules(G,pdag,cell,IDs,literal_conflict_indexing=False):
    G=G.copy();pdag=pdag.copy();cell=copy.deepcopy(cell);IDs=IDs.copy();d=G.shape[0]
    old=np.zeros_like(G);maxid=int(IDs.max());iters=0;conflicts=0;implicated=0
    while not np.array_equal(pdag,old):
        iters+=1
        if iters>1000: raise RuntimeError('orientation rules did not converge')
        old=pdag.copy();p2=pdag.copy();cell_d=[[[] for _ in range(d)] for __ in range(d)];pdag_u=[[[] for _ in range(d)] for __ in range(d)]
        for a,b in matlab_find(pdag==-1):
            Cs=[c for c in range(d) if pdag[b,c]==1 and G[a,c]==0]
            if Cs:
                for c in Cs:
                    p2[b,c]=-1;pdag_u[b][c]=[(a,b)]
                    val=cell[a][b][0] if cell[a][b] else 0.0
                    cell_d[b][c]=[sum(cell_d[b][c])+val]
        for a,b in matlab_find(pdag==1):
            idx=[k for k in range(d) if pdag[a,k]==-1 and pdag[k,b]==-1]
            if idx:
                p2[a,b]=-1
                pdag_u[a][b].extend([(a,k) for k in idx]);pdag_u[a][b].extend([(k,b) for k in idx])
                rows=[]
                for k in idx:
                    v1=cell[a][k][0] if cell[a][k] else 0.0;v2=cell[k][b][0] if cell[k][b] else 0.0;rows.append(max(v1,v2))
                cell_d[a][b]=[max([0.0]+cell_d[a][b])+sum(rows)]
        for a,b in matlab_find(pdag==1):
            Cs=[c for c in range(d) if pdag[a,c]==1 and pdag[c,b]==-1]
            found=any(G[c1,c2]==0 for c1,c2 in itertools.combinations(Cs,2))
            if found:
                p2[a,b]=-1;pdag_u[a][b].extend([(a,c) for c in Cs]);pdag_u[a][b].extend([(c,b) for c in Cs])
                pv=[]
                for c in Cs:
                    v1=cell[a][c][0] if cell[a][c] else 0.0;v2=cell[c][b][0] if cell[c][b] else 0.0;pv.append(max(v1,v2))
                s=sum(max(pv[i],pv[j]) for i,j in itertools.combinations(range(len(pv)),2))
                cell_d[a][b]=[max([0.0]+cell_d[a][b])+s]
        for i,j in list(matlab_find(p2==-1)):
            if p2[i,j]==-1 and p2[j,i]==-1:
                conflicts+=1;p2[i,j]=p2[j,i]=2;G[i,j]=G[j,i]=2
                for u,v in pdag_u[i][j]+pdag_u[j][i]:
                    implicated+=1
                    if literal_conflict_indexing:
                        _set_literal_linear(p2,[u,v],2);_set_literal_linear(G,[u,v],2)
                        _set_literal_linear(p2,[v,u],2);_set_literal_linear(G,[v,u],2)
                    else:
                        p2[u,v]=p2[v,u]=2;G[u,v]=G[v,u]=2
        nonempty=np.array([[bool(z) for z in row] for row in cell_d])
        for r,c in matlab_find(nonempty):
            if p2[r,c]!=2:
                base=cell[r][c] if cell[r][c] else []
                cell[r][c]=[max(cell_d[r][c]+base)]
                maxid+=1;IDs[r,c]=maxid;pdag[r,c]=p2[r,c]
                if not cell_d[c][r]:
                    cell[c][r]=[];IDs[c,r]=0;pdag[c,r]=0
            else:
                pdag[r,c]=pdag[c,r]=2
    return pdag,cell,IDs,dict(orientation_iterations=iters,orientation_conflicts=conflicts,orientation_implicated_edges=implicated)

def by_fdr(p,alpha):
    p=np.asarray(p,float);nt=len(p);den=int(np.sum(p<=alpha));den=den if den else 1
    return float(nt*alpha*np.sum(1/np.arange(1,nt+1))/den)

def binary_search_official(p,q):
    start=1e-100;last=1.0;cent=(start+last)/2;a2=cent
    for _ in range(50):
        a1=(start+cent)/2;ans1=by_fdr(p,a1)
        a2=(cent+last)/2;ans2=by_fdr(p,a2)
        if ans1>q:last=a1
        elif ans2>q:last=a2
        else:start=a2
        cent=(start+last)/2
    return float(a2)

def control_fdr_official(pdag,cell,IDs,q):
    d=IDs.shape[0]
    ids=sorted(int(z) for z in np.unique(IDs) if z>0)
    first_positions=[];pvals=[]
    for idv in ids:
        loc=np.argwhere(IDs==idv)
        if len(loc)==0:continue
        r,c=min(((int(r),int(c)) for r,c in loc),key=lambda rc:rc[0]+rc[1]*d)
        pv=cell[r][c]
        if not pv:
            raise RuntimeError(f'positive structure ID {idv} has empty p-value at first occurrence {(r,c)}')
        first_positions.append((idv,r,c));pvals.append(float(pv[0]))
    if not pvals:return np.zeros_like(pdag),0.0,np.array([])
    p=np.array(pvals,float);p[p>1]=1.0
    alpha=binary_search_official(p,q);out=pdag.copy()
    for (idv,_,_),pv in zip(first_positions,p):
        if pv>alpha:out[IDs==idv]=0
    return out,alpha,p

def pc_p_skeleton_faithful(X,q=.05,literal_conflict_indexing=False,return_details=False):
    t0=time.perf_counter();X=np.asarray(X,float);n,d=X.shape;C=np.corrcoef(X,rowvar=False)
    alpha=.20 if n<=250 else .20/np.sqrt(n/250.)
    G0,sep,cell0,IDs0=get_skeleton_stable(C,n,alpha,None)
    cell_orig=copy.deepcopy(cell0);IDs_orig=IDs0.copy();G_orig=G0.copy()
    pdag,cell1,IDs1=get_v_structures(C,n,G0,sep,cell0,IDs0,None)
    pdag,Gwork,cell2,IDs2=clamp_edges(pdag,G0,cell1,cell_orig,IDs1)
    pdag,cell3,IDs3,diag=orientation_rules(Gwork,pdag,cell2,IDs2,literal_conflict_indexing=literal_conflict_indexing)
    for i,j in list(matlab_find(pdag==2)):
        Gwork[i,j]=Gwork[j,i]=1;pdag[i,j]=pdag[j,i]=1
        cell3[i][j]=list(cell_orig[i][j]);cell3[j][i]=list(cell_orig[j][i])
        IDs3[i,j]=IDs_orig[i,j];IDs3[j,i]=IDs_orig[j,i]
    adj,astar,pvals=control_fdr_official(pdag,cell3,IDs3,q)
    S=((adj!=0)|(adj.T!=0)).astype(int);np.fill_diagonal(S,0)
    details=dict(initial_alpha=alpha,alpha_star=astar,raw_skeleton_edges=int(np.triu(G_orig!=0,1).sum()),pre_fdr_pdag_adjacencies=int(np.triu(((pdag!=0)|(pdag.T!=0)),1).sum()),fdr_edges=int(np.triu(S,1).sum()),unique_hypotheses=len(pvals),p_min=float(np.min(pvals)) if len(pvals) else np.nan,p_median=float(np.median(pvals)) if len(pvals) else np.nan,p_max=float(np.max(pvals)) if len(pvals) else np.nan,**diag)
    if return_details:return S,time.perf_counter()-t0,details
    return S,time.perf_counter()-t0,astar
