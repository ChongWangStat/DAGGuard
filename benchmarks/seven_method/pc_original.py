from __future__ import annotations
import itertools,time,numpy as np
from .common import fisherz_p

def pc_original_skeleton(X,alpha=.05):
    t0=time.perf_counter();X=np.asarray(X,float);n,d=X.shape;C=np.corrcoef(X,rowvar=False)
    G=np.ones((d,d),dtype=bool);np.fill_diagonal(G,False);depth=0;tests=0
    while True:
        any_eligible=False
        # Tetrad-style: ordered pairs; graph/neighborhood updated immediately.
        for a in range(d):
            bs=list(np.flatnonzero(G[a]))
            for b in bs:
                if not G[a,b]:continue
                nbr=[v for v in np.flatnonzero(G[a]) if v!=b]
                if len(nbr)<depth:continue
                any_eligible=True
                for Cset in itertools.combinations(nbr,depth):
                    p=fisherz_p(C,n,a,b,Cset);tests+=1
                    if p>=alpha:
                        G[a,b]=G[b,a]=False;break
        depth+=1
        possible=False
        for a in range(d):
            for b in np.flatnonzero(G[a]):
                if len([v for v in np.flatnonzero(G[a]) if v!=b])>=depth:
                    possible=True;break
            if possible:break
        if not possible or not any_eligible:break
    return G.astype(int),time.perf_counter()-t0,tests
