from __future__ import annotations
import time,itertools,math
import numpy as np,pandas as pd
from .common import simulate_dag_seeded, simulate_weights_seeded, simulate_lsem_noise, skeleton_metrics, BASE_SEED

def discretize_tertiles(X):
    X=np.asarray(X,float); n,d=X.shape; out=np.zeros((n,d),dtype=np.int16)
    for j in range(d):
        v=X[:,j]
        q1,q2=np.quantile(v,[1/3,2/3])
        if q1==q2:
            uniq=np.unique(v)
            if len(uniq)<=3:
                mp={x:i for i,x in enumerate(uniq)}; out[:,j]=[mp[x] for x in v]
            else:
                ranks=pd.Series(v).rank(method='average',pct=True).to_numpy(); out[:,j]=np.minimum((ranks*3).astype(int),2)
        else:
            out[:,j]=(v>q1).astype(np.int16)+(v>q2).astype(np.int16)
    return out

def entropy(D,cols):
    if len(cols)==0: return 0.0
    A=D[:,list(cols)]
    if A.ndim==1: A=A[:,None]
    _,cnt=np.unique(A,axis=0,return_counts=True); p=cnt/cnt.sum(); return float(-np.sum(p*np.log(p)))

def mi(D,x,y): return entropy(D,[x])+entropy(D,[y])-entropy(D,[x,y])

def cmi(D,x,y,Z):
    Z=tuple(sorted(set(Z)-{x,y}))
    if not Z: return mi(D,x,y)
    return entropy(D,(x,*Z))+entropy(D,(y,*Z))-entropy(D,Z)-entropy(D,(x,y,*Z))

def local_disc_bic(D,child,parents):
    n=len(D); parents=tuple(sorted(parents)); y=D[:,child]; r=int(y.max())+1
    if not parents:
        cnt=np.bincount(y,minlength=r); nz=cnt>0; ll=float(np.sum(cnt[nz]*np.log(cnt[nz]/n))); k=r-1
        return ll-.5*k*np.log(n)
    P=D[:,parents]; _,inv=np.unique(P,axis=0,return_inverse=True); q=int(inv.max())+1
    ll=0.0
    for g in range(q):
        idx=(inv==g); ng=int(idx.sum())
        if ng==0: continue
        cnt=np.bincount(y[idx],minlength=r); nz=cnt>0; ll+=float(np.sum(cnt[nz]*np.log(cnt[nz]/ng)))
    card=[int(D[:,p].max())+1 for p in parents]; k=(r-1)*int(np.prod(card))
    return ll-.5*k*np.log(n)

def wang_full_skeleton(X,eps_skeleton=.008,eps_collider=.005,eps_prune=.009):
    t0=time.perf_counter(); D=discretize_tertiles(X); n,d=D.shape
    M=np.zeros((d,d)); G=np.zeros((d,d),dtype=np.int8)
    for i in range(d):
        for j in range(i+1,d):
            m=mi(D,i,j);M[i,j]=M[j,i]=m
            if m>=eps_skeleton: G[i,j]=G[j,i]=1
    colliders=[]; protected=set(); parent_claims=[set() for _ in range(d)]
    for x in range(d):
        nbr=list(np.flatnonzero(G[x]))
        for n1,n2 in itertools.combinations(nbr,2):
            if G[n1,n2]: continue
            diff=cmi(D,n1,n2,[x])-M[n1,n2]
            if diff>eps_collider:
                colliders.append((n1,x,n2)); protected.add(tuple(sorted((n1,x))));protected.add(tuple(sorted((n2,x))))
                parent_claims[x].add(n1);parent_claims[x].add(n2)
    collider_nodes=set(v for tri in colliders for v in tri)
    for x in range(d):
        for y in range(x+1,d):
            if not G[x,y] or (x,y) in protected: continue
            neigh=(set(np.flatnonzero(G[x]))|set(np.flatnonzero(G[y])))-{x,y}-collider_nodes
            if cmi(D,x,y,neigh)<eps_prune: G[x,y]=G[y,x]=0
    for ch in range(d): parent_claims[ch]={p for p in parent_claims[ch] if G[p,ch]}
    for ch in range(d):
        cur=set(parent_claims[ch])
        while cur:
            base=local_disc_bic(D,ch,cur); rem=[]
            for p in sorted(cur):
                if local_disc_bic(D,ch,cur-{p})>base: rem.append(p)
            if not rem: break
            for p in rem:
                cur.discard(p)
                if ch not in parent_claims[p]: G[p,ch]=G[ch,p]=0
            parent_claims[ch]=set(cur)
    # Step 4 only orients the remaining skeleton. The published state-wise
    # orientation rule was not generalized for the common continuous benchmark.
    return G,time.perf_counter()-t0,dict(initial_edges=int(np.triu((M>=eps_skeleton),1).sum()),colliders=len(colliders),final_edges=int(np.triu(G,1).sum()))

def one(d=10,s=2,noise='normal',rep=0):
    ni=['normal','exponential','gumbel'].index(noise);seed=BASE_SEED+100000*d+1000*int(10*s)+10*ni+rep
    T=simulate_dag_seeded(d,2*d,seed);W=simulate_weights_seeded(T,s,seed+1);X=simulate_lsem_noise(W,500,noise,seed+2)
    S,rt,extra=wang_full_skeleton(X);return dict(d=d,s=s,noise=noise,rep=rep,seed=seed,method='Wang2026-full-adapted',runtime=rt,**extra,**skeleton_metrics(T,S))
if __name__=='__main__':
 import sys;print(one(int(sys.argv[1]),float(sys.argv[2]),sys.argv[3],int(sys.argv[4])))
