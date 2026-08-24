from __future__ import annotations
import itertools,time
import numpy as np,pandas as pd
from .wang_full import entropy, mi, cmi
from .common import simulate_dag_seeded, simulate_weights_seeded, simulate_lsem_noise, skeleton_metrics, BASE_SEED

THRESHOLDS=[(.008,.005,.009),(.008,.005,.010),(.009,.005,.009),(.009,.005,.010)]

def local_disc_bic_fast(D,child,parents):
    parents=tuple(sorted(parents)); n=len(D); y=D[:,child].astype(np.int64); r=int(y.max())+1
    if not parents:
        cnt=np.bincount(y,minlength=r); nz=cnt>0
        ll=float(np.sum(cnt[nz]*np.log(cnt[nz]/n))); k=r-1
        return ll-.5*k*np.log(n)
    P=D[:,parents]; _,inv=np.unique(P,axis=0,return_inverse=True); q=int(inv.max())+1
    joint=np.bincount(inv*r+y,minlength=q*r).reshape(q,r); ng=joint.sum(axis=1); nz=joint>0
    denom=np.repeat(ng[:,None],r,axis=1); ll=float(np.sum(joint[nz]*np.log(joint[nz]/denom[nz])))
    card=[int(D[:,pp].max())+1 for pp in parents]; k=(r-1)*int(np.prod(card))
    return ll-.5*k*np.log(n)

def discretize_quantiles(X,bins=3):
    X=np.asarray(X,float); n,d=X.shape; out=np.zeros((n,d),dtype=np.int16)
    for j in range(d):
        v=X[:,j]; uniq=np.unique(v)
        if len(uniq)<=bins:
            mp={x:i for i,x in enumerate(sorted(uniq.tolist()))}; out[:,j]=np.array([mp[x] for x in v],dtype=np.int16); continue
        qs=np.quantile(v,np.arange(1,bins)/bins)
        if len(np.unique(qs))<len(qs):
            ranks=pd.Series(v).rank(method='average',pct=True).to_numpy(); out[:,j]=np.minimum((ranks*bins).astype(int),bins-1)
        else:
            out[:,j]=np.sum(v[:,None]>qs[None,:],axis=1).astype(np.int16)
    return out

def wang_full_skeleton_bins(X,eps_skeleton=.008,eps_collider=.005,eps_prune=.009,bins=3):
    t0=time.perf_counter(); D=discretize_quantiles(X,bins); n,d=D.shape
    M=np.zeros((d,d)); G=np.zeros((d,d),dtype=np.int8)
    for i in range(d):
        for j in range(i+1,d):
            m=mi(D,i,j); M[i,j]=M[j,i]=m
            if m>=eps_skeleton: G[i,j]=G[j,i]=1
    colliders=[]; protected=set(); parent_claims=[set() for _ in range(d)]
    for x in range(d):
        nbr=list(np.flatnonzero(G[x]))
        for n1,n2 in itertools.combinations(nbr,2):
            if G[n1,n2]: continue
            diff=cmi(D,n1,n2,[x])-M[n1,n2]
            if diff>eps_collider:
                colliders.append((n1,x,n2)); protected.add(tuple(sorted((n1,x)))); protected.add(tuple(sorted((n2,x))))
                parent_claims[x].add(n1); parent_claims[x].add(n2)
    # The paper does not further formalize "collider-related nodes"; this
    # conservative interpretation is exposed as an implementation choice.
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
            base=local_disc_bic_fast(D,ch,cur); rem=[]
            for p in sorted(cur):
                if local_disc_bic_fast(D,ch,cur-{p})>base: rem.append(p)
            if not rem: break
            for p in rem:
                cur.discard(p)
                if ch not in parent_claims[p]: G[p,ch]=G[ch,p]=0
            parent_claims[ch]=set(cur)
    return G,time.perf_counter()-t0,dict(initial_edges=int(np.triu((M>=eps_skeleton),1).sum()),colliders=len(colliders),final_edges=int(np.triu(G,1).sum()))

def sim_one(d,s,noise,rep,eps_s,eps_c,eps_p,bins):
    ni=['normal','exponential','gumbel'].index(noise);seed=BASE_SEED+100000*d+1000*int(10*s)+10*ni+rep
    T=simulate_dag_seeded(d,2*d,seed);W=simulate_weights_seeded(T,s,seed+1);X=simulate_lsem_noise(W,500,noise,seed+2)
    S,rt,extra=wang_full_skeleton_bins(X,eps_s,eps_c,eps_p,bins)
    return dict(d=d,s=s,noise=noise,rep=rep,seed=seed,bins=bins,eps_skeleton=eps_s,eps_collider=eps_c,eps_prune=eps_p,runtime=rt,**extra,**skeleton_metrics(T,S))
