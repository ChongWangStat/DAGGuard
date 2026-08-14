#!/usr/bin/env python3
"""Re-run the six primary NOTEARS-BP simulation panels from the recovered original code."""
import argparse, json, random
from pathlib import Path
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.stats import norm
import networkx as nx
import igraph as ig
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.search.ScoreBased.GES import ges
import lingam

BASE_SEED=12123
METHODS=['GES','PC','LiNGAM','NOTEARS','NOTEARS-BP']

def notears_linear(X, lambda1=0, max_iter=100, h_tol=1e-8, rho_max=1e16, w_threshold=0):
    def _loss(W):
        M=X@W; R=X-M
        return 0.5/X.shape[0]*(R**2).sum(), -1.0/X.shape[0]*X.T@R
    def _h(W):
        d0=W.shape[0]; E=slin.expm(W*W)
        return np.trace(E)-d0, E.T*W*2
    def _adj(w): return (w[:d*d]-w[d*d:]).reshape([d,d])
    def _func(w):
        W=_adj(w); loss,G_loss=_loss(W); h0,G_h=_h(W)
        obj=loss+0.5*rho*h0*h0+alpha*h0+lambda1*w.sum()
        G=G_loss+(rho*h0+alpha)*G_h
        return obj,np.concatenate((G+lambda1,-G+lambda1),axis=None)
    n,d=X.shape; X=X-np.mean(X,axis=0,keepdims=True)
    w_est=np.zeros(2*d*d); rho=1.0; alpha=0.0; h=np.inf
    bnds=[(0,0) if i==j else (0,None) for _ in range(2) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        while rho<rho_max:
            sol=sopt.minimize(_func,w_est,method='L-BFGS-B',jac=True,bounds=bnds)
            w_new=sol.x; h_new,_=_h(_adj(w_new))
            if h_new>0.25*h: rho*=10
            else: break
        w_est,h=w_new,h_new; alpha+=rho*h
        if h<=h_tol or rho>=rho_max: break
    W_est=_adj(w_est); W_est[np.abs(W_est)<w_threshold]=0
    return W_est

def simulate_lsem(G,n):
    W=nx.to_numpy_array(G); d=W.shape[0]; X=np.zeros((n,d))
    for j in nx.topological_sort(G):
        parents=list(G.predecessors(j)); e=np.random.normal(scale=1,size=n)
        X[:,j]=X[:,parents].dot(W[parents,j])+e if parents else e
    return X

def simulate_dag(d,s0):
    def perm(M):
        P=np.random.permutation(np.eye(M.shape[0])); return P.T@M@P
    G=ig.Graph.Erdos_Renyi(n=d,m=s0)
    Bund=np.array(G.get_adjacency().data)
    B=np.tril(perm(Bund),k=-1); B=perm(B)
    assert ig.Graph.Adjacency(B.tolist()).is_dag()
    return B

def sim_uniform(B,s):
    W=np.zeros(B.shape); ranges=((-s,-0.5),(0.5,s)); S=np.random.randint(2,size=B.shape)
    for i,(lo,hi) in enumerate(ranges):
        U=np.random.uniform(lo,hi,size=B.shape); W+=B*(S==i)*U
    return W

def sim_modnormal(B,s):
    x=norm.rvs(loc=0,scale=s,size=B.shape); y=np.zeros(B.shape); W=np.zeros(B.shape)
    y[x<=0]=x[x<=0]-0.5; y[x>0]=x[x>0]+0.5; W[B!=0]=y[B!=0]
    return W

def local_graph(G,j):
    H=nx.DiGraph(); H.add_nodes_from(G.nodes()); H.add_edges_from((p,j) for p in G.predecessors(j)); return H

def bic(X,G):
    X=X-np.mean(X,axis=0,keepdims=True); n,d=X.shape; ll=0.0
    for i in range(d):
        pa=list(G.predecessors(i))
        if pa:
            beta=np.linalg.lstsq(X[:,pa],X[:,i],rcond=None)[0]; r=X[:,i]-X[:,pa]@beta
        else: r=X[:,i]
        s2=(r**2).mean(); ll+=-0.5*n*(np.log(2*np.pi*s2)+1)
    k=np.count_nonzero(nx.to_numpy_array(G,dtype=int))+d
    return -2*ll+k*np.log(n)

def prune(X,G):
    Xc=X-np.mean(X,axis=0,keepdims=True)
    improved=True
    while improved and len(G.edges())>0:
        improved=False; cand=[]
        for u,v in list(G.edges()):
            T=G.copy(); before=local_graph(T,v); T.remove_edge(u,v); after=local_graph(T,v)
            cand.append(((u,v),bic(Xc,after)-bic(Xc,before)))
        edge,delta=min(cand,key=lambda z:z[1])
        if delta<0: G.remove_edge(*edge); improved=True
    return G

def shd(A,B):
    z=np.sum(np.abs(A-B)); n=A.shape[0]
    for i in range(n):
        for j in range(n):
            if A[i,j]==1 and B[i,j]==0 and B[j,i]==1: z-=1
    return float(z)

def metric(A,B):
    den=np.sum(np.abs(B)); true=np.sum(np.abs(A))
    fd=np.sum(B-A==1)/den if den else np.nan; fn=np.sum(B-A==-1)/true
    return fd,1-fn,shd(A,B)

def one(rep,i,d,rg,kind):
    seed=BASE_SEED+1000*i+rep; np.random.seed(seed); random.seed(seed)
    B=simulate_dag(d,2*d); W=sim_uniform(B,rg[i]) if kind=='uniform' else sim_modnormal(B,rg[i])
    X=simulate_lsem(nx.DiGraph(W),500)
    W0=notears_linear(X,lambda1=0.1); A0=np.where(np.abs(W0)>0.3,1,0)
    A1=nx.to_numpy_array(prune(X,nx.DiGraph(A0)),dtype=int)
    S=StandardScaler().fit_transform(X); Ws=notears_linear(S,lambda1=0.1); Ast=np.where(np.abs(Ws)>0.3,1,0)
    g=ges(X)['G']; Ages=np.array(g.graph,dtype=int); Ages[Ages==-1]=0; Ages=Ages.T
    g=pc(X,alpha=0.05,ci_test=fisherz,max_cond_set=3).G; Apc=np.array(g.graph,dtype=int); Apc[Apc==-1]=0; Apc=Apc.T
    model=lingam.DirectLiNGAM(); model.fit(X); Alin=np.where(np.abs(model.adjacency_matrix_.T)>0,1,0)
    rows=[]
    for name,A in [('GES',Ages),('PC',Apc),('LiNGAM',Alin),('NOTEARS',A0),('NOTEARS-BP',A1),('STD_NOTEARS',Ast)]:
        fdr,tpr,s=metric(B,A); rows.append(dict(d=d,weight_kind=kind,scale=rg[i],rep=rep,seed=seed,method=name,fdr=fdr,tpr=tpr,shd=s,true_edges=int(B.sum()),estimated_edges=int(A.sum())))
    return rows

def plot_metric(df,metric,ylabel,title,rg,path):
    colors={'GES':'orange','PC':'green','LiNGAM':'yellow','NOTEARS':'blue','NOTEARS-BP':'red'}
    pos={'GES':-0.4,'PC':-0.24,'LiNGAM':-0.08,'NOTEARS':0.08,'NOTEARS-BP':0.24}
    fig,ax=plt.subplots(figsize=(18,5)); handles=[]
    for m in METHODS:
        vals=[df[(df.scale==s)&(df.method==m)][metric].to_numpy() for s in rg]
        bp=ax.boxplot(vals,positions=np.arange(1,len(rg)+1)+pos[m],widths=.15,patch_artist=True,boxprops=dict(facecolor=colors[m],color='black'),medianprops=dict(color='black'))
        handles.append(bp['boxes'][0])
    ax.set_xlim(0.4,len(rg)+0.74); ax.set_xticks(np.arange(1,len(rg)+1)); ax.set_xticklabels(rg,fontsize=16)
    ax.set_xlabel('S',fontsize=16); ax.set_ylabel(ylabel,fontsize=16); ax.legend(handles,METHODS,loc='upper right'); ax.set_title(title,fontsize=16)
    for k in range(len(rg)-1): ax.axvline(x=k+1.4,color='gray',linestyle='--',linewidth=.8)
    fig.tight_layout(); fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight'); fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight'); fig.savefig(path.with_suffix('.svg'),bbox_inches='tight'); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--d',type=int,required=True); p.add_argument('--kind',choices=['uniform','modnormal'],required=True); p.add_argument('--M',type=int,default=20); p.add_argument('--jobs',type=int,default=2); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    rg=[1,4,7,10] if a.kind=='uniform' else [1,2,3,4]; a.out.mkdir(parents=True,exist_ok=True)
    tasks=[(j,i) for i in range(4) for j in range(a.M)]
    result=Parallel(n_jobs=a.jobs,verbose=10)(delayed(one)(j,i,a.d,rg,a.kind) for j,i in tasks)
    df=pd.DataFrame([r for q in result for r in q]); df.to_csv(a.out/'replicate_metrics.csv',index=False)
    summ=df.groupby(['d','weight_kind','scale','method'],as_index=False).agg(fdr=('fdr','mean'),tpr=('tpr','mean'),shd=('shd','mean'),edges=('estimated_edges','mean')); summ.to_csv(a.out/'summary.csv',index=False)
    np.save(a.out/'all_results_rows.npy',df.to_records(index=False),allow_pickle=True)
    plot_metric(df,'fdr','FDR','False Discovery Rate',rg,a.out/'fdr'); plot_metric(df,'tpr','TPR','True Positive Rate',rg,a.out/'tpr'); plot_metric(df,'shd','SHD','SHD',rg,a.out/'shd')
    (a.out/'environment.json').write_text(json.dumps(dict(d=a.d,kind=a.kind,M=a.M,base_seed=BASE_SEED,scales=rg,n=500,lambda1=.1,threshold=.3),indent=2))
    print(summ.to_string(index=False))
if __name__=='__main__': main()
