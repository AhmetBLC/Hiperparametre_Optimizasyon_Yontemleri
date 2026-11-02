import json,gzip,glob,collections,itertools,sys
import numpy as np
from scipy import stats
J=sys.argv[1]
M=['Grid','Random','TPE','CMAES','GA','BOHB']
test=collections.defaultdict(list); cv=collections.defaultdict(list); t10=collections.defaultdict(list)
per_search=[]
for f in glob.glob(J+'/*.json.gz'):
    d=json.load(gzip.open(f)); m=d['method']; b=(d['dataset'],d['model'])
    if m=='Default': continue
    if m=='Grid' and d['tag']!='Grid100': continue
    fin={x['budget']:x for x in d['finals']}
    last=fin[max(fin)]
    test[(b,m)].append(last['test']['f1_macro']); cv[(b,m)].append(last['cv_score'])
    if 10 in fin: per_search.append((b,m,fin[10]['test']['f1_macro'],fin[100]['test']['f1_macro'],fin[10]['cv_score'],fin[100]['cv_score']))
blocks=sorted({k[0] for k in test})
T=np.array([[np.mean(test[(b,m)]) for m in M] for b in blocks])
C=np.array([[np.mean(cv[(b,m)]) for m in M] for b in blocks])
print('blocks',len(blocks)); print('Friedman test',stats.friedmanchisquare(*T.T)); print('Friedman cv',stats.friedmanchisquare(*C.T))
# Bayesian signed-rank (Benavoli et al. 2017)
def bsr(x,rope,nsamp=50000,s=0.5,seed=0):
    rng=np.random.default_rng(seed); z=np.concatenate([[0.0],x]); n=len(z)
    W=rng.dirichlet(np.concatenate([[s],np.ones(n-1)]),nsamp)
    S=z[:,None]+z[None,:]
    L=(S<-2*rope); R=(S>2*rope); E=~(L|R)
    pl=np.einsum('ki,ij,kj->k',W,L,W); pr=np.einsum('ki,ij,kj->k',W,R,W); pe=np.einsum('ki,ij,kj->k',W,E,W)
    win=np.argmax(np.stack([pl,pe,pr]),axis=0)
    return [(win==i).mean() for i in range(3)]
for name,X,r in [('TEST',T,0.005),('CV',C,0.005),('TEST rope0.01',T,0.01)]:
    print('\n'+name,'rope',r,' P(A<B) P(rope) P(A>B)')
    for i,j in itertools.combinations(range(6),2):
        p=bsr(X[:,i]-X[:,j],r); print(f'{M[i]:7s}-{M[j]:7s}', ' '.join(f'{v:.3f}' for v in p))
# Spearman within-block permutation
dc=C-C.mean(1,keepdims=True); dt=T-T.mean(1,keepdims=True)
rho=stats.spearmanr(dc.ravel(),dt.ravel())[0]
rng=np.random.default_rng(1); null=[]
for _ in range(20000):
    pt=np.array([row[rng.permutation(6)] for row in dt]); null.append(stats.spearmanr(dc.ravel(),pt.ravel())[0])
null=np.array(null); print('\nrho',rho,'perm p two-sided',(np.abs(null)>=abs(rho)).mean())
taus=[stats.kendalltau(C[k],T[k])[0] for k in range(len(blocks))]; print('mean within-block Kendall tau',np.mean(taus), 'wilcoxon vs0',stats.wilcoxon(taus))
# budget 10->100 change on test
a=np.array([(p[3]-p[2]) for p in per_search]); c_=np.array([(p[5]-p[4]) for p in per_search])
print('\nsearches',len(a),'test drop share',(a<0).mean(),'mean test change pp',a.mean()*100,'mean cv change pp',c_.mean()*100)
for m in M[1:]:
    aa=np.array([p[3]-p[2] for p in per_search if p[1]==m]); print(m,'share test(100)<test(10)',round((aa<0).mean(),3),'mean pp',round(aa.mean()*100,2))
