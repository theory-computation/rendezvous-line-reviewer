"""Independent NumPy-only physical kernels. No Numba, compiler or optimizer."""
import numpy as np

def lower_physical(T, k, lo, hi):
    nbits=T+k; n=1<<nbits
    if not (1<=T<=14 and 1<=k<=3 and 0<=lo<hi<=n):
        raise ValueError('bounded physical domain')
    i=np.arange(lo,hi,dtype=np.int32)[:,None]
    j=np.arange(n,dtype=np.int32)[None,:]
    delta=np.zeros((hi-lo,n),np.int16)
    positive=np.zeros_like(delta,dtype=np.uint8)
    negative=np.zeros_like(positive)
    for t in range(nbits):
        shift=nbits-1-t
        delta+=((i>>shift)&1)-((j>>shift)&1)
        positive[(positive==0)&(delta==1)]=t+1
        negative[(negative==0)&(delta==-1)]=t+1
    positive=np.where(positive,positive,nbits+1-delta)
    negative=np.where(negative,negative,nbits+1+delta)
    mask=((i>>k)!=(j>>k)).astype(np.uint8)
    return np.where(mask,positive+negative,2*T).astype(np.uint8),mask

class NumpyContacts:
    """Same public output contract as frozen C++ Contacts, independent code."""
    binary_sha='NONE_NUMPY_ONLY'
    def row(self,w,X,L):
        z=np.cumsum([1 if b=='1' else -1 for b in w],dtype=np.int16)
        h=np.zeros((len(L),4),np.int16)
        for t in range(min(len(w),X.shape[1])):
            alive=t<L
            for r,(a,b) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
                take=(h[:,r]==0)&alive&(a*z[t]-b*X[:,t]==2)
                h[take,r]=t+1
        return h
    def matrix(self,W):
        if not W or len(set(W))!=len(W):raise ValueError('EMPTY_OR_DUPLICATE')
        if any(not w or w[0]!='0' or set(w)-set('01') or len(w)>80 for w in W):raise ValueError('WORD')
        L=np.array(list(map(len,W)),np.int16);n=len(W)
        X=np.zeros((n,int(L.max())),np.int16)
        for i,w in enumerate(W):
            z=np.cumsum([1 if b=='1' else -1 for b in w],dtype=np.int16)
            X[i,:len(w)]=z;X[i,len(w):]=z[-1]
        A=np.zeros((n,n),np.int16);M=np.zeros_like(A)
        for i,w in enumerate(W):
            h=self.row(w,X[:i],L[:i])
            if np.any(h==0):raise ValueError('PAIR_INCOMPATIBLE')
            if i:A[i,:i]=A[:i,i]=h.sum(1);M[i,:i]=M[:i,i]=h.max(1)
            hits=[np.flatnonzero(X[i,:L[i]]==s) for s in (-1,1)]
            if any(len(a)==0 for a in hits):raise ValueError('SELF_INVALID')
            p,m=(int(a[0])+1 for a in hits)
            A[i,i]=2*L[i]+p+m;M[i,i]=max(p,m)
        return L,X,A,M
