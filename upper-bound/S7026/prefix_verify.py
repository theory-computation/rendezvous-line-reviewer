"""Independent all-ordered-pair integer-position verifier, no discovery imports."""
from pathlib import Path
from fractions import Fraction
import numpy as np,hashlib,json

def verify(path):
 words=[];C=[]
 for line in Path(path).read_text().splitlines():
  if not line or line.startswith('#'):continue
  w,n,c=line.split();assert len(w)==int(n) and int(c)>0 and w[0]=='0' and not set(w)-set('01');words.append(w);C.append(int(c))
 assert len(set(words))==len(words);n=len(words);lengths=np.array(list(map(len,words)),int);h=int(max(lengths));X=np.zeros((n,h+1),np.int16)
 for i,w in enumerate(words):X[i,1:len(w)+1]=np.cumsum([1 if v=='1' else -1 for v in w]);X[i,len(w)+1:]=X[i,len(w)]
 elapsed=0;hist={}
 for lo in range(0,n,64):
  hi=min(n,lo+64);limit=np.minimum(lengths[lo:hi,None],lengths[None,:]);A=np.zeros((hi-lo,n),np.int16)
  for a,b in ((-1,-1),(-1,1),(1,-1),(1,1)):
   first=np.zeros_like(A)
   for t in range(1,h+1):
    hits=(a*X[lo:hi,t,None]-b*X[None,:,t]==2)&(first==0)&(t<=limit);first[hits]=t
   expected=(np.arange(lo,hi)[:,None]==np.arange(n)[None,:]) if a==b else np.zeros_like(first,dtype=bool)
   assert np.array_equal(first==0,expected),'NON_HITTING_FAMILY'
   first[expected]=limit[expected];A+=first
  for i,row in enumerate(A):elapsed+=C[lo+i]*sum(int(v)*c for v,c in zip(row,C))
 R=Fraction(elapsed,4*sum(C)**2-2*sum(c*c for c in C))
 return {'status':'PASS','method':'independent_blocked_prefix_original_game','exact_R':str(R),'support':n,'ordered_orientation_tests':4*n*n,'strategy_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
if __name__=='__main__':
 import sys
 print(json.dumps(verify(sys.argv[1]),indent=2))
