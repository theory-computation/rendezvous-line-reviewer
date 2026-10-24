#!/usr/bin/env python3
"""Independent V9 full-domain exact verifier. Does not import discovery or core.
Reweighted parent Gram uses direct Int64 matrix products, not the producer's floating
integer BLAS or its cache. Physical costs use chronological bit scanning.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict
import hashlib,json,time,argparse
import numpy as np
if not __debug__:
 raise RuntimeError('Exact certification requires Python without -O or PYTHONOPTIMIZE.')
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'inputs/parent'
D=B
import sys
sys.path.insert(0,str(ROOT/'vendor'))
from vector_oracles import lower_physical as physical

def independent_motif(kind):
 if kind=='PETERSEN':
  n=10;edges=[(i,(i+1)%5) for i in range(5)]+[(i,i+5) for i in range(5)]+[(i+5,((i+2)%5)+5) for i in range(5)]
 else:
  if kind not in ['C5','C7','C9','C11','C13']:raise ValueError('UNSUPPORTED_COP_GRAPH')
  n=int(kind[1:]);edges=[(i,(i+1)%n) for i in range(n)]
 a=[[0]*n for _ in range(n)]
 for i,j in edges:a[i][j]=a[j][i]=1
 if any(a[i][j] and a[j][k] and a[k][i] for i in range(n) for j in range(i+1,n) for k in range(j+1,n)):raise ValueError('TRIANGLE_INVALIDATES_CONSTRUCTOR')
 return np.array([[1-2*a[i][j] for j in range(n)] for i in range(n)],np.int64)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--certificate',type=Path,default=ROOT/'inputs/baseline/certificate.json');ap.add_argument('--oracle',choices=['chronological','displacement'],default='chronological');ap.add_argument('--negative-control',action='store_true');ap.add_argument('--output',type=Path);args=ap.parse_args()
 start=time.time();p=json.loads((B/'parent_rank_one_certificate.json').read_text());base=json.loads((B/'candidate_best.json').read_text());suf=json.loads((B/'certificate.json').read_text());cert=json.loads(args.certificate.read_text());T=int(cert['T']);tail=int(cert['k']);assert (T,tail)==(11,2), 'THIS_ASSIGNMENT_IS_T11_K2_ONLY'
 for n,h in p['input_array_sha256'].items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==h
 get=lambda n:np.load(D/(n+'.npy')).astype(np.int64)
 U,V=get('gram_U_num'),get('gram_V_num');gu,gv=get('t11_group_u'),get('t11_group_v')
 assert base['group_delta_denominator']==1
 wu=get('t11_weights_u_num')+np.array(base['u_group_delta_numerators'],np.int64)
 wv=get('t11_weights_v_num')+np.array(base['v_group_delta_numerators'],np.int64)
 original_wu=wu.copy();original_wv=wv.copy()
 wu=np.array(cert['parent_u_weights'],np.int64);wv=np.array(cert['parent_v_weights'],np.int64)
 assert wu.shape==(256,) and wv.shape==(256,) and min(wu)>0 and min(wv)>=0
 seen=set()
 for typ,g,delta in cert['parent_changes_native_integer']:
  assert typ in ['u','v'] and 0<=g<256 and (typ,g) not in seen;seen.add((typ,g))
  old,new=(original_wu,wu) if typ=='u' else (original_wv,wv)
  assert int(new[g])-int(old[g])==delta
 for typ,old,new in [('u',original_wu,wu),('v',original_wv,wv)]:
  assert all(int(old[g])==int(new[g]) or (typ,g) in seen for g in range(256))
 den=int(p['gram_denominator']);H=int(p['coefficient_denominator']);alpha=F(p['downdate_coefficient'])
 cf=np.array(p['coefficient_numerators'],np.int64)
 # Verify every product and every accumulation is inside signed 64 bits,
 # using unsigned absolute Cauchy row-norm bounds computed before the products.
 def gram(A,w):
  rownorm=(A*A).sum(axis=1)
  assert int(rownorm.max())*int(np.abs(w).max())<2**63
  g=(A*w)@A.T
  assert np.array_equal(g,g.T)
  return g
 G1=gram(U,wu[gu]);G2=gram(V,wv[gv]);print('direct integer Gram rebuilt',time.time()-start,flush=True)
 inv=sum((F(sum(int(t)**2 for t in cf[gu==g]),int(wu[g])) for g in range(256)),F(0))
 margin=1-alpha*den*inv/(H*H);assert margin>0 and str(margin)==cert['exact_parent_psd_margin']
 assert int(np.abs(U).sum(axis=1).max())*int(np.abs(cf).max())<2**63
 vv=U@cf;assert min(vv)>0
 lookup={}
 for k,(a,b) in enumerate(zip(get('t11_parent_reps'),get('t11_parent_refl'))):lookup[int(a)]=(k,1);lookup[int(b)]=(k,-1)
 idx=np.zeros(8192,np.int32);sg=np.zeros(8192,np.int8);vec=np.zeros(8192,np.int64)
 for i in range(8192):
  h=i//2;key=h//2+2048*(1-h%2)
  if key in lookup:idx[i],sg[i]=lookup[key];vec[i]=vv[idx[i]]
 C=np.load(B/'column_numerators.npy');assert hashlib.sha256(C.astype('<i8').tobytes()).hexdigest()==suf['column_numerators_sha256']
 eta=list(map(int,base['suffix_eta_numerators']));CD=int(suf['column_denominator']);sd=int(base['suffix_eta_denominator'])*CD*CD
 assert all(e>=0 for e in eta) and int(np.abs(C).max())<=CD
 coremul=F(cert.get('psd_core_multiplier','1'));assert 0<coremul<=1
 extra_n=cert.get('global_weight_numerators',[]);extra_d=cert.get('global_weight_denominator',1)*cert.get('global_column_denominator',1)**2
 extra_cols=np.load(args.certificate.parent/'global_column_numerators.npy') if extra_n else np.empty((8192,0),dtype=np.int64)
 assert extra_d>0 and all(type(n) is int and n>=0 for n in extra_n)
 assert extra_cols.shape==(8192,len(extra_n))
 if extra_n:
  assert hashlib.sha256((args.certificate.parent/'global_column_numerators.npy').read_bytes()).hexdigest()==cert['global_column_sha256']
  assert np.max(np.abs(extra_cols))<=cert['global_column_denominator']
 extra_bound=F(sum(extra_n),cert.get('global_weight_denominator',1))
 coremul=F(cert.get('psd_core_multiplier','1'));assert 0<coremul<=1
 f=F(cert['target'])
 assert f-F(cert['previous_official_target'])==F(cert['gain_over_previous_official'])
 assert len(cert['edges'])==cert['edge_count']
 scale=list(map(int,cert.get('positive_scaling_numerators',[1]*8192)))
 assert len(scale)==8192 and min(scale)>0
 # Independent triangle-free graph COP pullback: each of five feature rows is a
 # nonnegative (possibly overlapping) union of complete four-route modules.
 # For every nonnegative x, Ex>=0, and H=J-2*A(C5) is copositive.
 # E^T H E can have negative entries but is globally copositive.
 ceil=lambda x:-(-x.numerator//x.denominator)
 horn_den=int(cert.get('horn_weight_denominator',1)); assert horn_den>0
 horn_records=cert.get('horn_factors',[]); HC=np.ones((5,5),np.int8)
 for h in range(5):HC[h,(h+1)%5]=HC[(h+1)%5,h]=-1
 hp=defaultdict(int);hm=[]
 for at in horn_records:
  rows=at['feature_groups']; assert all(type(g) is int for row in rows for g in row)
  HC=independent_motif(at.get('motif','C5'))
  w=int(at['integer_weight']);assert w>0 and len(rows)==len(HC)
  assert all(row and len(row)<=128 and all(0<=g<(1<<T) for g in row) for row in rows)
  groups=sorted({g for row in rows for g in row})
  E=np.array([[row.count(g) for g in groups] for row in rows],np.int64)
  mat=E.T@HC.astype(np.int64)@E
  assert (mat==mat.T).all() and np.diag(mat).min()>=0
  bounds={}
  for u,g in enumerate(groups):
   for v,h in enumerate(groups):
    coeff=int(mat[u,v])
    if not coeff:continue
    for i in range(g<<tail,(g+1)<<tail):
     for j in range(h<<tail,(h+1)<<tail):hp[i,j]+=coeff*w
    bounds[u,v]=coremul*F(coeff*w*den,horn_den)
  hm.append((groups,bounds))
 edge={};diag=defaultdict(F)
 for i,j,w in cert['edges']:
  w=F(w);assert 0<=i<j<8192 and w>0 and (i,j) not in edge
  edge[i,j]=edge[j,i]=w;diag[i]+=w*F(scale[j],scale[i]);diag[j]+=w*F(scale[i],scale[j])
 remainders=[]
 for i in range(8192):
  gi=int(G1[idx[i],idx[i]])+int(G2[idx[i],idx[i]]) if sg[i] else 0
  pi=F(gi,den)-alpha*F(int(vec[i])**2,H*H)
  pi+=F(sum(e*int(C[i,t])**2 for t,e in enumerate(eta)),sd)
  pi=coremul*(pi+F(sum(n*int(extra_cols[i,t])**2 for t,n in enumerate(extra_n)),extra_d)+F(hp.get((i,i),0),horn_den))+diag[i]
  remainders.append(F(T)-pi)
 min_diag=min(range(8192),key=lambda i:remainders[i]);assert remainders[min_diag]>=0
 assert args.negative_control or [str(remainders[min_diag]),min_diag]==cert['minimum_exact_diagonal_remainder']
 ceil=lambda x:-(-x.numerator//x.denominator)
 thc=ceil(f*den);ths=ceil(F(sum(eta)*CD*CD*den,sd)+den*extra_bound)
 gb=int(np.diag(G1).max())+int(np.diag(G2).max())
 core_guard=ceil((1-coremul)*gb)
 assert gb+14*den+thc+ths+core_guard+sum(max([ceil(abs(frac)) for frac in bounds.values()] or [0]) for _,bounds in hm)+max([ceil(x*den) for x in diag.values()]+[0])<2**63
 checks=0;total_b1=0;near=0;minimum=None;argmin=None;digest=hashlib.sha256()
 if args.oracle=='displacement':
  import sys
  sys.path.insert(0,str(B/'source'))
  from physical_oracle_b import BlockBuilder
  alt=BlockBuilder(T,tail)
 for lo in range(0,8192,64):
  hi=lo+64
  if args.oracle=='chronological':q,bb=physical(T,tail,lo,hi)
  else:q,bb=alt.block(lo,hi)
  total_b1+=int(bb.sum())
  digest.update(q.tobytes());digest.update(bb.tobytes())
  g=G1[np.ix_(idx[lo:hi],idx)]+sg[lo:hi,None]*sg[None,:]*G2[np.ix_(idx[lo:hi],idx)]
  g*=((sg[lo:hi,None]!=0)&(sg[None,:]!=0))
  coarse=q.astype(np.int64)*(den//2)-g-thc*bb.astype(np.int64)-ths-core_guard
  for groups,bounds in hm:
   for h,gg in enumerate(groups):
    rr=np.flatnonzero((np.arange(lo,hi)>>tail)==gg)
    if not len(rr):continue
    for h2,gg2 in enumerate(groups):
     frac=bounds.get((h,h2))
     if frac is None:continue
     cc=np.arange(gg2<<tail,(gg2+1)<<tail)
     if frac>0:coarse[np.ix_(rr,cc)]-=ceil(frac)
     else:coarse[np.ix_(rr,cc)]+=(-frac).numerator//(-frac).denominator
  for i in range(lo,hi):coarse[i-lo,i]-=ceil(den*diag[i])
  aa,jj=np.nonzero(coarse<0)
  for a,j in zip(aa,jj):
   i=int(lo+a);j=int(j)
   gram=F(int(g[a,j]),den)-alpha*F(int(vec[i])*int(vec[j]),H*H)
   suffix=F(sum(e*int(C[i,k])*int(C[j,k]) for k,e in enumerate(eta)),sd)
   r=F(int(q[a,j]),2)-f*int(bb[a,j])-coremul*(gram+suffix+F(sum(n*int(extra_cols[i,k])*int(extra_cols[j,k]) for k,n in enumerate(extra_n)),extra_d)+F(hp.get((i,j),0),horn_den))
   r+=edge.get((i,j),F(0)) if i!=j else -diag[i]
   assert r>=0,(i,j,str(r))
   if minimum is None or r<minimum:minimum=r;argmin=[i,j]
   near+=1
  checks+=q.size
 assert checks==67108864
 assert total_b1==67108864-8192*(2**tail),'physical B-mask population mismatch'
 if T==11:assert digest.hexdigest()=='78358e24a6c697bd6b2102f768c2fdc67a5f330b45b1afd03c4c7788c2aed133', 'T11 physical digest mismatch'
 out={'status':'INDEPENDENT_T11_EXACT_FULL_DOMAIN_PASS','target':str(f),'ordered_entries':checks,'rational_near_entries':near,'minimum_near_residual':str(minimum),'argmin':argmin,'physical_digest':digest.hexdigest(),'parent_psd_margin_positive':True,'positive_scaled_edge_atoms':len(edge)//2,'positive_shared_atoms':len(extra_n),'positive_graph_COP_atoms':len(horn_records),'seconds':time.time()-start,'independence':'No core/discovery imports; direct integer parent products; '+args.oracle+' physical oracle; separate PSD and rational residual reconstruction'}
 out['psd_core_multiplier']=str(coremul);out['oracle']=args.oracle;out['B1_entries']=total_b1
 out['certificate_sha256']=hashlib.sha256(args.certificate.read_bytes()).hexdigest()
 out['verifier_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 out['T']=T;out['k']=tail;out['changed_parent_groups']=len(seen)
 out['exact_gain_over_previous_official']=cert['gain_over_previous_official']
 out['minimum_exact_diagonal_remainder']=cert['minimum_exact_diagonal_remainder']
 (args.output or args.certificate.parent.joinpath('independent_'+args.oracle+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
