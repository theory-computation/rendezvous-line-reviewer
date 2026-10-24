// Independent physical proof: literal velocity segments, not discovery matrices.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using I=__int128;
std::string dec(I a){std::string s;do{s+=char('0'+a%10);a/=10;}while(a);std::reverse(s.begin(),s.end());return s;}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("ARGUMENTS");std::ifstream in(argv[1]);if(!in)throw std::runtime_error("FILE");
 std::vector<std::string>W;std::vector<int64_t>C;std::set<std::string>known;std::string line;I sum=0,squares=0;int horizon=0;
 while(std::getline(in,line)){
  if(line.empty()||line[0]=='#')continue;std::istringstream r(line);std::string w,extra;int l;int64_t c;if(!(r>>w>>l>>c)||r>>extra||l!=(int)w.size()||c<=0||w[0]!='0'||l>80||w.find_first_not_of("01")!=std::string::npos||!known.insert(w).second)throw std::runtime_error("INVALID_ROW");
  W.push_back(w);C.push_back(c);sum+=c;squares+=(I)c*c;horizon=std::max(horizon,l);
 }
 if(W.empty()||sum>(I)100000000000000000LL)throw std::runtime_error("INTEGER_PROOF_RANGE");
 I elapsed=0;uint64_t cases=0,renewals=0;
 for(size_t i=0;i<W.size();i++)for(size_t j=0;j<=i;j++){
  int cost=0;for(int sa=-1;sa<=1;sa+=2)for(int sb=-1;sb<=1;sb+=2){
   int left=0,right=2,hit=0;
   for(size_t t=0;t<std::min(W[i].size(),W[j].size());t++){
    int va=sa*(W[i][t]=='1'?1:-1),vb=sb*(W[j][t]=='1'?1:-1);int gap=right-left,next=gap+vb-va;
    if(gap*next<=0){int speed=va-vb;if(speed==0||gap%speed!=0||gap/speed<0||gap/speed>1)throw std::runtime_error("NONINTEGER_SEGMENT_CONTACT");hit=t+gap/speed;break;}
    left+=va;right+=vb;
   }
   if(!hit){if(i!=j||sa!=sb)throw std::runtime_error("NON_HITTING_PAIR");hit=W[i].size();renewals++;}
   cost+=hit;cases++;
  }
  elapsed+=(I)(i==j?1:2)*C[i]*C[j]*cost;
 }
 I den=4*sum*sum-2*squares;
 std::cout<<"{\"numerator\":\""<<dec(elapsed)<<"\",\"denominator\":\""<<dec(den)<<"\",\"canonical_support\":"<<W.size()<<",\"triangular_orientation_cases\":"<<cases<<",\"equivalent_ordered_orientation_cases\":"<<4*W.size()*W.size()<<",\"renewal_cases\":"<<renewals<<"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}
