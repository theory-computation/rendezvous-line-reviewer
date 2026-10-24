// Audit reference: expand signed movements, inspect EVERY ordered pair at integer times.
// No triangular symmetry reduction and no imports from the optimizer or other verifiers.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using U = unsigned __int128;
struct Movement {std::array<int,81> position{}; int length; uint64_t weight;};
std::string decimal(U n) {std::string s; do {s.push_back('0'+n%10);n/=10;}while(n);std::reverse(s.begin(),s.end());return s;}
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
int main(int argc,char** argv) {try {
 require(argc==2,"one strategy path required");std::ifstream file(argv[1]);require(bool(file),"cannot open strategy");
 std::vector<Movement> rows;std::set<std::string> seen;std::string line;U C=0,Q=0;int H=0;
 while(std::getline(file,line)) {
  if(line.empty()||line[0]=='#')continue;
  std::istringstream in(line);std::string word,extra;int length;int64_t weight;
  require(bool(in>>word>>length>>weight)&&!(in>>extra),"bad row");
  require(length>0&&length<=80&&int(word.size())==length&&word[0]=='0'&&word.find_first_not_of("01")==std::string::npos&&weight>0,"bad movement");
  require(seen.insert(word).second,"duplicate canonical word");
  require(C+U(weight)<=U(100000000000000000ULL),"arithmetic range exceeded");
  C+=weight;Q+=U(weight)*weight;H=std::max(H,length);
  for(int sign:{1,-1}) {Movement a{};a.length=length;a.weight=weight;for(int t=1;t<=length;++t)a.position[t]=a.position[t-1]+sign*(word[t-1]=='1'?1:-1);rows.push_back(a);}
 }
 require(!rows.empty(),"empty family");
 std::array<U,81> meeting{},failure{};std::array<uint64_t,81> meeting_count{},failure_count{};
 uint64_t pairs=0,diagonal=0,ordered_digest=14695981039346656037ULL;
 for(size_t i=0;i<rows.size();++i)for(size_t j=0;j<rows.size();++j) {
  int t=0;bool fail=i==j;const auto&a=rows[i];const auto&b=rows[j];
  if(fail) {t=a.length;failure[t]+=U(a.weight)*b.weight;++failure_count[t];++diagonal;}
  else {for(t=1;t<=std::min(a.length,b.length);++t)if(a.position[t]-b.position[t]==2)break;
   require(t<=std::min(a.length,b.length),"distinct ordered pair does not meet");meeting[t]+=U(a.weight)*b.weight;++meeting_count[t];}
  // FNV-1a-style recurrence over integer tokens (i,j,t,fail), diagnostic only.
  for(uint64_t token:{uint64_t(i),uint64_t(j),uint64_t(t),uint64_t(fail)}) {ordered_digest^=token;ordered_digest*=1099511628211ULL;}
  ++pairs;
 }
 U elapsed=0,total=0;for(int t=0;t<=H;++t){elapsed+=U(t)*(meeting[t]+failure[t]);total+=meeting[t]+failure[t];}
 require(total==4*C*C,"mass total");
 std::cout<<"{\"canonical_support\":"<<rows.size()/2<<",\"oriented_support\":"<<rows.size()<<",\"ordered_pairs\":"<<pairs<<",\"diagonal_pairs\":"<<diagonal<<",\"ordered_token_digest_u64\":\""<<ordered_digest<<"\",\"unconditional_denominator\":\""<<decimal(4*C*C)<<"\",\"numerator\":\""<<decimal(elapsed)<<"\",\"denominator\":\""<<decimal(4*C*C-2*Q)<<"\"";
 auto print=[&](const char*name,auto&v){std::cout<<",\""<<name<<"\":[";for(int t=0;t<=H;++t){if(t)std::cout<<",";std::cout<<"\""<<decimal(U(v[t]))<<"\"";}std::cout<<"]";};
 print("meeting_mass",meeting);print("failed_epoch_mass",failure);print("meeting_pair_counts",meeting_count);print("failed_pair_counts",failure_count);std::cout<<"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}
