// Independent SDK-free little-endian ABI decoder, no UNIGINE headers.
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
using Bytes=std::vector<unsigned char>;
static void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static Bytes read(const char* name){
    std::ifstream s(name,std::ios::binary);
    require(bool(s),"cannot open file");
    return Bytes(std::istreambuf_iterator<char>(s),std::istreambuf_iterator<char>());
}
static std::uint32_t u32(const Bytes& b,std::size_t off){
    require(off<=b.size() && b.size()-off>=4,"truncated word");
    return std::uint32_t(b[off])|(std::uint32_t(b[off+1])<<8)|
           (std::uint32_t(b[off+2])<<16)|(std::uint32_t(b[off+3])<<24);
}
int main(int argc,char** argv){
    try{
        require(argc==4,"usage: decoder params cells layer_count");
        const auto p=read(argv[1]),c=read(argv[2]);
        const auto layers=std::stoull(argv[3]);
        require(p.size()>=16,"header truncated");
        require(p[0]=='P'&&p[1]=='C'&&p[2]=='G'&&p[3]=='F',"bad magic");
        const auto count=u32(p,8),stride=u32(p,12);
        require(u32(p,4)==1 && count>0 && count<=65536 && stride==112,"unsupported ABI");
        require(p.size()==16+std::size_t(count)*stride,"record length mismatch");
        std::vector<unsigned char> seen(c.size(),0);
        std::uint32_t walls=0;
        for(std::uint32_t i=0;i<count;++i){
            const std::size_t off=16+std::size_t(i)*stride;
            const auto idx=u32(p,off),layer=u32(p,off+4),flags=u32(p,off+8);
            const auto cell=u32(p,off+12),floors=u32(p,off+16),bays=u32(p,off+20);
            require(idx==i && layer<layers,"invalid index or material layer");
            require((flags&~31u)==0 && (flags&3u)<3u,"invalid flags");
            require(u32(p,off+40)==0 && u32(p,off+44)==0,"reserved bits");
            float f[16];
            for(std::size_t j=0;j<16;++j){f[j]=std::bit_cast<float>(u32(p,off+48+j*4));require(std::isfinite(f[j]),"nonfinite float");}
            require(f[14]>0 && f[15]>=0 && f[15]<=1,"invalid repeat or wear");
            if((flags&3u)==0){
                ++walls;
                require(floors>0 && floors<=200 && f[0]>0 && f[1]>0 && f[2]>0 && f[4]>0,"invalid facade dimensions");
                const std::uint64_t end=std::uint64_t(cell)+std::uint64_t(floors)*bays;
                require(end<=c.size(),"cell range out of bounds");
                for(std::uint64_t j=cell;j<end;++j){
                    require(seen[j]++==0,"cell range overlap");
                    const unsigned role=c[j]&3u,var=c[j]>>2;
                    require(role<3 && var<3 && (role!=0||var==0),"invalid cell code");
                }
                if(flags&16u)require(f[10]>=0&&f[11]>=0&&f[12]>0&&f[13]>0&&f[10]+f[12]<=f[0]+1e-4f&&f[11]+f[13]<=f[1]+1e-4f,"invalid door");
            }else require(cell==0&&floors==0&&bays==0&&(flags&~3u)==0,"non-wall grid");
        }
        for(auto n:seen)require(n==1,"unreferenced cell");
        std::cout<<"{\"records\":"<<count<<",\"cells\":"<<c.size()<<",\"walls\":"<<walls<<",\"stride\":"<<stride<<"}\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
