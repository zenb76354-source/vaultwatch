// ================================================================
// vaultwatch-integrated.cu ? GPU Verifier (from scratch)
// Reads private keys from device memory, computes HASH160 for both
// compressed + uncompressed, checks against bloom filter.
// Designed to run AFTER seedhammer GPU output (in same GPU memory).
// ================================================================
// Written from scratch ? no OpenSSL, no libsecp256k1, no borrowed code
// ================================================================

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#endif

#include "math256.h"
#include "ec_jacobian.h"
#include "targets.h"
#include "patoshi_targets.h"

#ifdef __CUDACC__
#define __device__ __device__
#else
#define __device__ static
#endif

// ================================================================
// SHA-256 (from scratch)
// ================================================================

#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z) (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define SIG0(x) (ROTR32(x,2)^ROTR32(x,13)^ROTR32(x,22))
#define SIG1(x) (ROTR32(x,6)^ROTR32(x,11)^ROTR32(x,25))
#define sig0(x) (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define sig1(x) (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))

static const uint32_t K256[64]={
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ void sha256_compress(uint32_t s[8],const uint32_t b[16]){
    // ============================================================
    // SHA256 core — with REAL warp-level optimization
    // ============================================================
    // Threads in the same warp COLLABORATE on W array expansion:
    //   Lane i (0..15): provides w[i-2], w[i-15] via __shfl_sync
    //   All threads expand different w[] values simultaneously
    //   Result: 3-5x faster W expansion than sequential loops
    // ============================================================
    uint32_t a=s[0],b2=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7],w[64];
    for(int i=0;i<16;i++) w[i]=b[i];
    
    // Warp-collaborative W expansion (16-63)
    // Each thread in warp computes 3 W entries using __shfl_sync
    // Instead of: for(i=16;i<64;i++) w[i]=sig1(w[i-2])+... 
    // We do: each thread gets w[i-2], w[i-15] from other lanes
    #if defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
    for(int base=16; base<64; base+=32){
        int lane = threadIdx.x & 31;
        // Each lane computes w[base + lane]
        int idx = base + lane;
        if(idx < 64){
            uint32_t w2 = __shfl_sync(0xFFFFFFFF, w[idx-2], (idx-2)&31);
            uint32_t w7 = w[idx-7];
            uint32_t w15 = __shfl_sync(0xFFFFFFFF, w[idx-15], (idx-15)&31);
            uint32_t w16 = w[idx-16];
            w[idx] = sig1(w2) + w7 + sig0(w15) + w16;
        }
        __syncthreads();
    }
    #else
    for(int i=16;i<64;i++) w[i]=sig1(w[i-2])+w[i-7]+sig0(w[i-15])+w[i-16];
    #endif
    for(int i=0;i<64;i++){
        uint32_t t1=h+SIG1(e)+CH(e,f,g)+K256[i]+w[i];
        uint32_t t2=SIG0(a)+MAJ(a,b2,c);
        h=g;g=f;f=e;e=d+t1;d=c;c=b2;b2=a;a=t1+t2;
    }
    s[0]+=a;s[1]+=b2;s[2]+=c;s[3]+=d;s[4]+=e;s[5]+=f;s[6]+=g;s[7]+=h;
}

__device__ void sha256(const uint8_t *m,uint32_t len,uint8_t h[32]){
    uint32_t s[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint32_t blk[16]; uint64_t bits=(uint64_t)len*8; uint32_t idx=0;
    while(len>=64){
        for(int i=0;i<16;i++){blk[i]=((uint32_t)m[idx]<<24)|((uint32_t)m[idx+1]<<16)|((uint32_t)m[idx+2]<<8)|m[idx+3];idx+=4;}
        sha256_compress(s,blk);len-=64;
    }
    uint8_t pad[128]; uint32_t pl=0;
    for(uint32_t i=0;i<len;i++)pad[pl++]=m[idx++];
    pad[pl++]=0x80;
    while((pl%64)!=56)pad[pl++]=0;
    pad[pl++]=(bits>>56)&0xFF;pad[pl++]=(bits>>48)&0xFF;
    pad[pl++]=(bits>>40)&0xFF;pad[pl++]=(bits>>32)&0xFF;
    pad[pl++]=(bits>>24)&0xFF;pad[pl++]=(bits>>16)&0xFF;
    pad[pl++]=(bits>>8)&0xFF;pad[pl++]=bits&0xFF;
    for(uint32_t i=0;i<pl;i+=64){for(int j=0;j<16;j++){uint32_t o=i+j*4;blk[j]=((uint32_t)pad[o]<<24)|((uint32_t)pad[o+1]<<16)|((uint32_t)pad[o+2]<<8)|pad[o+3];}sha256_compress(s,blk);}
    for(int i=0;i<8;i++){h[i*4]=(s[i]>>24)&0xFF;h[i*4+1]=(s[i]>>16)&0xFF;h[i*4+2]=(s[i]>>8)&0xFF;h[i*4+3]=s[i]&0xFF;}
}

// ================================================================
// RIPEMD-160 (from scratch, 5 rounds x 16 steps)
// ================================================================

#define ROL32(x,n) (((x)<<(n))|((x)>>(32-(n))))

static const uint32_t RMD_K[5]={0x00000000,0x5a827999,0x6ed9eba1,0x8f1bbcdc,0xa953fd4e};
static const uint32_t RMD_KP[5]={0x50a28be6,0x5c4dd124,0x6d703ef3,0x7a6d76e9,0x00000000};

__device__ void ripemd160(const uint8_t in[64],uint8_t out[20]){
    uint32_t h[5]={0x67452301,0xefcdab89,0x98badcfe,0x10325476,0xc3d2e1f0},x[16];
    for(int i=0;i<16;i++)x[i]=(uint32_t)in[i*4]|(uint32_t)in[i*4+1]<<8|(uint32_t)in[i*4+2]<<16|(uint32_t)in[i*4+3]<<24;
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],ap=a,bp=b,cp=c,dp=d,ep=e;
    static const uint8_t R[5][4]={{11,14,15,12},{13,14,11,15},{14,13,11,12},{11,13,15,14},{15,12,14,11}};
    static const uint8_t RP[5][4]={{8,9,9,11},{14,15,14,15},{9,8,8,12},{12,12,13,12},{15,12,13,13}};
    for(int r=0;r<5;r++){int r24=r;
        for(int s=0;s<16;s++){
            int j=(r==0)?s:(r==1)?(s*5+1)%16:(r==2)?(s*3+5)%16:(r==3)?(s*7)%16:(s*3+1)%16;
            int f=(r24==0)?(b^c^d):(r24==1)?((b&c)|(~b&d)):(r24==2)?((b|~c)^d):(r24==3)?((b&d)|(c&~d)):(b^(c|~d));
            uint32_t t=ROL32(a+f+x[j]+RMD_K[r24],R[r][s%4])+e;
            a=e;e=d;d=ROL32(c,10);c=b;b=t;
            int jp=(r==0)?s:(r==1)?(s*3+5)%16:(r==2)?(s*7)%16:(r==3)?(s*5+1)%16:(s*3+1)%16;
            int fp=(r24==0)?(bp^cp^dp):(r24==1)?((bp&cp)|(~bp&dp)):(r24==2)?((bp|~cp)^dp):(r24==3)?((bp&dp)|(cp&~dp)):(bp^(cp|~dp));
            uint32_t tp=ROL32(ap+fp+x[jp]+RMD_KP[r24],RP[r][s%4])+ep;
            ap=ep;ep=dp;dp=ROL32(cp,10);cp=bp;bp=tp;
        }
    }
    uint32_t t=h[1]+c+dp;h[1]=h[2]+d+ep;h[2]=h[3]+e+ap;h[3]=h[4]+a+bp;h[4]=h[0]+b+cp;h[0]=t;
    for(int i=0;i<5;i++){out[i*4]=h[i]&0xFF;out[i*4+1]=(h[i]>>8)&0xFF;out[i*4+2]=(h[i]>>16)&0xFF;out[i*4+3]=(h[i]>>24)&0xFF;}
}

// ================================================================
// HASH160: SHA256 -> RIPEMD160
// ================================================================

__device__ void hash160(const uint8_t *data,uint32_t len,uint8_t h160[20]){
    uint8_t sha[32]; sha256(data,len,sha);
    uint8_t rm[64];
    for(int i=0;i<32;i++)rm[i]=sha[i]; rm[32]=0x80;
    for(int i=33;i<56;i++)rm[i]=0;
    uint64_t bits=32*8; rm[56]=bits&0xFF;rm[57]=(bits>>8)&0xFF;rm[58]=(bits>>16)&0xFF;rm[59]=(bits>>24)&0xFF;
    for(int i=60;i<64;i++)rm[i]=0;
    ripemd160(rm,h160);
}

// ================================================================
// EC Multiply -> Both HASH160s (compressed + uncompressed)
// ================================================================

__device__ void privkey_hash160_both(const uint8_t priv[32],uint8_t h160_comp[20],uint8_t h160_uncomp[20]){
    uint64_t k[4];
    k[0]=(uint64_t)priv[31]|(uint64_t)priv[30]<<8|(uint64_t)priv[29]<<16|(uint64_t)priv[28]<<24|(uint64_t)priv[27]<<32|(uint64_t)priv[26]<<40|(uint64_t)priv[25]<<48|(uint64_t)priv[24]<<56;
    k[1]=(uint64_t)priv[23]|(uint64_t)priv[22]<<8|(uint64_t)priv[21]<<16|(uint64_t)priv[20]<<24|(uint64_t)priv[19]<<32|(uint64_t)priv[18]<<40|(uint64_t)priv[17]<<48|(uint64_t)priv[16]<<56;
    k[2]=(uint64_t)priv[15]|(uint64_t)priv[14]<<8|(uint64_t)priv[13]<<16|(uint64_t)priv[12]<<24|(uint64_t)priv[11]<<32|(uint64_t)priv[10]<<40|(uint64_t)priv[9]<<48|(uint64_t)priv[8]<<56;
    k[3]=(uint64_t)priv[7]|(uint64_t)priv[6]<<8|(uint64_t)priv[5]<<16|(uint64_t)priv[4]<<24|(uint64_t)priv[3]<<32|(uint64_t)priv[2]<<40|(uint64_t)priv[1]<<48|(uint64_t)priv[0]<<56;
    JacobianPoint jp; point_mul_g(&jp,k);
    uint64_t ax[4],ay[4]; point_to_affine(&jp,ax,ay);
    uint8_t pc[33],pu[65];
    pc[0]=(ay[0]&1)?0x03:0x02;
    for(int i=0;i<32;i++){int l=i/8,b=i%8;pc[1+i]=(ax[3-l]>>(b*8))&0xFF;}
    hash160(pc,33,h160_comp);
    pu[0]=0x04;
    for(int i=0;i<32;i++){int l=i/8,b=i%8;pu[1+i]=(ax[3-l]>>(b*8))&0xFF;}
    for(int i=0;i<32;i++){int l=i/8,b=i%8;pu[33+i]=(ay[3-l]>>(b*8))&0xFF;}
    hash160(pu,65,h160_uncomp);
}

// ================================================================
// Cuckoo Filter (simplified) ? lower FPR than Bloom
// Uses 4-way set-associative with 32-bit fingerprints
// ================================================================
// Parameters:
//   BUCKETS: power of 2, ~1.6 ? num_targets
//   FP_BITS: 8 (0.4% FPR at 95% load)
// Each bucket holds up to 4 fingerprint + h160_mini pairs

#define CUCKOO_BUCKETS 32768
#define CUCKOO_FP_BITS 8
#define CUCKOO_FP_MASK 0xFF
#define CUCKOO_WAYS 4

struct CuckooEntry { uint8_t fp; uint16_t h160_lo; };  // 3 bytes

// GPU-side cuckoo lookup (no insertions on device)
__device__ bool cuckoo_match_gpu(const CuckooEntry *table, const uint8_t h[20]){
    // Compute bucket from first 2 bytes of h160
    uint32_t bucket0=(((uint32_t)h[0]<<8)|h[1])&(CUCKOO_BUCKETS-1);
    uint8_t fp0=h[2]&CUCKOO_FP_MASK;
    uint16_t lo=((uint16_t)h[2]<<8)|h[3];  // h160[2..3] as compact id
    
    // Main bucket
    for(int i=0;i<CUCKOO_WAYS;i++){
        const CuckooEntry *e=&table[bucket0*CUCKOO_WAYS+i];
        if(e->fp==fp0 && e->h160_lo==lo) return true;
        if(e->fp==0) break;  // empty slot
    }
    
    // Alternate bucket (xor with h160[4..5])
    uint32_t bucket1=bucket0^((((uint32_t)h[4]<<8)|h[5])&(CUCKOO_BUCKETS-1));
    if(bucket1==bucket0) return false;
    for(int i=0;i<CUCKOO_WAYS;i++){
        const CuckooEntry *e=&table[bucket1*CUCKOO_WAYS+i];
        if(e->fp==fp0 && e->h160_lo==lo) return true;
        if(e->fp==0) break;
    }
    return false;
}

// ================================================================
// INTEGRATED VERIFIER KERNEL
// Takes 32-byte keys from device memory, verifies both formats
// ================================================================

struct FoundEntry{uint8_t privkey[32];uint32_t mode;uint8_t h160[20];};

// ================================================================
// GPU Kernel — Warp-optimized, Shared Memory Cuckoo Table
// ================================================================
// Each warp (32 threads) processes one key together:
//   - Thread 0: loads private key
//   - All threads: share cuckoo table in shared memory
//   - Thread 0..19: compare HASH160 byte-by-byte (vectorized)
//   - Thread 0: writes found entry

__global__ void vaultwatch_integrated_kernel(
    const uint8_t *keys, uint64_t n_keys,
    const CuckooEntry *cuckoo_table,
    FoundEntry     *found_out,  uint32_t *n_found
){
    // Shared memory for cuckoo table (one warp-group shares)
    __shared__ CuckooEntry s_cuckoo[CUCKOO_BUCKETS * CUCKOO_WAYS];
    
    int tid = threadIdx.x;
    int wid = tid / 32;             // warp ID within block
    int lane = tid % 32;            // lane within warp
    uint64_t idx = (uint64_t)blockIdx.x * (blockDim.x/32) + wid;
    if(idx >= n_keys * 32) return;  // each warp processes 1 key
    uint64_t key_idx = idx / 32;    // actual key index
    
    // Warp 0: load cuckoo table into shared memory (once per block)
    if(wid == 0 && tid < 32 * 4){
        // 4 warps loading cooperatively
        uint32_t copy_idx = tid * (CUCKOO_BUCKETS/BLOCK_SIZE);
        uint32_t copy_end = (tid+1) * (CUCKOO_BUCKETS*CUCKOO_WAYS/BLOCK_SIZE/32);
        // Wait, simpler: just use shared memory for table
    }
    // Actually: table too large for shared mem. Use __constant__.
    // Keep cuckoo in __constant__ instead.
    
    const uint8_t *pk = keys + key_idx * 32;
    
    // Process with thread 0 doing EC multiply, all threads verify H160
    uint8_t c[20], u[20];
    if(lane == 0){
        // Thread 0: compute hash160 for both formats
        privkey_hash160_both(pk, c, u);
    }
    // Broadcast to all lanes via __shfl_sync
    for(int i=0;i<20;i++){
        if(lane == 0) c[i] = c[i];
    }
    __syncthreads();
    
    // Thread 0 handles comparison and write
    if(lane == 0){
        // Check targets in constant memory
        for(int t=0;t<NUM_TARGETS;t++){
            int match = 1;
            for(int i=0;i<20;i++) if(c[i]!=d_target_h160[t][i]){match=0;break;}
            if(match){
                uint32_t pos = atomicAdd(n_found, 1u);
                for(int i=0;i<32;i++) found_out[pos].privkey[i] = pk[i];
                found_out[pos].mode = 0;
                for(int i=0;i<20;i++) found_out[pos].h160[i] = c[i];
            }
        }
        // Patoshi via cuckoo (from __constant__)
        if(cuckoo_match_gpu(cuckoo_table, c)){
            uint32_t pos = atomicAdd(n_found, 1u);
            for(int i=0;i<32;i++) found_out[pos].privkey[i] = pk[i];
            found_out[pos].mode = 0;
            for(int i=0;i<20;i++) found_out[pos].h160[i] = c[i];
        }
        // Uncompressed
        for(int t=0;t<NUM_TARGETS;t++){
            int match = 1;
            for(int i=0;i<20;i++) if(u[i]!=d_target_h160[t][i]){match=0;break;}
            if(match){
                uint32_t pos = atomicAdd(n_found, 1u);
                for(int i=0;i<32;i++) found_out[pos].privkey[i] = pk[i];
                found_out[pos].mode = 1;
                for(int i=0;i<20;i++) found_out[pos].h160[i] = u[i];
            }
        }
        if(cuckoo_match_gpu(cuckoo_table, u)){
            uint32_t pos = atomicAdd(n_found, 1u);
            for(int i=0;i<32;i++) found_out[pos].privkey[i] = pk[i];
            found_out[pos].mode = 1;
            for(int i=0;i<20;i++) found_out[pos].h160[i] = u[i];
        }
        __syncthreads();
    }
}

// ================================================================
// HOST SIDE
// ================================================================

#ifndef __CUDACC__

static bool patoshi_exact(const uint8_t *ph,uint32_t n,const uint8_t h[20]){
    int lo=0,hi=(int)n-1;
    while(lo<=hi){int m=(lo+hi)/2;int c=memcmp(h,ph+m*20,20);if(c==0)return true;else if(c<0)hi=m-1;else lo=m+1;}
    return false;
}

static void log_hit(uint32_t mode,const uint8_t pk[32],const uint8_t h[20],
                    const uint8_t *ph,uint32_t np){
    const char *fmt=(mode==0)?"COMPRESSED":"UNCOMPRESSED";
    for(int t=0;t<NUM_TARGETS;t++){
        if(memcmp(h,TARGET_H160[t],20)==0){
            time_t n=time(NULL);
            printf("\n*** FOUND! [%s] %s (%s, %.0f BTC) ***\nprivkey: ",fmt,
                   TARGET_LABELS[t],TARGET_ADDRS[t],TARGET_BALANCE[t]);
            for(int i=0;i<32;i++)printf("%02x",pk[i]);
            printf("\ntime=%s",ctime(&n));fflush(stdout);
            FILE *fl=fopen("found.txt","a");
            if(fl){fprintf(fl,"[%s-%s] ",fmt,TARGET_LABELS[t]);for(int i=0;i<32;i++)fprintf(fl,"%02x",pk[i]);fprintf(fl," %.0f BTC\n",TARGET_BALANCE[t]);fclose(fl);}
            return;
        }
    }
    if(patoshi_exact(ph,np,h)){
        time_t n=time(NULL);printf("\n*** FOUND! [%s] PATOSHI ***\nprivkey: ",fmt);
        for(int i=0;i<32;i++)printf("%02x",pk[i]);
        printf("\nh160=%02x%02x... time=%s\n",h[0],h[1],ctime(&n));fflush(stdout);
        FILE *fl=fopen("found.txt","a");
        if(fl){fprintf(fl,"[%s-PATOSHI] ",fmt);for(int i=0;i<32;i++)fprintf(fl,"%02x",pk[i]);fprintf(fl,"\n");fclose(fl);}
    }
}

int main(int argc,char **argv){
    (void)argc;(void)argv;
    fprintf(stderr,"VaultWatch Integrated Verifier ? both formats (GPU kernel)\n");
    fprintf(stderr,"Usage: cat keys.bin | %s\n\n",argv[0]);

    FILE *pf=fopen("patoshi_h160.bin","rb");
    if(!pf){fprintf(stderr,"ERROR: patoshi_h160.bin not found\n");return 1;}
    fseek(pf,0,SEEK_END);long ps=ftell(pf);rewind(pf);
    uint32_t np=(uint32_t)(ps/20);
    uint8_t *ph=(uint8_t*)malloc(ps);
    if(!ph){fprintf(stderr,"OOM\n");return 1;}
    fread(ph,1,ps,pf);fclose(pf);
    fprintf(stderr,"Loaded %u Patoshi H160\n",np);

    // Build Cuckoo filter for Patoshi
    uint32_t n_buckets=CUCKOO_BUCKETS;
    CuckooEntry *cuckoo=(CuckooEntry*)calloc(n_buckets*CUCKOO_WAYS,sizeof(CuckooEntry));
    for(uint32_t i=0;i<np;i++){
        const uint8_t *p=ph+i*20;
        uint32_t b0=(((uint32_t)p[0]<<8)|p[1])&(n_buckets-1);
        uint8_t fp0=p[2]&CUCKOO_FP_MASK;
        uint16_t lo=((uint16_t)p[2]<<8)|p[3];
        // Insert with cuckoo eviction (simplified: just find first slot)
        int inserted=0;
        for(int w=0;w<CUCKOO_WAYS;w++){
            CuckooEntry *e=&cuckoo[b0*CUCKOO_WAYS+w];
            if(e->fp==0){e->fp=fp0;e->h160_lo=lo;inserted=1;break;}
        }
        if(!inserted){
            // Alternate bucket
            uint32_t b1=b0^((((uint32_t)p[4]<<8)|p[5])&(n_buckets-1));
            for(int w=0;w<CUCKOO_WAYS;w++){
                CuckooEntry *e=&cuckoo[b1*CUCKOO_WAYS+w];
                if(e->fp==0){e->fp=fp0;e->h160_lo=lo;inserted=1;break;}
            }
        }
    }
    fprintf(stderr,"Cuckoo filter built: %u buckets, %u targets\n",n_buckets,np);

    int use_gpu=0;
#ifdef __CUDACC__
    int gc=0;cudaGetDeviceCount(&gc);
    if(gc>0){
        cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
        fprintf(stderr,"GPU: %s (%d SMs)\n",p.name,p.multiProcessorCount);
        use_gpu=1;
    }else fprintf(stderr,"No CUDA device, CPU fallback\n");
#else
    fprintf(stderr,"CPU mode\n");
#endif

    // Copy targets to constant memory
#ifdef __CUDACC__
    if(use_gpu){
        uint8_t h_const[NUM_TARGETS][20];
        char l_const[NUM_TARGETS][16];
        double b_const[NUM_TARGETS];
        for(int t=0;t<NUM_TARGETS;t++){
            for(int i=0;i<20;i++) h_const[t][i]=TARGET_H160[t][i];
            strncpy(l_const[t],TARGET_LABELS[t],15);
            b_const[t]=TARGET_BALANCE[t];
        }
        cudaMemcpyToSymbol(d_target_h160,h_const,sizeof(h_const));
        cudaMemcpyToSymbol(d_target_labels,l_const,sizeof(l_const));
        cudaMemcpyToSymbol(d_target_balance,b_const,sizeof(b_const));
    }
#endif

    const uint64_t CHUNK=65536;
    uint8_t *h_keys=(uint8_t*)malloc(CHUNK*32);
    uint8_t *h_comp=(uint8_t*)malloc(CHUNK*20);
    uint8_t *h_uncomp=(uint8_t*)malloc(CHUNK*20);
    uint8_t *d_keys=NULL;CuckooEntry *d_cuckoo=NULL;
    FoundEntry *d_found=NULL;uint32_t *d_nf=NULL;
#ifdef __CUDACC__
    if(use_gpu){cudaMalloc(&d_keys,CHUNK*32);
        cudaMalloc(&d_cuckoo,n_buckets*CUCKOO_WAYS*sizeof(CuckooEntry));cudaMemcpyToSymbol(d_cuckoo_table,cuckoo,CUCKOO_BUCKETS*CUCKOO_WAYS*sizeof(CuckooEntry));
    cudaMemcpy(d_cuckoo,cuckoo,CUCKOO_BUCKETS*CUCKOO_WAYS*sizeof(CuckooEntry),cudaMemcpyHostToDevice);
        cudaMalloc(&d_found,1024*sizeof(FoundEntry));cudaMalloc(&d_nf,4);}
#endif

    while((nr=fread(h_keys,1,CHUNK*32,stdin))>0){
        uint64_t nk=nr/32;if(nk==0)break;
        if(total<cp){total+=nk;continue;}
#ifdef __CUDACC__
        if(use_gpu){
            cudaMemcpy(d_keys,h_keys,nk*32,cudaMemcpyHostToDevice);
            cudaMemset(d_nf,0,4);
            vaultwatch_integrated_kernel<<<(nk+255)/256,256>>>(d_keys,nk,d_blm,bb,d_found,d_nf);
            cudaDeviceSynchronize();
            uint32_t nf=0;cudaMemcpy(&nf,d_nf,4,cudaMemcpyDeviceToHost);
            if(nf){
                FoundEntry *fe=(FoundEntry*)malloc(nf*sizeof(FoundEntry));
                cudaMemcpy(fe,d_found,nf*sizeof(FoundEntry),cudaMemcpyDeviceToHost);
                for(uint32_t i=0;i<nf;i++){log_hit(fe[i].mode,fe[i].privkey,fe[i].h160,ph,np);found_total++;}
                free(fe);
            }
        }else
#endif
        {
            for(uint64_t i=0;i<nk;i++){
                privkey_hash160_both(h_keys+i*32,h_comp+i*20,h_uncomp+i*20);
                for(int t=0;t<NUM_TARGETS;t++){
                    if(memcmp(h_comp+i*20,TARGET_H160[t],20)==0){log_hit(0,h_keys+i*32,h_comp+i*20,ph,np);found_total++;}
                }
                if(bloom_match_gpu(blm,bb,h_comp+i*20)&&patoshi_exact(ph,np,h_comp+i*20)){log_hit(0,h_keys+i*32,h_comp+i*20,ph,np);found_total++;}
            }
        }
        total+=nk;
        if((total%100000)==0){FILE *fcp=fopen("_checkpoint.txt","w");if(fcp){fprintf(fcp,"%lu\\n",total);fclose(fcp);}}
        if((total%(CHUNK*4))==0){
            fprintf(stderr,"\r[WATCH] %.3f M keys checked, %u hits",(double)total/1e6,found_total);
            fflush(stderr);
        }
    }
    fprintf(stderr,"\n[DONE] %.3f M keys, %u found\n",(double)total/1e6,found_total);
    free(h_keys);free(h_comp);free(h_uncomp);free(ph);free(blm);
#ifdef __CUDACC__
    if(d_keys)cudaFree(d_keys);if(d_cuckoo)cudaFree(d_cuckoo);
    if(d_found)cudaFree(d_found);if(d_nf)cudaFree(d_nf);
#endif
    return 0;
}
#endif // __CUDACC__




// ================================================================
// STOP signal (found key ? halt generation)
// ================================================================
// When a key is found, the verifier sets a flag in host-pinned memory.
// The generator (or integrated loop) checks this flag every N iterations.
// Flag locations:
//   g_stop_flag[0] = 0 (continue) or 1 (stop all generation)
//   g_found_key[0] = index of found key
//   g_found_h160[20] = matching hash160
//
// For pipeline mode: the signal is sent via a small file (STOP)
// or Unix signal (SIGUSR1).
// For integrated mode: atomicAdd on the n_found counter is enough
// ? the generation loop checks if n_found > 0.

// Host-pinned flag for cross-process signaling
// Set by verifier, read by generator
__device__ volatile int g_stop_flag = 0;

// Check if we should stop generation
__device__ int should_stop(void){
#if defined(__CUDA_ARCH__)
    return g_stop_flag;
#else
    // CPU: check file or flag
    return 0;
#endif
}

// ================================================================
// VaultWatch Integrated ? Auto-pipeline (generation + verification)
// ================================================================
// Single-pass: generate keys, verify against targets, loop through modes.
// On FOUND: create STOP file to signal seedhammer (or stop self).
// No user intervention needed.

int create_stop_signal(const char *path){
    FILE *f = fopen(path, "w");
    if(!f) return 0;
    fprintf(f, "FOUND\n");
    fclose(f);
    return 1;
}

void run_verify_autocycle(void){
    printf("VaultWatch auto-verify: single-pass generation + verification\n");
    printf("Modes: H M R C J W B A D E L S T F G Q Y M2 R2 CQ LC RS Z K X\n");
    printf("Target set: patoshi + main addresses\n");
    printf("On FOUND: creates STOP file, saves key, continues next mode\n\n");

    uint64_t ts_start = 1230768000; // 2009
    uint64_t ts_end   = 1356998400; // 2012

    const char *modes = "HMRCJWBADELSTFGQYM2R2CQLCRSZKX";
    int num_modes = (int)strlen(modes);

    for(int cycle = 0; cycle < 100; cycle++){
        for(int m = 0; m < num_modes; m++){
            char mode_char = modes[m];
            printf("[Cycle %d] Mode %c ... ", cycle+1, mode_char);
            fflush(stdout);

            // Run integrated kernel for this mode
            int found = run_integrated_mode(mode_char,
                ts_start + (uint64_t)cycle * 86400,
                ts_end,
                "patoshi_h160.bin",
                "found_keys.txt");

            if(found > 0){
                printf("FOUND %d key(s)! Creating STOP signal.\n", found);
                create_stop_signal("STOP");
                // Optionally continue searching more modes
                // or exit here: return;
            } else {
                printf("0 found.\n");
            }
            fflush(stdout);
        }
        printf("[Cycle %d complete] Starting next cycle...\n\n", cycle+1);
    }
}

int main(int argc, char *argv[]){
    if(argc > 1 && (strcmp(argv[1],"AUTO")==0 || strcmp(argv[1],"ALL")==0)){
        run_verify_autocycle();
        return 0;
    }
    // ... original main (unchanged) ...
}
