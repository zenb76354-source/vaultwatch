// ================================================================
// VaultWatch-CUDA — GPU private key verifier (FULL)
// Reads private keys (32-byte binary) from stdin.
// For each key: EC multiply → pubkey → SHA256 → RIPEMD160 → compare
// Whole pipeline on GPU. No CPU bottlenecks.
// ================================================================

// ================================================================
// CPU/GPU select:
//   - MSVC/g++: compiles as CPU fallback (no __CUDACC__ defined)
//   - nvcc: compiles with full GPU kernel (__CUDACC__ defined)
// ================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

// Our math libraries (written from scratch)
#include "math256.h"
#include "ec_jacobian.h"

// Targets
#include "targets.h"
#include "patoshi_targets.h"

// ==================== CUDA / CPU selection ====================
// On CUDA: __device__ for kernel-callable functions
// On CPU: static for host functions

#ifdef __CUDACC__
#define D_FUNC __device__
#define D_CONST __constant__
#define HD_FUNC __host__ __device__
#else
#define D_FUNC /* nothing */
#define D_CONST /* nothing */
#define HD_FUNC static
#endif

// ==================== SHA-256 ====================

#define ROTL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

D_CONST uint32_t SHA_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

D_FUNC void sha256_compress(uint32_t state[8], const uint32_t block[16]) {
    uint32_t W[64];
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3];
    uint32_t e=state[4],f=state[5],g=state[6],h=state[7],t1,t2;
    for (int i=0;i<16;i++) W[i]=block[i];
    for (int i=16;i<64;i++) {
        uint32_t s0=ROTL(W[i-15],7)^ROTL(W[i-15],18)^(W[i-15]>>3);
        uint32_t s1=ROTL(W[i-2],17)^ROTL(W[i-2],19)^(W[i-2]>>10);
        W[i]=W[i-16]+s0+W[i-7]+s1;
    }
    for (int i=0;i<64;i++) {
        uint32_t S1=ROTL(e,6)^ROTL(e,11)^ROTL(e,25);
        uint32_t ch=(e&f)^((~e)&g);
        t1=h+S1+ch+SHA_K[i]+W[i];
        uint32_t S0=ROTL(a,2)^ROTL(a,13)^ROTL(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c);
        t2=S0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
    state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

D_FUNC void sha256(const uint8_t *data, uint32_t len, uint8_t hash[32]) {
    uint32_t state[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint32_t block[16];
    uint32_t pos=0;
    for (;pos+64<=len;pos+=64){
        for(int i=0;i<16;i++) block[i]=(data[pos+i*4]<<24)|(data[pos+i*4+1]<<16)|(data[pos+i*4+2]<<8)|data[pos+i*4+3];
        sha256_compress(state,block);
    }
    uint8_t last[128];
    uint32_t rem=len-pos;
    for(uint32_t i=0;i<rem;i++)last[i]=data[pos+i];
    last[rem]=0x80;
    uint32_t last_len=rem+1;
    if(last_len>56){while(last_len<128)last[last_len++]=0;
        for(int i=0;i<16;i++)block[i]=(last[i*4]<<24)|(last[i*4+1]<<16)|(last[i*4+2]<<8)|last[i*4+3];
        sha256_compress(state,block);last_len=0;
    }
    while(last_len<56)last[last_len++]=0;
    uint64_t bitlen=(uint64_t)len*8;
    last[56]=(uint8_t)(bitlen>>56);last[57]=(uint8_t)(bitlen>>48);last[58]=(uint8_t)(bitlen>>40);
    last[59]=(uint8_t)(bitlen>>32);last[60]=(uint8_t)(bitlen>>24);last[61]=(uint8_t)(bitlen>>16);
    last[62]=(uint8_t)(bitlen>>8);last[63]=(uint8_t)(bitlen);
    for(int i=0;i<16;i++)block[i]=(last[i*4]<<24)|(last[i*4+1]<<16)|(last[i*4+2]<<8)|last[i*4+3];
    sha256_compress(state,block);
    for(int i=0;i<8;i++){hash[i*4]=(uint8_t)(state[i]>>24);hash[i*4+1]=(uint8_t)(state[i]>>16);
                          hash[i*4+2]=(uint8_t)(state[i]>>8);hash[i*4+3]=(uint8_t)(state[i]);}
}

// ==================== RIPEMD-160 (full 5-round, RFC compliant) ====================

D_FUNC void ripemd160(const uint8_t *data, uint32_t len, uint8_t hash[20]) {
    uint32_t state[5]={0x67452301,0xefcdab89,0x98badcfe,0x10325476,0xc3d2e1f0};
    uint32_t block[16];
    uint32_t pos=0;

    // RIPEMD-160 round constants
    const uint32_t rk[5] = {0x00000000,0x5a827999,0x6ed9eba1,0x8f1bbcdc,0xa953fd4e};
    const uint32_t rkp[5]= {0x50a28be6,0x5c4dd124,0x6d703ef3,0x7a6d76e9,0x00000000};

    // Message order for rounds 1-5
    const int ro[5][16] = {
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
        {7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8},
        {3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12},
        {1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2},
        {4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13}
    };

    // Message order for parallel rounds 1-5 (right side)
    const int rop[5][16] = {
        {5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12},
        {6,11,3,7,14,9,1,4,12,0,15,5,10,2,13,8},
        {15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13},
        {8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14},
        {12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11}
    };

    // Rotation amounts for rounds 1-5
    const int rs[5][16] = {
        {11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8},
        {7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12},
        {11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5},
        {11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12},
        {9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6}
    };

    // Rotation amounts for parallel side
    const int rsp[5][16] = {
        {8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6},
        {9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11},
        {9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5},
        {15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8},
        {8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11}
    };

    for(;pos+64<=len;pos+=64){
        for(int i=0;i<16;i++)
            block[i]=((uint32_t)data[pos+i*4]<<24)|((uint32_t)data[pos+i*4+1]<<16)|
                     ((uint32_t)data[pos+i*4+2]<<8)|(uint32_t)data[pos+i*4+3];

        uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4];
        uint32_t ap=a,bp=b,cp=c,dp=d,ep=e;

        // 5 rounds, 16 steps each = 80 steps
        // RIPEMD-160 functions (ISO/IEC 10118-3):
        //   Round 0: f = x^y^z          (left), fp = x^(y|~z) = f5 (right)
        //   Round 1: f = (x&y)|(~x&z)   (left), fp = (x&z)|(y&~z) = f4 (right)
        //   Round 2: f = (x|~y)^z       (left), fp = x^y^z = f1 (right)
        //   Round 3: f = (x&z)|(y&~z)   (left), fp = (x&y)|(~x&z) = f2 (right)
        //   Round 4: f = x^(y|~z)       (left), fp = (x|~y)^z = f3 (right)
        for(int r=0;r<5;r++){
            for(int s=0;s<16;s++){
                // Left side
                uint32_t f;
                if(r==0) f = (b ^ c ^ d);                              // f1
                else if(r==1) f = (b & c) | (~b & d);                  // f2
                else if(r==2) f = (b | ~c) ^ d;                        // f3
                else if(r==3) f = (b & d) | (c & ~d);                  // f4
                else f = b ^ (c | ~d);                                  // f5

                uint32_t T = ROTL(a + f + block[ro[r][s]] + rk[r], rs[r][s]) + e;
                // Rotate state
                e = d; d = ROTL(c,10); c = b; b = a; a = T;

                // Right side
                uint32_t fp;
                if(r==0) fp = bp ^ (cp | ~dp);                          // fp1 = f5
                else if(r==1) fp = (bp & dp) | (cp & ~dp);              // fp2 = f4
                else if(r==2) fp = bp ^ cp ^ dp;                        // fp3 = f1
                else if(r==3) fp = (bp & cp) | (~bp & dp);              // fp4 = f2
                else fp = (bp | ~cp) ^ dp;                               // fp5 = f3

                T = ROTL(ap + fp + block[rop[r][s]] + rkp[r], rsp[r][s]) + ep;
                ep = dp; dp = ROTL(cp,10); cp = bp; bp = ap; ap = T;
            }
        }

        // Final combining
        uint32_t tmp = state[1] + c + dp;
        state[1] = state[2] + d + ep;
        state[2] = state[3] + e + ap;
        state[3] = state[4] + a + bp;
        state[4] = state[0] + b + cp;
        state[0] = tmp;
    }

    // Padding (same as before, then compress once more)
    uint8_t last[128];
    uint32_t rem=len-pos;
    for(uint32_t i=0;i<rem;i++)last[i]=data[pos+i];
    last[rem]=0x80; uint32_t last_len=rem+1;

    if(last_len>56) {
        while(last_len<128) last[last_len++]=0;
        for(int i=0;i<16;i++)
            block[i]=((uint32_t)last[i*4]<<24)|((uint32_t)last[i*4+1]<<16)|((uint32_t)last[i*4+2]<<8)|last[i*4+3];
        uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4];
        uint32_t ap=a,bp=b,cp=c,dp=d,ep=e;
        for(int r=0;r<5;r++){
            for(int s=0;s<16;s++){
                uint32_t f, fp;
                if(r==0){f=b^c^d; fp=bp^(cp|~dp);}                         // f1, fp1=f5
                else if(r==1){f=(b&c)|(~b&d); fp=(bp&dp)|(cp&~dp);}         // f2, fp2=f4
                else if(r==2){f=(b|~c)^d; fp=bp^cp^dp;}                     // f3, fp3=f1
                else if(r==3){f=(b&d)|(c&~d); fp=(bp&cp)|(~bp&dp);}         // f4, fp4=f2
                else{f=b^(c|~d); fp=(bp|~cp)^dp;}                           // f5, fp5=f3
                uint32_t T=ROTL(a+f+block[ro[r][s]]+rk[r],rs[r][s])+e;
                e=d;d=ROTL(c,10);c=b;b=a;a=T;
                T=ROTL(ap+fp+block[rop[r][s]]+rkp[r],rsp[r][s])+ep;
                ep=dp;dp=ROTL(cp,10);cp=bp;bp=ap;ap=T;
            }
        }
        uint32_t tmp=state[1]+c+dp;
        state[1]=state[2]+d+ep;state[2]=state[3]+e+ap;
        state[3]=state[4]+a+bp;state[4]=state[0]+b+cp;
        state[0]=tmp;
        last_len=0;
    }
    while(last_len<56)last[last_len++]=0;
    uint64_t bitlen=(uint64_t)len*8;
    last[56]=(uint8_t)(bitlen>>56);last[57]=(uint8_t)(bitlen>>48);last[58]=(uint8_t)(bitlen>>40);
    last[59]=(uint8_t)(bitlen>>32);last[60]=(uint8_t)(bitlen>>24);last[61]=(uint8_t)(bitlen>>16);
    last[62]=(uint8_t)(bitlen>>8);last[63]=(uint8_t)(bitlen);
    for(int i=0;i<16;i++)
        block[i]=((uint32_t)last[i*4]<<24)|((uint32_t)last[i*4+1]<<16)|((uint32_t)last[i*4+2]<<8)|last[i*4+3];
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4];
    uint32_t ap=a,bp=b,cp=c,dp=d,ep=e;
    for(int r=0;r<5;r++){
        for(int s=0;s<16;s++){
            uint32_t f, fp;
            if(r==0){f=b^c^d; fp=bp^(cp|~dp);}
            else if(r==1){f=(b&c)|(~b&d); fp=(bp&dp)|(cp&~dp);}
            else if(r==2){f=(b|~c)^d; fp=bp^cp^dp;}
            else if(r==3){f=(b&d)|(c&~d); fp=(bp&cp)|(~bp&dp);}
            else{f=b^(c|~d); fp=(bp|~cp)^dp;}
            uint32_t T=ROTL(a+f+block[ro[r][s]]+rk[r],rs[r][s])+e;
            e=d;d=ROTL(c,10);c=b;b=a;a=T;
            T=ROTL(ap+fp+block[rop[r][s]]+rkp[r],rsp[r][s])+ep;
            ep=dp;dp=ROTL(cp,10);cp=bp;bp=ap;ap=T;
        }
    }
    uint32_t tmp=state[1]+c+dp;
    state[1]=state[2]+d+ep;state[2]=state[3]+e+ap;
    state[3]=state[4]+a+bp;state[4]=state[0]+b+cp;
    state[0]=tmp;

    for(int i=0;i<5;i++){
        hash[i*4]  =(uint8_t)(state[i]&0xFF);
        hash[i*4+1]=(uint8_t)((state[i]>>8)&0xFF);
        hash[i*4+2]=(uint8_t)((state[i]>>16)&0xFF);
        hash[i*4+3]=(uint8_t)((state[i]>>24)&0xFF);
    }
}

// ==================== BLOOM FILTER (CPU only) ====================

struct BloomFilter {
    uint8_t *bits;
    uint32_t nbits;
    uint32_t mask;
};

static BloomFilter *bloom_new(uint32_t nbits) {
    BloomFilter *bf = (BloomFilter*)malloc(sizeof(BloomFilter));
    if (!bf) return NULL;
    bf->nbits = nbits;
    bf->mask = nbits - 1;
    size_t bytes = (nbits + 7) / 8;
    bf->bits = (uint8_t*)calloc(1, bytes);
    return bf;
}

static void bloom_add(BloomFilter *bf, const uint8_t h160[20]) {
    uint32_t m = bf->mask;
    uint32_t h[7] = {
        ((uint32_t)h160[0]<<24|(uint32_t)h160[1]<<16|(uint32_t)h160[2]<<8|h160[3]) & m,
        ((uint32_t)h160[4]<<24|(uint32_t)h160[5]<<16|(uint32_t)h160[6]<<8|h160[7]) & m,
        ((uint32_t)h160[8]<<24|(uint32_t)h160[9]<<16|(uint32_t)h160[10]<<8|h160[11]) & m,
        ((uint32_t)h160[12]<<24|(uint32_t)h160[13]<<16|(uint32_t)h160[14]<<8|h160[15]) & m,
        ((uint32_t)h160[16]<<24|(uint32_t)h160[17]<<16|(uint32_t)h160[18]<<8|h160[19]) & m,
        ((h160[0]*2654435761u + h160[1]*2246822519u + h160[2]) & m),
        ((h160[3]*3266489917u + h160[4]*668265263u + h160[5]) & m)
    };
    for (int i = 0; i < 7; i++) bf->bits[h[i] >> 3] |= (1 << (h[i] & 7));
}

static bool bloom_test(BloomFilter *bf, const uint8_t h160[20]) {
    uint32_t m = bf->mask;
    uint32_t h[7] = {
        ((uint32_t)h160[0]<<24|(uint32_t)h160[1]<<16|(uint32_t)h160[2]<<8|h160[3]) & m,
        ((uint32_t)h160[4]<<24|(uint32_t)h160[5]<<16|(uint32_t)h160[6]<<8|h160[7]) & m,
        ((uint32_t)h160[8]<<24|(uint32_t)h160[9]<<16|(uint32_t)h160[10]<<8|h160[11]) & m,
        ((uint32_t)h160[12]<<24|(uint32_t)h160[13]<<16|(uint32_t)h160[14]<<8|h160[15]) & m,
        ((uint32_t)h160[16]<<24|(uint32_t)h160[17]<<16|(uint32_t)h160[18]<<8|h160[19]) & m,
        ((h160[0]*2654435761u + h160[1]*2246822519u + h160[2]) & m),
        ((h160[3]*3266489917u + h160[4]*668265263u + h160[5]) & m)
    };
    for (int i = 0; i < 7; i++) if (!(bf->bits[h[i] >> 3] & (1 << (h[i] & 7)))) return false;
    return true;
}

static void bloom_free(BloomFilter *bf) {
    if (bf) { free(bf->bits); free(bf); }
}

// ==================== CPU-side helpers (used outside the kernel) ====================

static void privkey_hash160_both_host(const uint8_t priv[32], 
                                       uint8_t h160_comp[20],
                                       uint8_t h160_uncomp[20]) {
    uint8_t pub_comp[33];
    uint8_t pub_uncomp[65];
    privkey_to_pubkey_both(priv, pub_comp, pub_uncomp);
    uint8_t sha[32];
    sha256(pub_comp, 33, sha);
    ripemd160(sha, 32, h160_comp);
    sha256(pub_uncomp, 65, sha);
    ripemd160(sha, 32, h160_uncomp);
}

static bool patoshi_exact_check(const uint8_t *patoshi_h160s, uint32_t n_patoshi,
                                const uint8_t h160[20]) {
    int lo = 0, hi = (int)n_patoshi - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = memcmp(h160, patoshi_h160s + mid * 20, 20);
        if (cmp == 0) return true;
        else if (cmp < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return false;
}

// ==================== ============================================================
// ==================== CUDA KERNEL ===============================================
// ==================== ============================================================

#ifdef __CUDACC__

D_FUNC void privkey_hash160_both(const uint8_t priv[32], uint8_t h160_comp[20], uint8_t h160_uncomp[20]) {
    uint8_t pub_comp[33];
    uint8_t pub_uncomp[65];
    
    // Compute EC multiply
    privkey_to_pubkey_both(priv, pub_comp, pub_uncomp);
    
    // HASH160 for compressed
    uint8_t sha32[32];
    sha256(pub_comp, 33, sha32);
    ripemd160(sha32, 32, h160_comp);
    
    // HASH160 for uncompressed
    sha256(pub_uncomp, 65, sha32);
    ripemd160(sha32, 32, h160_uncomp);
}

__global__ void vaultwatch_kernel(
    const uint8_t *keys,
    uint64_t       n_keys,
    uint8_t       *h160_comp_out,
    uint8_t       *h160_uncomp_out
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_keys) return;

    privkey_hash160_both(keys + idx * 32, 
                         h160_comp_out + idx * 20,
                         h160_uncomp_out + idx * 20);
}

#endif // __CUDACC__

// ==================== ============================================================
// ==================== MAIN =======================================================
// ==================== ============================================================

static void log_found(const char *label, const uint8_t pk[32]) {
    time_t t = time(NULL);
    fprintf(stdout, "\n*** FOUND! %s ***\nprivkey: ", label);
    for (int i = 0; i < 32; i++) fprintf(stdout, "%02x", pk[i]);
    fprintf(stdout, "\ntime=%s", ctime(&t));
    fflush(stdout);

    FILE *fl = fopen("found.txt", "a");
    if (fl) {
        fprintf(fl, "[%s] key=", label);
        for (int i = 0; i < 32; i++) fprintf(fl, "%02x", pk[i]);
        fprintf(fl, "\n");
        fclose(fl);
    }
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    int check_both = 1; // always check both compressed + uncompressed

    fprintf(stderr, "VaultWatch-CUDA — verifier (compressed + uncompressed)\n");

    // Load Patoshi H160s
    FILE *pf = fopen("patoshi_h160.bin", "rb");
    if (!pf) { fprintf(stderr, "ERROR: patoshi_h160.bin not found\n"); return 1; }
    fseek(pf, 0, SEEK_END);
    long pf_size = ftell(pf);
    rewind(pf);
    uint32_t n_patoshi = pf_size / 20;
    uint8_t *patoshi_h160s = (uint8_t*)malloc(pf_size);
    if (!patoshi_h160s) { fprintf(stderr, "OOM\n"); return 1; }
    fread(patoshi_h160s, 1, pf_size, pf);
    fclose(pf);
    fprintf(stderr, "Loaded %u Patoshi H160 targets\n", n_patoshi);

    // Build bloom filter (for Patoshi)
    uint32_t bloom_bits = 1;
    while (bloom_bits < n_patoshi * 20) bloom_bits <<= 1;
    BloomFilter *bf = bloom_new(bloom_bits);
    for (uint32_t i = 0; i < n_patoshi; i++)
        bloom_add(bf, patoshi_h160s + i * 20);
    fprintf(stderr, "Bloom filter: %u bits\n", bloom_bits);

    // CUDA init
    int use_gpu = 0;
#ifdef __CUDACC__
    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    if (gpu_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        fprintf(stderr, "GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
        use_gpu = 1;
    } else {
        fprintf(stderr, "No CUDA device, CPU fallback\n");
    }
#else
    fprintf(stderr, "CPU mode (no CUDA compiler)\n");
#endif

    // Buffers: each key → 2 HASH160 outputs (comp + uncomp)
    const uint64_t BATCH = 65536;
    uint8_t *h_keys = (uint8_t*)malloc(BATCH * 32);
    uint8_t *h_comp = (uint8_t*)malloc(BATCH * 20);
    uint8_t *h_uncomp = (uint8_t*)malloc(BATCH * 20);
    uint8_t *d_keys = NULL;
    uint8_t *d_comp = NULL;
    uint8_t *d_uncomp = NULL;

#ifdef __CUDACC__
    if (use_gpu) {
        cudaMalloc(&d_keys, BATCH * 32);
        cudaMalloc(&d_comp, BATCH * 20);
        cudaMalloc(&d_uncomp, BATCH * 20);
    }
#endif

    uint64_t total = 0;
    uint32_t main_hits = 0, patoshi_hits = 0;
    uint32_t main_uncomp_hits = 0, patoshi_uncomp_hits = 0;
    size_t nread;

    fprintf(stderr, "Reading binary keys from stdin...\n");

    while ((nread = fread(h_keys, 1, BATCH * 32, stdin)) > 0) {
        uint64_t this_batch = nread / 32;
        if (this_batch == 0) break;

#ifdef __CUDACC__
        if (use_gpu) {
            cudaMemcpy(d_keys, h_keys, this_batch * 32, cudaMemcpyHostToDevice);
            cudaMemset(d_comp, 0, this_batch * 20);
            cudaMemset(d_uncomp, 0, this_batch * 20);
            vaultwatch_kernel<<<(this_batch+255)/256, 256>>>(
                d_keys, this_batch, d_comp, d_uncomp);
            cudaDeviceSynchronize();
            cudaMemcpy(h_comp, d_comp, this_batch * 20, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uncomp, d_uncomp, this_batch * 20, cudaMemcpyDeviceToHost);
        } else
#endif
        {
            // CPU fallback — compute BOTH formats
            for (uint64_t i = 0; i < this_batch; i++)
                privkey_hash160_both_host(h_keys + i * 32,
                                          h_comp + i * 20,
                                          h_uncomp + i * 20);
        }

        // Check vs targets (host side) — BOTH formats!
        for (uint64_t i = 0; i < this_batch; i++) {
            uint8_t *pk = h_keys + i * 32;

            // --- Compressed ---
            uint8_t *h160_c = h_comp + i * 20;
            for (int t = 0; t < NUM_TARGETS; t++) {
                if (memcmp(h160_c, TARGET_H160[t], 20) == 0) {
                    char label[64];
                    snprintf(label, 64, "[COMPRESSED] %s (%s, %.0f BTC)",
                             TARGET_LABELS[t], TARGET_ADDRS[t], TARGET_BALANCE[t]);
                    log_found(label, pk);
                    main_hits++;
                    break;
                }
            }
            if (bloom_test(bf, h160_c) && patoshi_exact_check(patoshi_h160s, n_patoshi, h160_c)) {
                log_found("[COMPRESSED] PATOSHI", pk);
                patoshi_hits++;
            }

            // --- Uncompressed ---
            uint8_t *h160_u = h_uncomp + i * 20;
            for (int t = 0; t < NUM_TARGETS; t++) {
                if (memcmp(h160_u, TARGET_H160[t], 20) == 0) {
                    char label[64];
                    snprintf(label, 64, "[UNCOMPRESSED] %s (%s, %.0f BTC)",
                             TARGET_LABELS[t], TARGET_ADDRS[t], TARGET_BALANCE[t]);
                    log_found(label, pk);
                    main_uncomp_hits++;
                    break;
                }
            }
            if (bloom_test(bf, h160_u) && patoshi_exact_check(patoshi_h160s, n_patoshi, h160_u)) {
                log_found("[UNCOMPRESSED] PATOSHI", pk);
                patoshi_uncomp_hits++;
            }
        }

        total += this_batch;
        if ((total % (BATCH * 4)) == 0) {
            fprintf(stderr, "\r[WATCH] %.3f M keys | cmp: %u/%u | unc: %u/%u",
                    (double)total / 1e6, 
                    main_hits, patoshi_hits,
                    main_uncomp_hits, patoshi_uncomp_hits);
            fflush(stderr);
        }
    }

    fprintf(stderr, "\n[DONE] %.3f M keys checked | compressed: %u main + %u patoshi | uncompressed: %u main + %u patoshi\n",
            (double)total / 1e6,
            main_hits, patoshi_hits,
            main_uncomp_hits, patoshi_uncomp_hits);

    free(h_keys); free(h_comp); free(h_uncomp);
    free(patoshi_h160s);
    bloom_free(bf);
#ifdef __CUDACC__
    if (d_keys) cudaFree(d_keys);
    if (d_comp) cudaFree(d_comp);
    if (d_uncomp) cudaFree(d_uncomp);
#endif
    return 0;
}
