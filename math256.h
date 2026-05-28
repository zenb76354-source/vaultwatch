// ================================================================
// VaultWatch 256-bit modular arithmetic for secp256k1 on CUDA
// Written from scratch, no external dependencies
// ================================================================

#ifndef VAULTWATCH_MATH_H
#define VAULTWATCH_MATH_H

#ifdef __CUDACC__
// CUDA device
#define INLINE __host__ __device__ __forceinline__
#define STATIC
#else
// CPU fallback
#define INLINE inline
#define STATIC static
#endif

#include <stdint.h>

// ==================== Representation ====================
// secp256k1 prime p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// n (order)        = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
//
// Numbers: uint64_t[4] little-endian (limb0 = bits 0-63, limb3 = bits 192-255)

#define P0  0xFFFFFC2FULL
#define P1  0xFFFFFFFFFFFFFFFFULL
#define P2  0xFFFFFFFFFFFFFFFFULL
#define P3  0xFFFFFFFFFFFFFFFFULL

// ==================== Core Operations ====================

// Load 4 u64 limbs from memory
INLINE void load256(uint64_t r[4], const uint64_t *src) {
    r[0] = src[0]; r[1] = src[1]; r[2] = src[2]; r[3] = src[3];
}

// Store 4 u64 limbs to memory
INLINE void store256(uint64_t *dst, const uint64_t r[4]) {
    dst[0] = r[0]; dst[1] = r[1]; dst[2] = r[2]; dst[3] = r[3];
}

// Set to zero
INLINE void set_zero(uint64_t r[4]) {
    r[0] = 0; r[1] = 0; r[2] = 0; r[3] = 0;
}

// Copy a -> r
INLINE void copy256(uint64_t r[4], const uint64_t a[4]) {
    r[0] = a[0]; r[1] = a[1]; r[2] = a[2]; r[3] = a[3];
}

// r = (a[0]==b[0] && a[1]==b[1] && a[2]==b[2] && a[3]==b[3])
INLINE bool eq256(const uint64_t a[4], const uint64_t b[4]) {
    return (a[0]==b[0] && a[1]==b[1] && a[2]==b[2] && a[3]==b[3]);
}

// Check if a is zero
INLINE bool is_zero(const uint64_t a[4]) {
    return (a[0]|a[1]|a[2]|a[3]) == 0;
}

// Check if a >= p
INLINE bool ge_p(const uint64_t a[4]) {
    if (a[3] > P3) return true;
    if (a[3] < P3) return false;
    if (a[2] > P2) return true;
    if (a[2] < P2) return false;
    if (a[1] > P1) return true;
    if (a[1] < P1) return false;
    return (a[0] >= P0);
}

// ==================== Modular Addition ====================
// r = (a + b) mod p

INLINE void mod_add(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t carry = 0;

    // Addition without reduction first
    for (int i = 0; i < 4; i++) {
        __uint128_t sum = (__uint128_t)a[i] + b[i] + carry;
        r[i] = (uint64_t)sum;
        carry = (uint64_t)(sum >> 64);
    }

    // If carry or r >= p, subtract p
    if (carry || ge_p(r)) {
        uint64_t borrow = 0;
        // Subtract p = P0 .. P3
        __int128 diff = (__int128)r[0] - P0 - borrow;
        r[0] = (uint64_t)diff; borrow = (diff < 0) ? 1 : 0;
        diff = (__int128)r[1] - P1 - borrow;
        r[1] = (uint64_t)diff; borrow = (diff < 0) ? 1 : 0;
        diff = (__int128)r[2] - P2 - borrow;
        r[2] = (uint64_t)diff; borrow = (diff < 0) ? 1 : 0;
        diff = (__int128)r[3] - P3 - borrow;
        r[3] = (uint64_t)diff;
    }
}

// ==================== Modular Subtraction ====================
// r = (a - b) mod p

INLINE void mod_sub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        __int128 diff = (__int128)a[i] - b[i] - borrow;
        r[i] = (uint64_t)diff;
        borrow = (diff < 0) ? 1 : 0;
    }

    // If borrow, add p back
    if (borrow) {
        uint64_t carry = 0;
        __uint128_t sum = (__uint128_t)r[0] + P0 + carry;
        r[0] = (uint64_t)sum; carry = (uint64_t)(sum >> 64);
        sum = (__uint128_t)r[1] + P1 + carry;
        r[1] = (uint64_t)sum; carry = (uint64_t)(sum >> 64);
        sum = (__uint128_t)r[2] + P2 + carry;
        r[2] = (uint64_t)sum; carry = (uint64_t)(sum >> 64);
        sum = (__uint128_t)r[3] + P3 + carry;
        r[3] = (uint64_t)sum;
    }
}

// ==================== Modular Negation ====================
// r = (-a) mod p

INLINE void mod_neg(uint64_t r[4], const uint64_t a[4]) {
    uint64_t zero[4] = {0,0,0,0};
    mod_sub(r, zero, a);
}

// ==================== 256-bit Shift Right by 1 ====================
// r = a >> 1 (arithmetic for large numbers)

INLINE void shr1(uint64_t r[4], const uint64_t a[4]) {
    r[0] = (a[0] >> 1) | (a[1] << 63);
    r[1] = (a[1] >> 1) | (a[2] << 63);
    r[2] = (a[2] >> 1) | (a[3] << 63);
    r[3] = a[3] >> 1;
}

// ==================== Modular Multiplication ====================
// r = (a * b) mod p
// Standard schoolbook: compute full 512-bit product, then reduce
// Reduction using secp256k1 property:
//   2^256 mod p = 2^32 + 977 = 0x1000003D1
//   So upper 256 bits * 0x1000003D1 + lower 256 bits, then reduce

INLINE void mod_mul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    // Full 512-bit product: p[i] = a * b (8 limbs)
    __uint128_t p[8] = {0,0,0,0,0,0,0,0};

    // Schoolbook multiplication
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            __uint128_t prod = (__uint128_t)a[i] * b[j] + p[i+j] + carry;
            p[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        p[i+4] += carry;
    }

    // Carry propagation only if needed (p[4]..p[7] may overflow)
    for (int i = 4; i < 7; i++) {
        if (p[i] > 0xFFFFFFFFFFFFFFFFULL) {
            p[i+1] += p[i] >> 64;
            p[i] &= 0xFFFFFFFFFFFFFFFFULL;
        }
    }

    // Now reduce:
    // Let L = lower 256 bits = p[0..3]
    // Let U = upper 256 bits = p[4..7]
    // result = L + U * 0x1000003D1  (mod p)
    // 0x1000003D1 = 2^32 + 977
    // So U * 0x1000003D1 = U * 2^32 + U * 977
    // U * 2^32 = (U << 32)  (shift left by 32 within 256 bits)

    uint64_t L[4] = {(uint64_t)p[0], (uint64_t)p[1], (uint64_t)p[2], (uint64_t)p[3]};
    uint64_t U[4] = {(uint64_t)p[4], (uint64_t)p[5], (uint64_t)p[6], (uint64_t)p[7]};

    // Step 1: add L + U*977 (lower part)
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        __uint128_t prod = (__uint128_t)U[i] * 977ULL;
        __uint128_t sum = (__uint128_t)L[i] + (uint64_t)prod + carry;
        r[i] = (uint64_t)sum;
        carry = (uint64_t)(prod >> 64) + (uint64_t)(sum >> 64);
    }
    // carry now may be > 0 due to U*977 overflow

    // Step 2: add (U << 32) + carry
    uint64_t U_shifted[4];
    U_shifted[0] = U[0] << 32;
    U_shifted[1] = (U[0] >> 32) | (U[1] << 32);
    U_shifted[2] = (U[1] >> 32) | (U[2] << 32);
    U_shifted[3] = (U[2] >> 32);

    for (int i = 0; i < 4; i++) {
        __uint128_t sum = (__uint128_t)r[i] + U_shifted[i] + carry;
        r[i] = (uint64_t)sum;
        carry = (uint64_t)(sum >> 64);
    }

    // Step 3: reduce if carry or r >= p
    // We may need up to 2 subtractions
    for (int k = 0; k < 2; k++) {
        if (!carry && !ge_p(r)) break;

        __int128 diff;
        uint64_t borrow = 0;
        diff = (__int128)r[0] - (carry ? 0 : P0) - borrow;
        r[0] = (uint64_t)diff; borrow = (diff < 0) ? 1 : 0;
        diff = (__int128)r[1] - (carry ? (uint64_t)(-1) : P1) - borrow;
        r[1] = (uint64_t)diff; borrow = (diff < 0) ? 1 : 0;
        diff = (__int128)r[2] - (carry ? (uint64_t)(-1) : P2) - borrow;
        r[2] = (uint64_t)diff; borrow = (diff < 0) ? 1 : 0;
        diff = (__int128)r[3] - (carry ? (uint64_t)(-1) : P3) - borrow;
        r[3] = (uint64_t)diff; borrow = (diff < 0) ? 1 : 0;

        // If still underflow (borrow true), add p back
        if (borrow) {
            __uint128_t sum;
            uint64_t carry2 = 0;
            sum = (__uint128_t)r[0] + P0 + carry2;
            r[0] = (uint64_t)sum; carry2 = (uint64_t)(sum >> 64);
            sum = (__uint128_t)r[1] + P1 + carry2;
            r[1] = (uint64_t)sum; carry2 = (uint64_t)(sum >> 64);
            sum = (__uint128_t)r[2] + P2 + carry2;
            r[2] = (uint64_t)sum; carry2 = (uint64_t)(sum >> 64);
            sum = (__uint128_t)r[3] + P3 + carry2;
            r[3] = (uint64_t)sum;
        }

        carry = 0; // after subtraction, no more carry
    }
}

// ==================== Modular Square ====================
// r = a^2 mod p (faster than mod_mul)

INLINE void mod_sqr(uint64_t r[4], const uint64_t a[4]) {
    mod_mul(r, a, a);
}

// ==================== Modular Inverse ====================
// r = a^(-1) mod p  using Fermat: a^(p-2) mod p
// p-2 = 0xFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
// Square-and-multiply from MSB to LSB of (p-2)

INLINE void mod_inv(uint64_t r[4], const uint64_t a[4]) {
    if (is_zero(a)) {
        set_zero(r);
        return;
    }

    // (p-2) as 4 little-endian limbs
    // p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    // p-2 = same but minus 2 => last bytes: ...FC2D
    uint64_t pm2[4];
    pm2[0] = 0xFFFFFFFEFFFFFC2DULL;
    pm2[1] = 0xFFFFFFFFFFFFFFFFULL;
    pm2[2] = 0xFFFFFFFFFFFFFFFFULL;
    pm2[3] = 0xFFFFFFFFFFFFFFFFULL;

    // Square-and-multiply: result = a^(pm2)
    // Standard: result = 1, then for each bit MSB to LSB:
    //   result = result^2
    //   if bit==1: result = result * a
    uint64_t result[4];
    result[0] = 1; result[1] = 0; result[2] = 0; result[3] = 0;  // start with 1

    // pm2 bits: bit 255 is 0 (pm2[3] has all bits set, but MSB bit 255 is the top bit of 0xFFFFFFFFFFFFFFFF = 1)
    // Actually bit 255 of pm2 = 1 (because pm2[3]=0xFFFFFFFFFFFFFFFF, bit 63 of that is 1)
    // So we start with bit 254 (the next)

    // First handle bit 255 separately (best to just square once)
    mod_sqr(result, result);  // 1^2 = 1

    for (int bit = 254; bit >= 0; bit--) {
        mod_sqr(result, result);

        int limb = bit / 64;
        int pos  = bit % 64;
        if ((pm2[limb] >> pos) & 1) {
            mod_mul(result, result, a);
        }
    }

    copy256(r, result);
}

#endif // VAULTWATCH_MATH_H
