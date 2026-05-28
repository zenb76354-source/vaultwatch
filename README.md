# VaultWatch-CUDA 👁️‍🗨️

**GPU private key verifier — written from scratch (no OpenSSL, no libsecp256k1, no borrowed CUDA kernels).**

## What it does

Reads raw 32-byte private keys from stdin → computes EC multiply → SHA256 → RIPEMD160
→ checks against 29,961 targets (8 main + 21,953 Patoshi).

All on **GPU** using CUDA. No CPU bottleneck.

## Build

### GPU (CUDA + NVIDIA GPU)

```bash
nvcc -arch=sm_61 -O3 -std=c++14 -lineinfo -o vaultwatch-cuda vaultwatch-cuda.cu -lcudart -lcuda
```

Change `-arch=sm_61` for your GPU:
- GT 1030 / GTX 1050: `-arch=sm_61`
- RTX 2060+: `-arch=sm_75`
- RTX 3070+: `-arch=sm_86`
- RTX 4090: `-arch=sm_89`

### CPU (no GPU, for testing)

```bash
# MSVC (Windows)
cl /O2 /EHsc vaultwatch-cuda.cu /Fevaultwatch-cpu.exe

# g++ (Linux/macOS)
g++ -O3 -o vaultwatch-cpu vaultwatch-cuda.cu
```

## Usage

```bash
# Pipe from key generator
./seedhammer --out - | ./vaultwatch-cuda

# From binary file
type keys.bin | ./vaultwatch-cuda
# or on Linux:
cat keys.bin | ./vaultwatch-cuda
```

## Pipeline (recommended)

```
SeedHammer  ──stdout──▶  VaultWatch-CUDA  ──found.txt──▶  You
(GPU gen)                  (GPU verify)                    (🎉)
```

## What's inside (all from scratch)

| Component | File | Lines |
|-----------|------|-------|
| 256-bit modular arithmetic | `math256.h` | ~300 |
| secp256k1 EC (Jacobian) | `ec_jacobian.h` | ~360 |
| SHA-256 kernel | `vaultwatch-cuda.cu` | ~80 |
| RIPEMD-160 kernel (5 rounds) | `vaultwatch-cuda.cu` | ~180 |
| CUDA kernel + Bloom filter | `vaultwatch-cuda.cu` | ~200 |
| Main (host code) | `vaultwatch-cuda.cu` | ~150 |

## Targets

- **8 main targets** (~19,860 BTC): addresses from 2009-2010 era
- **21,953 Patoshi** (~1,097,652 BTC): Satoshi's early mining addresses

## Verification

The math has been verified against Python's hashlib + ecdsa:
- SHA256 ✓ matches hashlib vectors
- RIPEMD160 ✓ matches hashlib vectors
- EC pubkey ✓ (priv=1 → `0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798`)
- HASH160 ✓ (pubkey[priv=1] → `751e76e8199196d454941c45d1b3a323f1433bd6` = 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH)

## License

MIT
