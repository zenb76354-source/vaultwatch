# VaultWatch 👁️‍🗨️

CPU-based Bitcoin private key verifier.
Reads raw 32-byte private keys from a binary file (or stdin), computes the full
ECC pipeline (pubkey → SHA256 → RIPEMD160), and compares against known target
addresses.

## Philosophy

> "Watch the vault. Let hammers be hammers."

VaultWatch **only verifies**. It does not:
- Generate keys
- Use a GPU
- Do anything clever

It uses **secp256k1 + OpenSSL** — the same libraries every Bitcoin node uses.
If VaultWatch says "found", it's found.

## Build

Requires: `libsecp256k1-dev`, `libssl-dev`, g++

```bash
git clone https://github.com/you/vaultwatch
cd vaultwatch
make
```

## Usage

```bash
# Check a file
./vaultwatch --keys keys.bin

# Pipe from SeedHammer
./seedhammer --mode h36 --start 1223424000000 --count 50000000 --out - | ./vaultwatch --pipe

# Check specific hypothesis
./vaultwatch --keys keys.bin --hypothesis H36
```

## Input Format

Raw binary: each key is **32 bytes** (big-endian uint256).
Concatenated sequentially. File size must be multiple of 32.

## Targets

Eight addresses from the 2009-2010 era, ~$150M+ at peak:

| # | Label | First TX | Balance |
|:-:|:-----:|:--------:|:------:|
| 1 | A1 | 2010-03-16 | 400 BTC |
| 2 | A2* | 2020-12-19 | 9260 BTC |
| 3 | A3 | 2010-07-15 | 400 BTC |
| 4 | A4 | 2010-07-15 | 200 BTC |
| 5 | A5 | 2010-07-17 | 200 BTC |
| 6 | A6 | 2010-09-10 | 1200 BTC |
| 7 | A7 | 2010-09-16 | 200 BTC |
| 8 | E1 | 2009 | 250 BTC |

*A2 is a known dust collector (active 2020+) — included anyway.*

## On FOUND

Writes to `found.txt`:
```
[H36] FOUND! Target A1 (12rMpw5...)
privkey: 0000000000000000000000000000000000000000000000000000000000000001
```

Also prints to stderr for piping/logging.

## License

MIT
