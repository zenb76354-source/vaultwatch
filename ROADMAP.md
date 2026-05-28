# ROADMAP: Vulnerability Coverage

✅ Done | 🟡 In SeedHammer | 🟢 Deferred

| Vuln / Method | Status | Component |
|:-------------|:------:|:---------|
| Timestamp H36 | ✅ | SeedHammer (H1-H14) |
| MWC1616 | 🟡 | SeedHammer mode M |
| Randstorm/JSBN | 🟡 | SeedHammer mode R |
| Debian OpenSSL | 🟡 | SeedHammer H28 |
| Brainwallet dict | 🟡 | SeedHammer H16-H27 |
| Compressed pubkeys | ✅ | VaultWatch |
| Uncompressed pubkeys | ✅ | VaultWatch |
| Integrated kernel | ✅ | VaultWatch (vaultwatch-integrated.cu) |
| Nonce reuse | 🟢 | Needs TX indexer |

## Structure

- **SeedHammer** → GPU key generator (H36, MWC, Randstorm, Debian, Brainwallet)
- **VaultWatch** → GPU verifier (compressed + uncompressed HASH160, bloom filter)
