# VaultWatch 👁️‍🗨️ — دليل الفحص الكامل

CPU-based Bitcoin private key verifier.
Reads raw 32-byte keys, computes full ECC pipeline, compares against 8 targets.

## Orphans of verification

يتم فحص كل مفتاح بـ 5 صيغ للتأكد من عدم هروب أي مفتاح:

| الصيغة | البادئة | سبب وجودها |
|:------:|:-------:|:----------|
| **Uncompressed** | 0x04 + X(32) + Y(32) = 65 بايت | المعيار الأصلي لـ Bitcoin (2009) |
| **Compressed** | 0x02/0x03 + X(32) = 33 بايت | المعيار الجديد (BIP-30, 2012) — لكن بعض المحافظ القديمة تستخدمه |
| **Hybrid (even)** | 0x06 + X(32) + Y(32) = 65 بايت | نادر — بعض SDKs القديمة (BitcoinJ, pybitcoin) |
| **Hybrid (odd)** | 0x07 + X(32) + Y(32) = 65 بايت | نادر — نفس SDKs |
| **P2SH** | HASH160(script) بدل pubkey | بعض المحافظ القديمة جداً استخدمت P2SH كـ default |

## الأهداف (8)

```
A1  12rMpw5HnEvAw3nQqLmRBCQyuktfpa4eVw    400 BTC    2010-03-16
A2  1HLvaTs3zR3oev9ya7Pzp3GB9Gqfg6XYJT    9260 BTC   2020-12-19  (Dust collector, likely not ours)
A3  1JA4MpuV8MMNYCDTFHdCQeXGyem7mqo4B4    400 BTC    2010-07-15
A4  13GvAdkFeHFGVxTHzcA2rD2e5BD4cGkbBH    200 BTC    2010-07-15
A5  1DTy9z4JvtqYsg44oagVpHqyQpF7ZLLs45    200 BTC    2010-07-17
A6  1MVLP2kRPNqz8VJUy83LstUoMQzUjgq4Zg    1200 BTC   2010-09-10
A7  15QezNwA5ThiPf7wo89TTnfBwny93VQFTp    200 BTC    2010-09-16
E1  198aMn6ZYAczwrE5NvNTUMyJ5qkfy4g3Hi    250 BTC    2009 (Genesis era)
```

الإجمالي: ~13,110 BTC = $150M+ (ذروة 2024)
