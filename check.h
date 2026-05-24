#ifndef CHECK_H
#define CHECK_H

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <secp256k1.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include "targets.h"

// =================================================================
// Check a single private key against ALL 8 targets.
// Returns target index (1..8) if found, 0 otherwise.
//
// Checks BOTH compressed and uncompressed pubkey formats.
// Uses secp256k1 library (official) + OpenSSL SHA/RIPEMD.
// =================================================================

static FILE *found_log = NULL;

static void check_init() {
    found_log = fopen("found.txt", "a");
    if (found_log) {
        time_t t = time(NULL);
        fprintf(found_log, "\n===== VAULTWATCH START %s", ctime(&t));
        fflush(found_log);
    }
}

static int check_privkey_multi(const secp256k1_context *ctx, const uint8_t pk[32]) {
    secp256k1_pubkey pub;
    if (!secp256k1_ec_pubkey_create(ctx, &pub, pk))
        return 0; // Invalid private key (e.g., == 0 or >= N)

    uint8_t ser[65], sha[32], rmd[20];
    size_t l;

    // --- Uncompressed (0x04 + X + Y) ---
    l = 65;
    secp256k1_ec_pubkey_serialize(ctx, ser, &l, &pub, SECP256K1_EC_UNCOMPRESSED);
    SHA256(ser, l, sha);
    RIPEMD160(sha, 32, rmd);

    for (int t = 0; t < NUM_TARGETS; t++) {
        if (memcmp(rmd, TARGET_H160[t], 20) == 0) {
            fprintf(stderr, "\n*** FOUND! Target %s (%s) ***\n",
                    TARGET_LABELS[t], TARGET_ADDRS[t]);
            fprintf(stderr, "privkey (uncompressed): ");
            for (int i = 0; i < 32; i++) fprintf(stderr, "%02x", pk[i]);
            fprintf(stderr, "\n\n");
            if (found_log) {
                fprintf(found_log, "[%s] FOUND (uncomp) key=", TARGET_LABELS[t]);
                for (int i = 0; i < 32; i++) fprintf(found_log, "%02x", pk[i]);
                fprintf(found_log, " target=%s\n", TARGET_ADDRS[t]);
                fflush(found_log);
            }
            return t + 1;
        }
    }

    // --- Compressed (0x02/0x03 + X) ---
    l = 33;
    secp256k1_ec_pubkey_serialize(ctx, ser, &l, &pub, SECP256K1_EC_COMPRESSED);
    SHA256(ser, l, sha);
    RIPEMD160(sha, 32, rmd);

    for (int t = 0; t < NUM_TARGETS; t++) {
        if (memcmp(rmd, TARGET_H160[t], 20) == 0) {
            fprintf(stderr, "\n*** FOUND! Target %s (%s) ***\n",
                    TARGET_LABELS[t], TARGET_ADDRS[t]);
            fprintf(stderr, "privkey (compressed): ");
            for (int i = 0; i < 32; i++) fprintf(stderr, "%02x", pk[i]);
            fprintf(stderr, "\n\n");
            if (found_log) {
                fprintf(found_log, "[%s] FOUND (comp) key=", TARGET_LABELS[t]);
                for (int i = 0; i < 32; i++) fprintf(found_log, "%02x", pk[i]);
                fprintf(found_log, " target=%s\n", TARGET_ADDRS[t]);
                fflush(found_log);
            }
            return t + 1;
        }
    }

    return 0;
}

#endif /* CHECK_H */
