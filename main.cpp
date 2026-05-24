// ================================================================
// VaultWatch — CPU private key verifier
// Reads 32-byte keys from file or stdin, verifies via secp256k1
// ================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <secp256k1.h>
#include "targets.h"
#include "check.h"

// Known good test vector: privkey = 1 (0x000...001)
// Expected: hash160(compressed) = 751e76e8199196d454941c45d1b3a323f1433bd6
//           hash160(uncompressed) = 91b24bf9f5288532960ac687abb035127b1d28a5
static int test_sanity(secp256k1_context *ctx) {
    fprintf(stderr, "[sanity] Testing privkey=1 (known vector)...\n");

    uint8_t pk[32];
    memset(pk, 0, 32);
    pk[31] = 1; // Big-endian: last byte = 1

    int result = check_privkey_multi(ctx, pk);
    if (result) {
        fprintf(stderr, "[sanity] FOUND target %d!\n", result);
    } else {
        // It shouldn't find a target (Satoshi's key isn't one of our targets),
        // but we at least tested the pipeline doesn't crash.
        fprintf(stderr, "[sanity] Pipeline OK (privkey=1 processed without error).\n");
    }
    return 0;
}

int main(int argc, char **argv) {
    fprintf(stderr, "VaultWatch — private key verifier\n");

    const char *keypath = NULL;
    int pipe_mode = 0;
    int do_sanity = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--keys") == 0 && i + 1 < argc) keypath = argv[++i];
        else if (strcmp(argv[i], "--pipe") == 0) pipe_mode = 1;
        else if (strcmp(argv[i], "--sanity") == 0) do_sanity = 1;
        else {
            fprintf(stderr, "Usage:\n");
            fprintf(stderr, "  ./vaultwatch --keys keys.bin\n");
            fprintf(stderr, "  ./vaultwatch --pipe\n");
            fprintf(stderr, "  ./vaultwatch --sanity\n");
            return 1;
        }
    }

    // Init secp256k1
    secp256k1_context *ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    check_init();

    if (do_sanity) {
        test_sanity(ctx);
        secp256k1_context_destroy(ctx);
        return 0;
    }

    FILE *in = NULL;
    if (pipe_mode) {
        in = stdin;
        fprintf(stderr, "[vaultwatch] Reading from stdin (pipe mode).\n");
    } else if (keypath) {
        in = fopen(keypath, "rb");
        if (!in) { fprintf(stderr, "vaultwatch: cannot open %s\n", keypath); return 1; }
        fprintf(stderr, "[vaultwatch] Reading from %s\n", keypath);
    } else {
        fprintf(stderr, "vaultwatch: specify --keys or --pipe\n");
        return 1;
    }

    // Process keys in batches
    const uint64_t BATCH = 1000000; // 1M keys per buffer
    uint8_t *buf = (uint8_t *)malloc(BATCH * 32);
    if (!buf) { fprintf(stderr, "vaultwatch: malloc failed\n"); return 1; }

    uint64_t total = 0;
    uint64_t found = 0;
    uint64_t reported = 0;

    while (!feof(in) && !ferror(in)) {
        uint64_t read = fread(buf, 32, BATCH, in);
        if (read == 0) break;

        for (uint64_t i = 0; i < read; i++) {
            int res = check_privkey_multi(ctx, buf + i * 32);
            if (res) {
                found++;
                // Log to stdout (machine-readable)
                printf("FOUND key=");
                for (int j = 0; j < 32; j++) printf("%02x", buf[i * 32 + j]);
                printf(" target=%d\n", res);
                fflush(stdout);
            }
        }

        total += read;

        // Progress every 10M keys
        if (total - reported >= 10000000) {
            fprintf(stderr, "[vaultwatch] %llu keys checked, %llu found\n",
                    (unsigned long long)total, (unsigned long long)found);
            reported = total;
        }
    }

    fprintf(stderr, "[vaultwatch] Done. %llu keys total, %llu found.\n",
            (unsigned long long)total, (unsigned long long)found);

    fclose(in);
    free(buf);
    secp256k1_context_destroy(ctx);
    return found > 0 ? 0 : 1;
}
