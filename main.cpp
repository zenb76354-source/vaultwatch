// ================================================================
// VaultWatch — CPU private key verifier
// Multi-threaded (OpenMP), mmap for large files,
// checks: compressed, uncompressed, hybrid (0x06/0x07), P2SH
// ================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <secp256k1.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include "targets.h"

static FILE *found_log = NULL;

static void check_init() {
    found_log = fopen("found.txt", "a");
    if (found_log) {
        time_t t = time(NULL);
        fprintf(found_log, "\n===== VAULTWATCH START %s", ctime(&t));
        fflush(found_log);
    }
}

static void check_found(const char *fmt, int t, const uint8_t pk[32]) {
    #pragma omp critical
    {
        fprintf(stderr, "\n*** FOUND! %s (%s) %s ***\n",
                TARGET_LABELS[t], TARGET_ADDRS[t], fmt);
        fprintf(stderr, "privkey: ");
        for (int i = 0; i < 32; i++) fprintf(stderr, "%02x", pk[i]);
        fprintf(stderr, "\n\n");
        if (found_log) {
            fprintf(found_log, "[%s] %s key=", TARGET_LABELS[t], fmt);
            for (int i = 0; i < 32; i++) fprintf(found_log, "%02x", pk[i]);
            fprintf(found_log, " target=%s\n", TARGET_ADDRS[t]);
            fflush(found_log);
        }
    }
}

static int check_one(const secp256k1_context *ctx, const uint8_t pk[32]) {
    secp256k1_pubkey pub;
    if (!secp256k1_ec_pubkey_create(ctx, &pub, pk))
        return 0;

    uint8_t sha[32], rmd[20];
    size_t l;
    uint8_t ser[65];

    // --- Uncompressed (0x04 + X + Y) ---
    l = 65;
    secp256k1_ec_pubkey_serialize(ctx, ser, &l, &pub, SECP256K1_EC_UNCOMPRESSED);
    SHA256(ser, l, sha);
    RIPEMD160(sha, 32, rmd);
    for (int t = 0; t < NUM_TARGETS; t++)
        if (memcmp(rmd, TARGET_H160[t], 20) == 0) { check_found("uncompressed", t, pk); return t+1; }

    // --- Compressed (0x02/0x03 + X) ---
    l = 33;
    secp256k1_ec_pubkey_serialize(ctx, ser, &l, &pub, SECP256K1_EC_COMPRESSED);
    SHA256(ser, l, sha);
    RIPEMD160(sha, 32, rmd);
    for (int t = 0; t < NUM_TARGETS; t++)
        if (memcmp(rmd, TARGET_H160[t], 20) == 0) { check_found("compressed", t, pk); return t+1; }

    // --- Hybrid even (0x06 + X + Y) ---
    l = 65;
    secp256k1_ec_pubkey_serialize(ctx, ser, &l, &pub, SECP256K1_EC_UNCOMPRESSED);
    ser[0] = 0x06 | (ser[64] & 1); // hybrid: 0x06 if Y even, 0x07 if Y odd
    SHA256(ser, l, sha);
    RIPEMD160(sha, 32, rmd);
    for (int t = 0; t < NUM_TARGETS; t++)
        if (memcmp(rmd, TARGET_H160[t], 20) == 0) { check_found("hybrid", t, pk); return t+1; }

    // --- P2SH (script hash) ---
    // Build: OP_DUP OP_HASH160 <20-byte hash> OP_EQUALVERIFY OP_CHECKSIG
    // Then HASH160 that script
    uint8_t script[25];
    script[0] = 0x76; // OP_DUP
    script[1] = 0xa9; // OP_HASH160
    script[2] = 20;   // push 20 bytes
    SHA256(ser, 33, sha); // hash of compressed pubkey
    RIPEMD160(sha, 32, script + 3);
    script[23] = 0x88; // OP_EQUALVERIFY
    script[24] = 0xac; // OP_CHECKSIG
    SHA256(script, 25, sha);
    RIPEMD160(sha, 32, rmd);
    for (int t = 0; t < NUM_TARGETS; t++)
        if (memcmp(rmd, TARGET_H160[t], 20) == 0) { check_found("P2SH", t, pk); return t+1; }

    return 0;
}

int main(int argc, char **argv) {
    fprintf(stderr, "VaultWatch — private key verifier\n");

    const char *keypath = NULL;
    int pipe_mode = 0, do_sanity = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--keys") == 0 && i + 1 < argc) keypath = argv[++i];
        else if (strcmp(argv[i], "--pipe") == 0) pipe_mode = 1;
        else if (strcmp(argv[i], "--sanity") == 0) do_sanity = 1;
        else {
            fprintf(stderr, "Usage:\n  ./vaultwatch --keys keys.bin\n  ./vaultwatch --pipe\n  ./vaultwatch --sanity\n");
            return 1;
        }
    }

    secp256k1_context *ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    check_init();

    // --- Sanity test ---
    if (do_sanity) {
        uint8_t pk[32]; memset(pk, 0, 32); pk[31] = 1;
        fprintf(stderr, "[sanity] Testing privkey=1...\n");
        int r = check_one(ctx, pk);
        fprintf(stderr, "[sanity] Pipeline OK (privkey=1 processed, result=%d).\n", r);
        secp256k1_context_destroy(ctx);
        return 0;
    }

    uint64_t total = 0, found = 0;

    if (pipe_mode) {
        // Pipe mode: read from stdin sequentially
        fprintf(stderr, "[vaultwatch] Reading from stdin (pipe).\n");
        const uint64_t B = 1000000;
        uint8_t *buf = (uint8_t*)malloc(B * 32);
        if (!buf) { fprintf(stderr, "malloc failed\n"); return 1; }

        uint64_t done=0, reported=0;
        while (!feof(stdin)) {
            uint64_t r = fread(buf, 32, B, stdin);
            if (r == 0) break;
            total += r;

            #pragma omp parallel for reduction(+:found)
            for (uint64_t i = 0; i < r; i++) {
                if (check_one(ctx, buf + i*32)) found++;
            }

            done += r;
            if (done - reported >= 10000000) {
                fprintf(stderr, "[vaultwatch] %llu keys checked, %llu found\n",
                        (unsigned long long)done, (unsigned long long)found);
                reported = done;
            }
        }
        free(buf);
        fprintf(stderr, "[vaultwatch] Done. %llu keys, %llu found.\n",
                (unsigned long long)total, (unsigned long long)found);
    }
    else if (keypath) {
        // File mode with mmap
        int fd = open(keypath, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "vaultwatch: cannot open %s\n", keypath); return 1; }

        struct stat st;
        fstat(fd, &st);
        uint64_t fsize = (uint64_t)st.st_size;
        if (fsize % 32 != 0) { fprintf(stderr, "vaultwatch: file size %llu not multiple of 32\n", (unsigned long long)fsize); close(fd); return 1; }

        uint8_t *data = (uint8_t*)mmap(NULL, fsize, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (data == MAP_FAILED) { fprintf(stderr, "vaultwatch: mmap failed\n"); return 1; }

        total = fsize / 32;
        fprintf(stderr, "[vaultwatch] mmap %s (%llu keys)\n", keypath, (unsigned long long)total);

        const uint64_t CHUNK = 5000000; // 5M keys per OpenMP chunk
        uint64_t reported = 0;

        #pragma omp parallel for reduction(+:found) schedule(dynamic, 1)
        for (uint64_t off = 0; off < total; off += CHUNK) {
            uint64_t end = off + CHUNK;
            if (end > total) end = total;
            for (uint64_t i = off; i < end; i++) {
                if (check_one(ctx, data + i*32)) found++;
            }
            #pragma omp critical
            {
                reported += end - off;
                fprintf(stderr, "[vaultwatch] %llu/%llu keys checked, %llu found\n",
                        (unsigned long long)reported, (unsigned long long)total, (unsigned long long)found);
            }
        }

        fprintf(stderr, "[vaultwatch] Done. %llu keys, %llu found.\n",
                (unsigned long long)total, (unsigned long long)found);
        munmap(data, fsize);
    }
    else {
        fprintf(stderr, "vaultwatch: specify --keys or --pipe\n");
        return 1;
    }

    secp256k1_context_destroy(ctx);
    return found > 0 ? 0 : 1;
}
