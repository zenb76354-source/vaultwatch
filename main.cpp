// ================================================================
// VaultWatch — CPU private key verifier
// Multi-threaded (OpenMP), mmap for large files,
// checks: compressed, uncompressed, hybrid (0x06/0x07), P2SH
// Auto-delete: --auto-delete N  (delete checked keys after N secs)
// ================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
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
        fprintf(stderr, "\n*** FOUND! TARGET %s (%s) %s ***\n",
                TARGET_LABELS[t], TARGET_ADDRS[t], fmt);
        fprintf(stderr, "privkey: ");
        for (int i = 0; i < 32; i++) fprintf(stderr, "%02x", pk[i]);
        fprintf(stderr, "\n\n");
        if (found_log) {
            fprintf(found_log, "[%s] %s key=", TARGET_LABELS[t], fmt);
            for (int i = 0; i < 32; i++) fprintf(found_log, "%02x", pk[i]);
            fprintf(found_log, " target=%s val=USD%.2f\n", TARGET_ADDRS[t], TARGET_BALANCE[t]);
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

    // --- Hybrid (0x06 even, 0x07 odd) ---
    l = 65;
    secp256k1_ec_pubkey_serialize(ctx, ser, &l, &pub, SECP256K1_EC_UNCOMPRESSED);
    ser[0] = 0x06 | (ser[64] & 1);
    SHA256(ser, l, sha);
    RIPEMD160(sha, 32, rmd);
    for (int t = 0; t < NUM_TARGETS; t++)
        if (memcmp(rmd, TARGET_H160[t], 20) == 0) { check_found("hybrid", t, pk); return t+1; }

    // --- P2SH (script hash of compressed pubkey) ---
    uint8_t script[25];
    script[0] = 0x76; script[1] = 0xa9; script[2] = 20;
    SHA256(ser, 33, sha);
    RIPEMD160(sha, 32, script + 3);
    script[23] = 0x88; script[24] = 0xac;
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
    int auto_delete = 0; // seconds to wait before deleting checked keys (0=disabled)

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--keys") == 0 && i + 1 < argc) keypath = argv[++i];
        else if (strcmp(argv[i], "--pipe") == 0) pipe_mode = 1;
        else if (strcmp(argv[i], "--sanity") == 0) do_sanity = 1;
        else if (strcmp(argv[i], "--auto-delete") == 0 && i + 1 < argc) auto_delete = atoi(argv[++i]);
        else {
            fprintf(stderr, "Usage:\n  ./vaultwatch --keys keys.bin\n  ./vaultwatch --pipe\n  ./vaultwatch --sanity\n  ./vaultwatch --keys keys.bin --auto-delete 300\n");
            return 1;
        }
    }

    secp256k1_context *ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    check_init();

    if (do_sanity) {
        uint8_t pk[32]; memset(pk, 0, 32); pk[31] = 1;
        fprintf(stderr, "[sanity] Testing privkey=1...\n");
        int r = check_one(ctx, pk);
        fprintf(stderr, "[sanity] Pipeline OK (privkey=1 processed, result=%d).\n", r);
        secp256k1_context_destroy(ctx);
        return 0;
    }

    uint64_t total = 0, found = 0;
    double start_time = (double)time(NULL);

    if (pipe_mode) {
        fprintf(stderr, "[vaultwatch] Reading from stdin (pipe).\n");
        const uint64_t B = 1000000;
        uint8_t *buf = (uint8_t*)malloc(B * 32);
        if (!buf) { fprintf(stderr, "malloc failed\n"); return 1; }

        uint64_t done=0, next_progress=100000;
        while (!feof(stdin)) {
            uint64_t r = fread(buf, 32, B, stdin);
            if (r == 0) break;
            total += r;

            #pragma omp parallel for reduction(+:found)
            for (uint64_t i = 0; i < r; i++) {
                if (check_one(ctx, buf + i*32)) found++;
            }

            done += r;
            if (done >= next_progress) {
                double elapsed = (double)time(NULL) - start_time;
                double rate = elapsed > 0.0 ? (double)done / elapsed : 0.0;
                double rate_k = rate / 1000.0;
                int pct = total > 0 ? (int)(100.0 * done / total) : 0;
                // In pipe mode we don't know total, show keys only
                fprintf(stderr, "\r[vaultwatch] %llu keys checked | %.0f k/s | %.0fs elapsed | %llu found   ",
                        (unsigned long long)done, rate_k, elapsed, (unsigned long long)found);
                if (next_progress < 10000000) next_progress += 100000;
                else if (next_progress < 100000000) next_progress += 1000000;
                else next_progress += 10000000;
            }
        }
        free(buf);
        fprintf(stderr, "\n[vaultwatch] Done. %llu keys checked, %llu found (%.0f sec).\n",
                (unsigned long long)total, (unsigned long long)found,
                (double)time(NULL) - start_time);
    }
    else if (keypath) {
        // Retry loop: wait for file to appear if it doesn't exist yet
        while (access(keypath, R_OK) != 0) {
            fprintf(stderr, "[vaultwatch] Waiting for %s...\n", keypath);
            sleep(30);
        }

        int fd = open(keypath, O_RDWR, 0644);
        if (fd < 0) { fprintf(stderr, "vaultwatch: cannot open %s\n", keypath); return 1; }

        struct stat st;
        fstat(fd, &st);
        uint64_t fsize = (uint64_t)st.st_size;

        // If auto-delete, remap as writable
        int prot = auto_delete ? (PROT_READ | PROT_WRITE) : PROT_READ;
        int flags = auto_delete ? MAP_SHARED : MAP_SHARED;

        uint8_t *data = (uint8_t*)mmap(NULL, fsize, prot, flags, fd, 0);
        if (data == (uint8_t*)MAP_FAILED) { fprintf(stderr, "vaultwatch: mmap failed\n"); close(fd); return 1; }
        close(fd);

        uint64_t offset = 0;
        uint64_t last_delete_time = 0;
        uint64_t progress_counter = 0;

        // Main loop: keep re-checking as file grows
        while (1) {
            // Re-check file size (it may have grown)
            fd = open(keypath, O_RDWR, 0644);
            if (fd < 0) break;
            fstat(fd, &st);
            uint64_t new_fsize = (uint64_t)st.st_size;
            close(fd);

            if (new_fsize != fsize || data == (uint8_t*)MAP_FAILED) {
                // File grew or was unmapped — remap
                if (data != (uint8_t*)MAP_FAILED) munmap(data, fsize);
                fsize = new_fsize;
                if (fsize == 0) {
                    data = (uint8_t*)MAP_FAILED;
                    sleep(1);
                    continue;
                }
                fd = open(keypath, O_RDWR, 0644);
                if (fd < 0) break;
                data = (uint8_t*)mmap(NULL, fsize, PROT_READ, MAP_SHARED, fd, 0);
                close(fd);
                if (data == (uint8_t*)MAP_FAILED) { fprintf(stderr, "vaultwatch: remap failed\n"); break; }
            }

            if (fsize % 32 != 0) {
                // Wait for full key to arrive
                sleep(5);
                continue;
            }

            total = fsize / 32;
            if (total <= offset) {
                // No new keys — wait
                sleep(60);
                continue;
            }

            uint64_t batch_total = total - offset;
            double batch_start = (double)time(NULL);
            fprintf(stderr, "\n[vaultwatch] Batch: %llu new keys (total file: %llu keys)\n",
                    (unsigned long long)batch_total, (unsigned long long)total);

            const uint64_t CHUNK = 5000000;
            uint64_t batch_found = 0;
            int found_any = 0;
            uint64_t done_in_batch = 0;
            uint64_t next_progress = 500000;

            #pragma omp parallel for reduction(+:batch_found) schedule(dynamic, 1)
            for (uint64_t off = offset; off < total; off += CHUNK) {
                uint64_t end = off + CHUNK;
                if (end > total) end = total;
                for (uint64_t i = off; i < end; i++) {
                    if (check_one(ctx, data + i*32)) batch_found++;
                }
                #pragma omp critical
                {
                    done_in_batch += (end - off);
                    progress_counter += (end - off);
                    if (done_in_batch >= next_progress || done_in_batch >= batch_total) {
                        double batch_elapsed = (double)time(NULL) - batch_start;
                        double total_elapsed = (double)time(NULL) - start_time;
                        double rate = batch_elapsed > 0.0 ? (double)done_in_batch / batch_elapsed : 0.0;
                        double rate_k = rate / 1000.0;
                        uint64_t remain = done_in_batch < batch_total ? batch_total - done_in_batch : 0;
                        double eta = rate > 0.0 && remain > 0 ? (double)remain / rate : 0.0;
                        double pct = 100.0 * done_in_batch / batch_total;
                        int c = progress_counter;
                        fprintf(stderr,
                            "\r[vaultwatch] [%5.1f%%] %llu/%llu keys | %.0f k/s | elapsed %ds | ETA %ds | found=%llu   "
                            , pct
                            , (unsigned long long)done_in_batch, (unsigned long long)batch_total
                            , rate_k
                            , (int)batch_elapsed, (int)eta
                            , (unsigned long long)batch_found);
                        fflush(stderr);
                        if (next_progress < 10000000) next_progress += 500000;
                        else next_progress += 5000000;
                    }
                }
            }

            if (batch_found > 0) found_any = 1;
            found += batch_found;

            if (found_any) {
                fprintf(stderr, "\n\n*** FOUND %llu KEYS! ***\n"
                        "Stopping check to protect discovery.\n"
                        "Found keys logged in found.txt\n\n",
                        (unsigned long long)batch_found);
                break;
            }

            double batch_done = (double)time(NULL) - batch_start;
            double total_elapsed = (double)time(NULL) - start_time;
            fprintf(stderr, "\n[vaultwatch] Batch done: %llu keys in %.0f sec | total: %llu keys checked in %.0f sec\n",
                    (unsigned long long)batch_total, batch_done,
                    (unsigned long long)total, total_elapsed);

            offset = total;
            uint64_t now = (uint64_t)time(NULL);

            // Auto-delete: delete immediately after each batch, or wait if auto_delete > 0
            if (auto_delete >= 0 && total > 0) {
                int do_delete = 0;
                if (auto_delete == 0) {
                    // Immediate delete after each batch
                    do_delete = 1;
                } else {
                    if (last_delete_time == 0) {
                        last_delete_time = now;
                    }
                    if (now - last_delete_time >= (uint64_t)auto_delete) {
                        do_delete = 1;
                    } else {
                        uint64_t remaining_sec = (uint64_t)auto_delete - (now - last_delete_time);
                        fprintf(stderr, "[vaultwatch] Next auto-delete in %llu seconds.\n",
                                (unsigned long long)remaining_sec);
                    }
                }
                if (do_delete) {
                    fprintf(stderr, "[vaultwatch] Auto-delete: trimming %llu checked keys...\n",
                            (unsigned long long)offset);

                    // Truncate the file — remove all checked keys
                    fd = open(keypath, O_RDWR, 0644);
                    if (fd >= 0) {
                        // Get current file size from disk (generator may have added more)
                        fstat(fd, &st);
                        uint64_t cur_fsize = (uint64_t)st.st_size;
                        uint64_t cur_total = cur_fsize / 32;

                        // Count how many real keys we can safely delete
                        uint64_t safe_delete = offset;
                        if (safe_delete > cur_total) safe_delete = cur_total;
                        uint64_t remaining = cur_total - safe_delete;

                        if (remaining > 0) {
                            // memcpy only what we can actually read from data
                            uint64_t copy_bytes = (remaining * 32 < fsize - safe_delete * 32) ? remaining * 32 : fsize - safe_delete * 32;
                            uint8_t *tmp = (uint8_t*)malloc(copy_bytes);
                            if (tmp && data != (uint8_t*)MAP_FAILED) {
                                memcpy(tmp, data + safe_delete * 32, copy_bytes);
                            }
                            close(fd);
                            // Remove old checked portion, rewrite remaining
                            FILE *fw = fopen(keypath, "wb");
                            if (fw && tmp && data != (uint8_t*)MAP_FAILED) {
                                fwrite(tmp, 1, copy_bytes, fw);
                                ftruncate(fileno(fw), copy_bytes);
                                fclose(fw);
                            } else {
                                ftruncate(fd, 0);
                            }
                            if (tmp) free(tmp);
                        } else {
                            close(fd);
                            FILE *fw = fopen(keypath, "wb");
                            if (fw) {
                                ftruncate(fileno(fw), 0);
                                fclose(fw);
                            }
                        }

                        // Remap safely
                        if (data != (uint8_t*)MAP_FAILED) munmap(data, fsize);
                        data = (uint8_t*)MAP_FAILED;
                        fsize = remaining * 32;
                        offset = 0;
                        if (fsize > 0) {
                            fd = open(keypath, O_RDWR, 0644);
                            if (fd >= 0) {
                                data = (uint8_t*)mmap(NULL, fsize, PROT_READ, MAP_SHARED, fd, 0);
                                close(fd);
                            }
                        }
                        fprintf(stderr, "[vaultwatch] Trimmed. Remaining: %llu keys.\n",
                                (unsigned long long)(fsize / 32));
                    }

                    last_delete_time = now;
                } else {
                    uint64_t remaining_sec = (uint64_t)auto_delete - (now - last_delete_time);
                    fprintf(stderr, "[vaultwatch] Next auto-delete in %llu seconds.\n",
                            (unsigned long long)remaining_sec);
                }
            }

            if (found > 0) break;
            if (data == (uint8_t*)MAP_FAILED) {
                // After auto-delete with 0 remaining, wait for generator to add keys
                sleep(5);
                continue;
            }
            sleep(5); // Quick re-check for new keys
        }

        if (data != (uint8_t*)MAP_FAILED) munmap(data, fsize);
    }
    else {
        fprintf(stderr, "vaultwatch: specify --keys or --pipe\n");
        return 1;
    }

    secp256k1_context_destroy(ctx);
    fprintf(stderr, "[vaultwatch] Session complete. %llu keys, %llu found.\n",
            (unsigned long long)total, (unsigned long long)found);
    return found > 0 ? 0 : 1;
}
