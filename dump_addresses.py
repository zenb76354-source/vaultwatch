#!/usr/bin/env python3
"""
Blockchain address dumper (2009-2012)
Extracts ALL Bitcoin addresses from blockchain range
Output: hex(h160) per line ? ready for vaultwatch target list

Usage:
  python3 dump_addresses.py --blocks 0,250000  # blocks 0 to 250k
  python3 dump_addresses.py --csv addresses_2009_2010.csv

This requires:
  - bitcoind running (txindex=1)
  - or: blockchain data directory
"""

import sys
import os
import struct
import hashlib
import base58

def hash160(pubkey):
    """SHA256 ? RIPEMD160 of a public key"""
    h = hashlib.sha256(pubkey).digest()
    return hashlib.new("ripemd160", h).digest()

def parse_txout(script, txid_hash, vout):
    """Extract addresses from a tx output script"""
    addresses = []
    
    if len(script) == 25 and script[0] == 0x76 and script[1] == 0xa9 and script[-2] == 0x88 and script[-1] == 0xac:
        # P2PKH: OP_DUP OP_HASH160 <20 bytes> OP_EQUALVERIFY OP_CHECKSIG
        h160 = script[3:23]
        addresses.append(h160.hex())
    
    elif len(script) == 23 and script[0] == 0xa9 and script[1] == 0x14 and script[-2] == 0x87 and script[-1] == 0xac:
        # P2SH: OP_HASH160 <20 bytes> OP_EQUAL
        h160 = script[2:22]
        addresses.append("a" + h160.hex())
    
    elif script[0] == 0x41 and script[-1] == 0xac:
        # Raw public key: <33 or 65 bytes compressed/uncompressed> OP_CHECKSIG
        pk = script[1:-1]
        if len(pk) in (33, 65):
            h160 = hash160(pk)
            addresses.append(h160.hex())
    
    elif script[0] == 0x21 and script[-1] == 0xac:
        # Compressed public key
        pk = script[1:-1]
        if len(pk) == 33:
            h160 = hash160(pk)
            addresses.append(h160.hex())
    
    return addresses

def process_block(blk_path, blk_height):
    """Parse a block file and extract addresses"""
    try:
        with open(blk_path, "rb") as f:
            data = f.read()
    except:
        return []
    
    addrs = []
    pos = 0
    
    while pos < len(data) - 80:
        # Skip block header (80 bytes)
        pos += 80
        
        if pos >= len(data) - 1:
            break
        
        # Transaction count (varint)
        tx_count, varint_size = decode_varint(data, pos)
        pos += varint_size
        
        for tx_idx in range(tx_count):
            if pos >= len(data) - 4:
                break
            
            # Version (4 bytes)
            version = struct.unpack("<i", data[pos:pos+4])[0]
            pos += 4
            
            # Input count (varint)
            in_count, vs = decode_varint(data, pos)
            pos += vs
            
            # Skip inputs
            for _ in range(in_count):
                # Previous tx hash (32 bytes)
                pos += 32
                # Output index (4 bytes)
                pos += 4
                # ScriptSig length (varint)
                ss_len, vs = decode_varint(data, pos)
                pos += vs
                # ScriptSig
                pos += ss_len
                # Sequence (4 bytes)
                pos += 4
            
            if pos >= len(data):
                break
            
            # Output count (varint)
            out_count, vs = decode_varint(data, pos)
            pos += vs
            
            # Process outputs
            for _ in range(out_count):
                if pos >= len(data) - 8:
                    break
                # Value (8 bytes)
                pos += 8
                # ScriptPubKey length (varint)
                spk_len, vs = decode_varint(data, pos)
                pos += vs
                # ScriptPubKey
                script = data[pos:pos+spk_len]
                pos += spk_len
                
                parsed = parse_txout(script, "", _)
                addrs.extend(parsed)
            
            # Locktime (4 bytes)
            pos += 4
    
    return addrs

def decode_varint(data, pos):
    val = data[pos]
    if val < 0xfd:
        return val, 1
    elif val == 0xfd:
        return struct.unpack("<H", data[pos+1:pos+3])[0], 3
    elif val == 0xfe:
        return struct.unpack("<I", data[pos+1:pos+5])[0], 5
    else:
        return struct.unpack("<Q", data[pos+1:pos+9])[0], 9

if __name__ == "__main__":
    print("Blockchain address dumper for VaultWatch")
    print("Usage: see source for instructions")
