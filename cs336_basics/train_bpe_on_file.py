#!/usr/bin/env python3
"""
Train a BPE tokenizer on the TinyStories dataset.
"""

import time
import os
import json
import sys
import multiprocessing as mp
from cs336_basics.bpe import (
    pretokenize_file,
    train_bpe_from_pre_tokens_indexed,
)

def main():
    # Get input file from command line argument
    if len(sys.argv) < 2:
        print("Usage: python train_tinystories_bpe.py <input_file> [vocab_size]")
        print("Example: python train_tinystories_bpe.py data/TinyStoriesV2-GPT4-train.txt 10000")
        return
    
    # Configuration
    input_path = sys.argv[1]
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 10000  # Default to 10000
    special_tokens = ["<|endoftext|>"]
    # Always use the indexed trainer
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        return
    
    print(f"Training BPE tokenizer on {input_path}")
    print(f"Target vocabulary size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print(f"File size: {os.path.getsize(input_path) / (1024**3):.2f} GB")
    print()
    
    # Train the tokenizer with progress tracking
    print("Starting BPE training...")
    total_start_time = time.time()
    
    # Phase 1: Pre-tokenization with progress
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    num_processes = mp.cpu_count()
    
    print(f"📖 Pre-tokenizing file: {file_size_mb:.1f} MB using {num_processes} processes")
    pretokenize_start = time.time()
    
    pre_tokens = pretokenize_file(input_path, special_tokens)
    
    pretokenize_end = time.time()
    pretokenize_time = pretokenize_end - pretokenize_start
    total_sequences = len(pre_tokens)
    pretokenize_throughput = file_size_mb / pretokenize_time if pretokenize_time > 0 else 0
    
    print(f"✅ Pre-tokenization complete: {total_sequences:,} unique sequences | "
          f"{pretokenize_throughput:.1f} MB/s | {pretokenize_time:.1f}s")
    
    # Phase 2: BPE training with progress
    target_merges = vocab_size - 257 - len(special_tokens)  # 256 bytes + special tokens
    print(f"🚀 Starting BPE training: {target_merges} merges needed")
    
    last_print = {"t": time.time()}
    def progress_cb(n_merges: int, merged_token: bytes):
        now = time.time()
        if (n_merges % 100 == 0) or (now - last_print["t"] > 30) or (n_merges == target_merges):
            elapsed = now - total_start_time
            merges_per_sec = n_merges / elapsed if elapsed > 0 else 0.0
            remaining = max(0, target_merges - n_merges)
            eta = remaining / merges_per_sec if merges_per_sec > 0 else float("inf")
            if eta == float("inf"):
                eta_str = "calculating..."
            else:
                eta_min = eta / 60
                eta_str = f"{eta_min/60:.1f}h" if eta_min > 60 else (f"{eta_min:.1f}m" if eta_min > 1 else f"{eta:.0f}s")
            try:
                merged_str = merged_token.decode('utf-8')
                merge_info = f'"{merged_str}"' if merged_str.isprintable() and len(merged_str) <= 10 else f"{merged_token!r}"
            except Exception:
                merge_info = f"{merged_token!r}"
            pct = (n_merges / target_merges) * 100 if target_merges > 0 else 0
            print(f"📊 Progress: {n_merges}/{target_merges} ({pct:.1f}%) | {merges_per_sec:.1f} merges/s | ETA: {eta_str} | Latest: {merge_info}")
            last_print["t"] = now

    vocab, merges = train_bpe_from_pre_tokens_indexed(
        pre_tokens, vocab_size, special_tokens, progress_callback=progress_cb
    )
    
    total_end_time = time.time()
    training_time = total_end_time - total_start_time
    
    # Results
    print(f"\n✅ BPE training completed!")
    print(f"⏱️  Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"📖 Final vocabulary size: {len(vocab)}")
    print(f"🔀 Number of merges: {len(merges)}")
    
    # Show some example merges
    print(f"\n🔍 First 10 merges:")
    for i, (token1, token2) in enumerate(merges[:10]):
        merged = token1 + token2
        print(f"  {i+1:2d}. {token1!r} + {token2!r} → {merged!r}")
    
    # Show some vocabulary entries
    print(f"\n📚 Sample vocabulary entries:")
    for token_id in range(min(20, len(vocab))):
        token_bytes = vocab[token_id]
        try:
            # Try to decode as UTF-8 for readability
            token_str = token_bytes.decode('utf-8')
            print(f"  {token_id:3d}: {token_bytes!r} → '{token_str}'")
        except UnicodeDecodeError:
            print(f"  {token_id:3d}: {token_bytes!r}")
    
    # Performance metrics
    file_size_mb = os.path.getsize(input_path) / (1024**2)
    throughput = file_size_mb / training_time
    print(f"\n⚡ Throughput: {throughput:.2f} MB/s")
    
    # Serialize vocabulary and merges to disk
    print(f"\n💾 Serializing tokenizer to disk...")
    
    # Generate output filenames based on input file
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    vocab_json = f"{base_name}_bpe_vocab.json"
    merges_json = f"{base_name}_bpe_merges.json"
    
    # Save as JSON (human-readable format)
    # Convert vocab keys to strings and bytes to base64 for JSON serialization
    import base64
    vocab_json_data = {}
    for token_id, token_bytes in vocab.items():
        vocab_json_data[str(token_id)] = base64.b64encode(token_bytes).decode('ascii')
    
    merges_json_data = []
    for token1, token2 in merges:
        merges_json_data.append([
            base64.b64encode(token1).decode('ascii'),
            base64.b64encode(token2).decode('ascii')
        ])
    
    with open(vocab_json, "w") as f:
        json.dump(vocab_json_data, f, indent=2)
    
    with open(merges_json, "w") as f:
        json.dump(merges_json_data, f, indent=2)
    
    print(f"✅ Saved tokenizer files:")
    print(f"  - {vocab_json} (vocabulary)")
    print(f"  - {merges_json} (merges)")
    
    return vocab, merges

if __name__ == "__main__":
    vocab, merges = main()