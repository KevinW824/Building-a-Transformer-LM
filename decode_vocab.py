#!/usr/bin/env python3
import json
import base64
import sys
import os
import random

def main():
    # Allow user to specify vocab file, default to tinystories
    vocab_file = sys.argv[1] if len(sys.argv) > 1 else "tinystories_bpe_vocab.json"
    
    if not os.path.exists(vocab_file):
        print(f"Error: {vocab_file} not found!")
        print("Usage: python decode_vocab.py [vocab_file.json]")
        return
    
    # Load the vocabulary from JSON file
    with open(vocab_file, 'r') as f:
        vocab_json = json.load(f)
    
    # Convert from JSON format back to the original format
    vocab = {}
    for token_id_str, b64_encoded in vocab_json.items():
        token_id = int(token_id_str)
        token_bytes = base64.b64decode(b64_encoded)
        vocab[token_id] = token_bytes
    
    # Randomly sample 10 tokens from the vocabulary
    vocab_size = len(vocab)
    sample_size = min(10, vocab_size)
    random_indices = sorted(random.sample(range(vocab_size), sample_size))
    
    print(f'Random sample of {sample_size} vocabulary entries from {vocab_file}:')
    for i in random_indices:
        token_bytes = vocab[i]
        try:
            token_str = token_bytes.decode('utf-8')
            # Handle special characters for better display
            if token_str.isprintable() and token_str.strip():
                print(f'  Token {i:4d}: "{token_str}"')
            else:
                # Show non-printable characters more clearly
                print(f'  Token {i:4d}: {token_bytes!r} -> (special/control char)')
        except UnicodeDecodeError:
            print(f'  Token {i:4d}: {token_bytes!r} -> (non-UTF8)')
    
    print(f'\nTotal vocabulary size: {vocab_size}')

if __name__ == "__main__":
    main()