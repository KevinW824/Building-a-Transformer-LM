import json
import regex as re
from collections.abc import Iterable, Iterator
from typing import Dict, List, Tuple

# GPT-2 style pretokenization pattern (supports unicode letters/numbers)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """
    A BPE tokenizer that encodes text into integer IDs and decodes integer IDs into text.
    
    Supports user-provided special tokens that are appended to the vocabulary if they
    aren't already there.
    """
    
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) special tokens.
        
        Args:
            vocab: dict[int, bytes] mapping token IDs to byte tokens
            merges: list[tuple[bytes, bytes]] of BPE merges in order of creation
            special_tokens: list[str] | None optional list of special tokens
        """
        # Copy vocabulary to avoid modifying original
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        
        # Create reverse mapping from bytes to token IDs
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        
        # Handle special tokens
        if special_tokens:
            for special_token in special_tokens:
                special_token_bytes = special_token.encode('utf-8')
                if special_token_bytes not in self.byte_to_id:
                    # Add to vocabulary with next available ID
                    new_id = len(self.vocab)
                    self.vocab[new_id] = special_token_bytes
                    self.byte_to_id[special_token_bytes] = new_id
        
        self.special_tokens = special_tokens or []
        
        # Create merge lookup for efficiency
        self.merge_dict = {pair: i for i, pair in enumerate(merges)}
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        """
        Class method that constructs and returns a Tokenizer from serialized vocabulary and merges.
        
        Args:
            vocab_filepath: str path to vocabulary JSON file
            merges_filepath: str path to merges text file  
            special_tokens: list[str] | None optional list of special tokens
            
        Returns:
            Tokenizer instance
        """
        # Load vocabulary from JSON
        with open(vocab_filepath, 'r') as f:
            vocab_data = json.load(f)
        
        # Convert vocab to expected format (int -> bytes)
        vocab = {}
        for token_bytes, token_id in vocab_data.items():
            vocab[token_id] = token_bytes.encode('utf-8')
        
        # Load merges from text file
        merges = []
        with open(merges_filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ' ' in line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        token1, token2 = parts
                        merges.append((token1.encode('utf-8'), token2.encode('utf-8')))
        
        return cls(vocab, merges, special_tokens)
    
    def _pretokenize(self, text: str) -> List[str]:
        """
        Pre-tokenize text handling special tokens with greedy matching (longest match first).
        
        Args:
            text: Input text to pre-tokenize
            
        Returns:
            List of pre-token strings
        """
        if not self.special_tokens:
            # No special tokens, use regex directly
            return re.findall(PAT, text)
        
        # Sort special tokens by length (longest first) for greedy matching
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        escaped_special_tokens = [re.escape(token) for token in sorted_special_tokens]
        split_pattern = "|".join(escaped_special_tokens)
        
        # Split text on special tokens, keeping the delimiters
        chunks = re.split(f"({split_pattern})", text)
        
        pretokens = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                # This is a special token
                pretokens.append(chunk)
            else:
                # Regular text, apply regex tokenization
                pretokens.extend(re.findall(PAT, chunk))
        
        return pretokens
    
    def _apply_bpe_to_pretoken(self, pretoken: str) -> List[int]:
        """
        Apply BPE merges to a single pre-token and return token IDs.
        
        Args:
            pretoken: Single pre-token string
            
        Returns:
            List of token IDs
        """
        # Check if this is a special token
        if pretoken in self.special_tokens:
            special_token_bytes = pretoken.encode('utf-8')
            return [self.byte_to_id[special_token_bytes]]
        
        # Convert pretoken to sequence of individual byte tokens
        pretoken_bytes = pretoken.encode('utf-8')
        tokens = [bytes([b]) for b in pretoken_bytes]
        
        # Apply merges in order
        for merge_pair in self.merges:
            # Look for this merge pair in current tokens
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == merge_pair[0] and tokens[i + 1] == merge_pair[1]:
                    # Merge these two tokens
                    merged_token = merge_pair[0] + merge_pair[1]
                    tokens = tokens[:i] + [merged_token] + tokens[i + 2:]
                else:
                    i += 1
        
        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            if token in self.byte_to_id:
                token_ids.append(self.byte_to_id[token])
            else:
                # For edge case. This shouldn't happen with proper BPE, but handle gracefully
                # Split into individual bytes
                for byte_val in token:
                    byte_token = bytes([byte_val])
                    token_ids.append(self.byte_to_id[byte_token])
        
        return token_ids
    
    def encode(self, text: str) -> List[int]:
        """
        Encode an input text into a sequence of token IDs.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        # Pre-tokenize the text
        pretokens = self._pretokenize(text)
        
        # Apply BPE to each pre-token and collect all token IDs
        all_token_ids = []
        for pretoken in pretokens:
            token_ids = self._apply_bpe_to_pretoken(pretoken)
            all_token_ids.extend(token_ids)
        
        return all_token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files.
        
        Args:
            iterable: Iterable of strings (e.g., file handle)
            
        Yields:
            Token IDs one at a time
        """
        for text_chunk in iterable:
            token_ids = self.encode(text_chunk)
            for token_id in token_ids:
                yield token_id
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        
        Args:
            ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        if not ids:
            return ""
        
        # Look up each ID in vocabulary and concatenate bytes
        byte_sequence = b""
        for token_id in ids:
            if token_id in self.vocab:
                byte_sequence += self.vocab[token_id]
            # If token_id not in vocab, skip it (graceful degradation)
        
        # Decode bytes to string, replacing malformed bytes with replacement character
        try:
            return byte_sequence.decode('utf-8', errors='replace')
        except Exception:
            # If any other error occurs, return empty string
            return ""
