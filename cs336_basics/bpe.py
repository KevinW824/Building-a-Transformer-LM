import regex as re
from collections import Counter
from typing import List, Tuple, Dict, BinaryIO
import multiprocessing as mp
import os

# GPT-2 style pretokenization pattern (supports unicode letters/numbers)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def vocabulary_initialization(special_tokens: List[str] = None) -> Dict[int, bytes]:
    """
    Build the initial vocabulary for a byte-level BPE tokenizer.
    
    Args:
        special_tokens: List of special token strings to add to vocabulary.
    
    Returns:
        A dictionary mapping integer IDs to bytes objects.

    This creates a one-to-one mapping from bytestring tokens to integer IDs.
    Special tokens are added first (IDs 0, 1, 2, ...), followed by
    all 256 possible byte values.
    """
    vocab = {}
    token_id = 0
    
    # Add special tokens first (they get lower IDs)
    if special_tokens:
        for special_token in special_tokens:
            vocab[token_id] = special_token.encode('utf-8')
            token_id += 1
    
    # Add byte tokens (IDs start after special tokens)
    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1
    
    return vocab


def pretokenize_chunk(chunk: str, special_tokens: List[str] = None) -> Dict[Tuple[bytes, ...], int]:
    """
    Pre-tokenize a single chunk of text.
    
    Args:
        chunk: Text chunk to pre-tokenize.
        special_tokens: List of special tokens that should be preserved as single tokens.
    
    Returns:
        Dictionary mapping pre-token sequences to their frequency counts.
    """
    if special_tokens is None:
        special_tokens = []
    
    pre_token_counts = Counter()
    
    if not special_tokens:
        # No special tokens, process the chunk normally
        for match in re.finditer(PAT, chunk):
            pre_token = match.group()
            # Convert to sequence of individual bytes
            token_sequence = tuple(bytes([b]) for b in pre_token.encode('utf-8'))
            pre_token_counts[token_sequence] += 1
    else:
        # Split chunk on special tokens using re.split
        escaped_special_tokens = [re.escape(token) for token in special_tokens]
        split_pattern = "|".join(escaped_special_tokens)
        
        # Split the chunk on special tokens
        sub_chunks = re.split(f"({split_pattern})", chunk)
        
        for sub_chunk in sub_chunks:
            if not sub_chunk:  # Skip empty chunks
                continue
            
            if sub_chunk in special_tokens:
                # This chunk is a special token, add it as a single token
                special_token_bytes = sub_chunk.encode('utf-8')
                pre_token_counts[tuple([special_token_bytes])] += 1
            else:
                # This chunk is regular text, pre-tokenize it normally
                for match in re.finditer(PAT, sub_chunk):
                    pre_token = match.group()
                    # Convert to sequence of individual bytes
                    token_sequence = tuple(bytes([b]) for b in pre_token.encode('utf-8'))
                    pre_token_counts[token_sequence] += 1
    
    return dict(pre_token_counts)


def process_chunk_worker(args):
    """Worker function for multiprocessing."""
    file_path, start, end, special_tokens = args
    
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk = chunk_bytes.decode('utf-8', errors='ignore')
    
    return pretokenize_chunk(chunk, special_tokens)





def pretokenize_file(input_path: str, special_tokens: List[str] = None) -> Dict[Tuple[bytes, ...], int]:
    """
    Pre-tokenize a file using parallel processing.
    
    Args:
        input_path: Path to the input file.
        special_tokens: List of special tokens.
    
    Returns:
        Dictionary mapping pre-token sequences to their frequency counts.
    """
    if special_tokens is None:
        special_tokens = []
    
    # Use parallel processing
    num_processes = mp.cpu_count()
    
    # If no special tokens, fall back to simple chunking
    if not special_tokens:
        # Simple chunking by file size
        file_size = os.path.getsize(input_path)
        chunk_size = file_size // num_processes
        
        chunk_args = []
        for i in range(num_processes):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_processes - 1 else file_size
            chunk_args.append((input_path, start, end, special_tokens))
    else:
        # Use special token boundaries for chunking
        with open(input_path, 'rb') as f:
            # Use the first special token for boundary detection
            split_token = special_tokens[0].encode('utf-8')
            boundaries = find_chunk_boundaries(f, num_processes, split_token)
        
        chunk_args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunk_args.append((input_path, start, end, special_tokens))
    
    # Process chunks in parallel
    with mp.Pool(num_processes) as pool:
        chunk_results = pool.map(process_chunk_worker, chunk_args)
    
    # Merge results from all chunks
    merged_counts = Counter()
    for chunk_result in chunk_results:
        for token_sequence, count in chunk_result.items():
            merged_counts[token_sequence] += count
    
    return dict(merged_counts)


def get_byte_pairs(pre_tokens: Dict[Tuple[bytes, ...], int], special_tokens: List[str] = None) -> Counter:
    """
    Count all adjacent byte pairs within each pre-token sequence.
    
    Args:
        pre_tokens: Dictionary mapping pre-token sequences to their frequencies.
        special_tokens: List of special tokens that should not be split.
    
    Returns:
        Counter of byte pairs with their frequencies.
    """
    if special_tokens is None:
        special_tokens = []
    
    # Convert special tokens to bytes for comparison
    special_tokens_bytes = {tuple([token.encode('utf-8')]) for token in special_tokens}
    
    pair_counts = Counter()
    
    for token_sequence, freq in pre_tokens.items():
        # Skip counting pairs within special tokens
        if token_sequence in special_tokens_bytes:
            continue
        
        # Count adjacent pairs in the token sequence
        for i in range(len(token_sequence) - 1):
            pair = (token_sequence[i], token_sequence[i + 1])
            pair_counts[pair] += freq
    
    return pair_counts


def apply_merge_to_token_sequence(token_sequence: Tuple[bytes, ...], pair: Tuple[bytes, bytes]) -> Tuple[bytes, ...]:
    """
    Apply a merge of two bytes to a token sequence.
    i.e. ("h", "e", "l", "l", "o") and ("h", "e") -> ("he", "l", "l", "o")

    Args:
        token_sequence: Tuple of byte tokens.
        pair: The pair of bytes to merge.
    
    Returns:
        New token sequence with the pair merged.
    """
    if len(token_sequence) < 2:
        return token_sequence
    
    result = []
    i = 0
    while i < len(token_sequence):
        if i < len(token_sequence) - 1 and token_sequence[i] == pair[0] and token_sequence[i + 1] == pair[1]:
            # Merge the pair into a single token
            merged_token = pair[0] + pair[1]
            result.append(merged_token)
            i += 2
        else:
            result.append(token_sequence[i])
            i += 1
    
    return tuple(result)


def apply_merge_to_pre_tokens(pre_tokens: Dict[Tuple[bytes, ...], int], pair: Tuple[bytes, bytes], special_tokens: List[str] = None) -> Dict[Tuple[bytes, ...], int]:
    """
    Apply a merge operation to all pre-token sequences, excluding special tokens.
    
    Args:
        pre_tokens: Dictionary mapping pre-token sequences to their frequencies.
        pair: The pair of bytes to merge.
        special_tokens: List of special tokens that should not be modified.
    
    Returns:
        Updated pre-tokens dictionary with the merge applied.
    """
    if special_tokens is None:
        special_tokens = []
    
    # Convert special tokens to the format we use for comparison
    special_tokens_bytes = {tuple([token.encode('utf-8')]) for token in special_tokens}
    
    new_pre_tokens = Counter()
    
    for token_sequence, freq in pre_tokens.items():
        # Don't modify special tokens
        if token_sequence in special_tokens_bytes:
            new_pre_tokens[token_sequence] += freq
        else:
            merged_sequence = apply_merge_to_token_sequence(token_sequence, pair)
            new_pre_tokens[merged_sequence] += freq
    
    return dict(new_pre_tokens)


def find_most_frequent_pair(pair_counts: Counter) -> Tuple[bytes, bytes]:
    """
    Find the most frequent pair, breaking ties lexicographically.
    
    Args:
        pair_counts: Counter of pair frequencies.
    
    Returns:
        The most frequent pair (lexicographically greatest in case of ties).
    """
    max_freq = max(pair_counts.values())
    most_frequent_pairs = [pair for pair, count in pair_counts.items() if count == max_freq]
    return max(most_frequent_pairs)  # Lexicographically greatest


def sequence_contains_pair(token_sequence: Tuple[bytes, ...], target_pair: Tuple[bytes, bytes]) -> bool:
    """
    Check if a token sequence contains a specific pair.
    
    Args:
        token_sequence: Sequence of byte tokens.
        target_pair: The pair to search for.
    
    Returns:
        True if the sequence contains the pair.
    """
    for i in range(len(token_sequence) - 1):
        if token_sequence[i] == target_pair[0] and token_sequence[i + 1] == target_pair[1]:
            return True
    return False


def count_pairs_in_sequence(token_sequence: Tuple[bytes, ...]) -> Counter:
    """
    Count all adjacent pairs in a token sequence.
    
    Args:
        token_sequence: Sequence of byte tokens.
    
    Returns:
        Counter of pairs in the sequence.
    """
    pairs = Counter()
    for i in range(len(token_sequence) - 1):
        pair = (token_sequence[i], token_sequence[i + 1])
        pairs[pair] += 1
    return pairs


def apply_merge_and_track_changes(
    current_pre_tokens: Dict[Tuple[bytes, ...], int],
    merge_pair: Tuple[bytes, bytes],
    special_tokens_bytes: set
) -> Tuple[Dict[Tuple[bytes, ...], int], Counter]:
    """
    Apply a merge operation and track the changes to pair counts.
    
    Args:
        current_pre_tokens: Current pre-token sequences and frequencies.
        merge_pair: The pair to merge.
        special_tokens_bytes: Set of special token sequences to skip.
    
    Returns:
        Tuple of (new_pre_tokens, pair_changes).
    """
    new_pre_tokens = {}
    pair_changes = Counter()
    
    for token_sequence, freq in current_pre_tokens.items():
        if token_sequence in special_tokens_bytes:
            # Special tokens don't change
            new_pre_tokens[token_sequence] = freq
            continue
        
        if not sequence_contains_pair(token_sequence, merge_pair):
            # No changes to this sequence
            new_pre_tokens[token_sequence] = freq
            continue
        
        # This sequence will change - remove its old pairs
        old_pairs = count_pairs_in_sequence(token_sequence)
        for pair, count in old_pairs.items():
            pair_changes[pair] -= freq * count
        
        # Apply merge and count new pairs
        merged_sequence = apply_merge_to_token_sequence(token_sequence, merge_pair)
        
        if merged_sequence in new_pre_tokens:
            new_pre_tokens[merged_sequence] += freq
        else:
            new_pre_tokens[merged_sequence] = freq
        
        new_pairs = count_pairs_in_sequence(merged_sequence)
        for pair, count in new_pairs.items():
            pair_changes[pair] += freq * count
    
    return new_pre_tokens, pair_changes


def update_pair_counts(pair_counts: Counter, pair_changes: Counter) -> None:
    """
    Update pair counts incrementally based on changes.
    
    Args:
        pair_counts: Current pair counts to update (modified in place).
        pair_changes: Changes to apply to pair counts.
    """
    for pair, change in pair_changes.items():
        pair_counts[pair] += change
        if pair_counts[pair] <= 0:
            del pair_counts[pair]


def train_bpe_from_pre_tokens(pre_tokens: Dict[Tuple[bytes, ...], int], vocab_size: int, special_tokens: List[str] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train BPE on pre-token sequences until reaching the target vocabulary size.
    
    Args:
        pre_tokens: Dictionary mapping pre-token sequences to their frequencies.
        vocab_size: Target vocabulary size.
        special_tokens: List of special tokens that should never be split.
    
    Returns:
        Tuple of (vocabulary, merges) where:
        - vocabulary: Dict[int, bytes] mapping token IDs to byte tokens
        - merges: List[Tuple[bytes, bytes]] of merge operations in order
    """
    if special_tokens is None:
        special_tokens = []
    
    # Initialize vocabulary with special tokens and byte tokens
    vocab = vocabulary_initialization(special_tokens)
    merges = []
    
    # Current pre-tokens (will be updated as we merge)
    current_pre_tokens = pre_tokens.copy()
    
    # Precompute special tokens bytes for efficiency
    special_tokens_bytes = {tuple([token.encode('utf-8')]) for token in special_tokens}
    
    # Initialize pair counts cache
    pair_counts = Counter()
    for token_sequence, freq in current_pre_tokens.items():
        if token_sequence in special_tokens_bytes:
            continue
        sequence_pairs = count_pairs_in_sequence(token_sequence)
        for pair, count in sequence_pairs.items():
            pair_counts[pair] += freq * count
    
    while len(vocab) < vocab_size and pair_counts:
        # Find the most frequent pair
        best_pair = find_most_frequent_pair(pair_counts)
        
        # Add the merge to our list and vocabulary
        merges.append(best_pair)
        merged_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_token
        
        # Apply merge and track changes to pair counts
        current_pre_tokens, pair_changes = apply_merge_and_track_changes(
            current_pre_tokens, best_pair, special_tokens_bytes
        )
        
        # Update pair counts incrementally
        update_pair_counts(pair_counts, pair_changes)
    
    return vocab, merges


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    **kwargs,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Given a path to an input text file, train a (byte-level) BPE tokenizer.
    
    Args:
        input_path: Path to a text file with BPE tokenizer training data.
        vocab_size: A positive integer that defines the maximum final vocabulary size 
                   (including the initial byte vocabulary, vocabulary items produced 
                   from merging, and any special tokens).
        special_tokens: A list of strings to add to the vocabulary. These special tokens 
                       do not otherwise affect BPE training.
        **kwargs: Additional keyword arguments (for compatibility).
    
    Returns:
        Tuple of (vocab, merges) where:
        - vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID 
                 in the vocabulary) to bytes (token bytes).
        - merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. 
                  Each list item is a tuple of bytes (<token1>, <token2>), representing 
                  that <token1> was merged with <token2>. The merges are ordered by 
                  order of creation.
    """
    # Use optimized file processing (auto-decides on parallel based on file size)
    pre_tokens = pretokenize_file(input_path, special_tokens)
    
    # Train BPE on the pre-tokens
    vocab, merges = train_bpe_from_pre_tokens(pre_tokens, vocab_size, special_tokens)
    
    return vocab, merges