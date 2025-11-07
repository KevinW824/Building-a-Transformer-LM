from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
from torch import nn

from einops import einsum, reduce

class Linear(nn.Module):
    """Bias-free linear layer: y = x W^T

    Shapes
    -------
    x: (..., in_features)
    W: (out_features, in_features)
    y: (..., out_features)

    Notes
    -----
    - Store weight as shape (out_features, in_features) for row-major friendliness.
    - Initialize with trunc_normal_ per spec (σ² = 2/(din + dout)), truncated to ±3σ.
    - Do *not* use nn.Linear or torch.nn.functional.linear.
    """
    def __init__(self, in_features: int, out_features: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__() 
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """Token embedding lookup.

    Shapes
    -------
    weight: (num_embeddings, embedding_dim)
    token_ids: (batch, seq_len) [torch.long]
    output: (batch, seq_len, embedding_dim)

    Notes
    -----
    - Do *not* use nn.Embedding.
    - Initialize with trunc_normal_ per spec (N(0,1) truncated to ±3).
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, T, D)
        return self.weight[token_ids]


# ----------------------------------------------------------------------------------
# §3.5.1  RMSNorm
# ----------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm.

    y = x / RMS(x) * g,  with per-dim learnable gain g and epsilon for stability.

    Inputs/Outputs: (batch, seq_len, d_model)
    - Upcast to float32 before squaring; downcast to original dtype on return.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D) -> (B, T, D)
        # TODO: upcast to float32, compute rms over last dim, normalize, scale by gain, restore dtype
        original_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(reduce(x * x, "... d -> ... 1", "mean") + self.eps)
        x = x / rms
        x = x.to(original_dtype)
        return x * self.gain

# ----------------------------------------------------------------------------------
# §3.5.3  Rotary Positional Embeddings (RoPE)
# ----------------------------------------------------------------------------------
class RotaryPositionalEmbedding(nn.Module):
    """Applies RoPE to last dimension (d_k) for positions provided.

    Precompute sin/cos for max_seq_len and slice via token_positions.
    Apply identical rotation per head (treat head as a batch-like dimension).

    Args:
        theta: base Θ value
        d_k: head dimension
        max_seq_len: maximum supported sequence length
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: Optional[torch.device] = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device 

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        frequency_bases = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)

        inv_freq = 1.0 / (self.theta ** (frequency_bases / d_k))
        angles = einsum(positions, inv_freq, "i, k -> i k")

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """x: (..., seq_len, d_k), token_positions: (..., seq_len) -> rotated x of same shape."""
        # TODO: index into precomputed sin/cos via token_positions and rotate pairs [2k, 2k+1]
        token_positions = token_positions.long()
        xc = x.float().reshape(*x.shape[:-1], -1, 2)
        xc = torch.view_as_complex(xc)
        rot = torch.complex(self.cos[token_positions], self.sin[token_positions])
        xc = xc * rot
        out = torch.view_as_real(xc).reshape(*x.shape)
        return out.to(x.dtype)
        # cos_pos = self.cos[token_positions]
        # sin_pos = self.sin[token_positions]
        
        # x_reshaped = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        # x_even = x_reshaped[..., 0]
        # x_odd = x_reshaped[..., 1]
        # x_even_rotated = x_even * cos_pos - x_odd * sin_pos
        # x_odd_rotated = x_even * sin_pos + x_odd * cos_pos
        
        # x_rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)
        # return x_rotated.reshape(*x.shape[:-1], self.d_k)
        


# ----------------------------------------------------------------------------------
# §3.5.4  Softmax and Scaled Dot-Product Attention
# ----------------------------------------------------------------------------------

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax along `dim` using max-shift trick.

    Return tensor with same shape as `x`.
    """
    # TODO: subtract max along `dim`, exponentiate, normalize by sum
    x_shifted = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Shapes (broadcast batch dims allowed):
      Q: (..., q_len, d_k)
      K: (..., k_len, d_k)
      V: (..., k_len, d_v)
      mask (optional): (q_len, k_len) or broadcastable to scores

    Returns: (..., q_len, d_v)

    Notes
    -----
    - Compute scores = Q @ K^T / sqrt(d_k)
    - If `mask` is provided, set disallowed positions to -inf before softmax
    - Apply softmax over key dimension; output = P @ V
    """
    # TODO: implement attention per spec using the `softmax` above
    att_scores = einsum(Q, K, "... q_len d_k, ... k_len d_k -> ... q_len k_len") / math.sqrt(Q.shape[-1])
    if mask is not None:
        att_scores = att_scores.masked_fill(mask == 0, float("-inf"))
    att_scores = softmax(att_scores, dim=-1)
    return einsum(att_scores, V, "... q_len k_len, ... k_len d_v -> ... q_len d_v")


# ----------------------------------------------------------------------------------
# §3.5.2  Position-wise Feed-Forward (SwiGLU)
# ----------------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    FFN(x) = W2( SiLU(W1 x) ⊙ W3 x )
    with d_ff ≈ (8/3) * d_model, rounded to multiple of 64.
    No biases.
    """
    def __init__(self, d_model: int, d_ff: Optional[int] = None, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        if d_ff is None:
            raw = int(math.ceil((8.0/3.0) * d_model))
            # Round up to nearest multiple of 64
            d_ff = (raw + 63) // 64 * 64
        self.d_model = d_model
        self.d_ff = d_ff
        # Projections (no bias):
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D) -> (B, T, D)
        # TODO: s = SiLU(W1 x); g = W3 x; out = W2( s ⊙ g )
        w1_out = self.W1(x)
        s = w1_out * torch.sigmoid(w1_out)
        g = self.W3(x)
        return self.W2(s * g)


# ----------------------------------------------------------------------------------
# §3.5.5  Causal Multi-Head Self-Attention (with RoPE)
# ----------------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """Decoder-only causal MHA with RoPE on Q/K.

    Args
    ----
    d_model: model width
    num_heads: number of heads (d_k = d_v = d_model // num_heads)
    rope: optional RotaryPositionalEmbedding instance

    Inputs
    ------
    x: (B, T, D)
    token_positions: (B, T) or (T,) positions for RoPE (required if rope is not None)

    Returns
    -------
    y: (B, T, D)
    """
    def __init__(self, d_model: int, num_heads: int, rope: Optional[RotaryPositionalEmbedding] = None, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.rope = rope
        # Projections (pack heads along last dim):
        self.Wq = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.Wk = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.Wv = Linear(d_model, num_heads * self.d_v, device=device, dtype=dtype)
        self.Wo = Linear(num_heads * self.d_v, d_model, device=device, dtype=dtype)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        # Construct (T, T) causal mask: allow j <= i (past/current), block j > i (future)
        # scaled_dot_product_attention uses mask == 0 to set to -inf
        # So we want: mask[i, j] = 1 for j <= i (allow), 0 for j > i (block)
        # Lower triangular matrix: 1s on and below diagonal, 0s above
        return torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        # TODO: project to Q,K,V; reshape to (B, H, T, d_k/d_v)
        # Apply RoPE to Q and K per head if self.rope is not None
        # Build causal mask and apply scaled_dot_product_attention head-wise
        # Concatenate heads and project with Wo
        q = self.Wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.num_heads, self.d_v).transpose(1, 2)
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        mask = self._causal_mask(T, x.device)
        attn_output = scaled_dot_product_attention(q, k, v, mask)  # (B, H, T, d_v)
        # Transpose back to (B, T, H, d_v) and reshape to concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.num_heads * self.d_v)
        return self.Wo(attn_output)


# ----------------------------------------------------------------------------------
# §3.6  Transformer Block (pre-norm) and Full Transformer LM
# ----------------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Pre-norm Transformer block.

    Forward
    -------
    y = x + MHA( RMSNorm1(x) )
    z = y + FFN( RMSNorm2(y) )
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: Optional[int] = None, rope: Optional[RotaryPositionalEmbedding] = None, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        # TODO: pre-norm → MHA + residual; pre-norm → FFN + residual
        x = x + self.attn(self.norm1(x), token_positions)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer Language Model (Section §3 overview).

    Pipeline: token_ids → token embedding → [N × TransformerBlock] → final RMSNorm → LM head (linear → logits)

    Args
    ----
    vocab_size: size of tokenizer vocab
    context_length: max sequence length supported (for RoPE precompute)
    num_layers: number of blocks
    d_model: model width
    num_heads: heads per block
    d_ff: inner FF dimension (optional; defaults to ~8/3 d_model rounded to 64)
    theta: RoPE base (e.g., 10_000)
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        theta: float = 10_000.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_emb = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # Single shared RoPE for all blocks (optional optimization per spec):
        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_model // num_heads, max_seq_len=context_length, device=device)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=d_ff, rope=self.rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute next-token logits.

        token_ids: (B, T) int64
        token_positions: (B, T) or (T,) positions (required if using RoPE)
        returns logits: (B, T, vocab_size)
        """
        # TODO: embed; iterate blocks; final norm; lm_head; return logits
        # Embed tokens
        x = self.token_emb(token_ids)  # (B, T, d_model)
        
        # Create token positions if not provided
        if token_positions is None:
            T = token_ids.size(1)
            token_positions = torch.arange(T, device=token_ids.device, dtype=torch.long)
        
        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x, token_positions)
        
        # Apply final normalization and language model head
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        return logits
        

    # (Optional) inference helpers (left unimplemented)
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Generate new tokens autoregressively.
        
        Args:
            input_ids: Initial sequence of token IDs, shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1.0 = more deterministic, >1.0 = more random)
            top_k: If provided, only sample from top k most likely tokens
        
        Returns:
            Generated sequence including input_ids, shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()  # Set to evaluation mode
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Start with input_ids
        generated = input_ids.clone()
        
        # Generate new tokens one at a time
        for _ in range(max_new_tokens):
            # Check if we've reached context length limit
            current_length = generated.size(1)
            if current_length >= self.context_length:
                break
            
            # Get current sequence length (may need to truncate if too long)
            # For efficiency, we can use only the last context_length tokens
            if current_length > self.context_length:
                # Use sliding window: take last context_length tokens
                seq_to_use = generated[:, -self.context_length:]
                # Position indices start from (current_length - context_length)
                start_pos = current_length - self.context_length
            else:
                seq_to_use = generated
                start_pos = 0
            
            # Forward pass to get logits
            # Create position indices for RoPE
            seq_len = seq_to_use.size(1)
            token_positions = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=torch.long)
            
            logits = self.forward(seq_to_use, token_positions)  # (B, T, vocab_size)
            
            # Get logits for the last position (next token prediction)
            next_token_logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                # Get top-k values and indices
                top_k_values, top_k_indices = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)), dim=-1)
                
                # Create a mask: set non-top-k logits to -inf
                mask = torch.full_like(next_token_logits, float('-inf'))
                mask.scatter_(-1, top_k_indices, top_k_values)
                next_token_logits = mask
            
            # Convert to probabilities using softmax
            probs = softmax(next_token_logits, dim=-1)  # (B, vocab_size)
            
            # Sample from the distribution
            # Use multinomial sampling
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)  # (B, seq_len + 1)
        
        return generated
