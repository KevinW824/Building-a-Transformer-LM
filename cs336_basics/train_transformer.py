from __future__ import annotations

import math
import os
from typing import Optional, Any
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import torch
from torch import nn, Tensor
from torch.optim import Optimizer

from cs336_basics.transformer import TransformerLM

# Note: The functions in this file are the actual implementations.
# The adapter functions in tests/adapters.py should call these functions
# to match the expected signatures for testing.


def cross_entropy_loss(
    logits: Tensor,  # (batch_size, seq_len, vocab_size) or (batch_size * seq_len, vocab_size)
    targets: Tensor,  # (batch_size, seq_len) or (batch_size * seq_len,)
) -> Tensor:
    """Compute cross-entropy loss for language modeling.
    
    Args:
        logits: Unnormalized log probabilities. Shape can be (B, T, V) or (B*T, V)
        targets: Target token indices. Shape can be (B, T) or (B*T,)
    
    Returns:
        Scalar tensor containing the average cross-entropy loss.
    
    Notes:
        - Should handle numerical stability (use log-sum-exp trick)
        - Should average over all positions in the sequence
        - For language modeling, targets are typically shifted by one position
    """
    # Flatten logits and targets if needed (handle both (B, T, V) and (B*T, V) shapes)
    if logits.dim() > 2:
        # Reshape to (B*T, V) and (B*T,)
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
    
    # Subtract max for numerical stability: logits_shifted = logits - max(logits)
    logits_shifted = logits - logits.max(dim=-1, keepdim=True).values
    
    # Compute log-sum-exp: log(sum(exp(logits_shifted)))
    # This is the normalization term in log-softmax
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_shifted), dim=-1))
    
    # Get logits for target indices: logits_shifted[targets]
    # Use gather to select the logit value for each target
    logits_for_targets = logits_shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # Cross-entropy: -log(softmax(logits)[targets])
    # = -log(exp(logits[targets]) / sum(exp(logits)))
    # = -logits[targets] + log(sum(exp(logits)))
    # = -logits_for_targets + log_sum_exp
    loss = -logits_for_targets + log_sum_exp
    
    # Return average loss across all batch dimensions
    return loss.mean()


def get_batch(
    dataset: npt.NDArray[np.int64],
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample a batch of sequences from the dataset.
    
    Args:
        dataset: 1D numpy array of token IDs
        batch_size: Number of sequences to sample
        context_length: Length of each sequence
        device: Device to place tensors on (e.g., 'cpu' or 'cuda:0')
    
    Returns:
        Tuple of (input_ids, target_ids) tensors, each of shape (batch_size, context_length)
        For language modeling, target_ids are typically input_ids shifted by 1 position
    
    Notes:
        - Randomly sample starting positions for each sequence in the batch
        - Ensure sequences don't go out of bounds
        - Return LongTensors
    """
    # Calculate valid range for starting indices
    # We need to ensure we can extract a full sequence of length context_length
    # Maximum starting index: len(dataset) - context_length (exclusive)
    max_start_idx = len(dataset) - context_length
    
    # Sample random starting indices for each sequence in the batch
    # Each index is in [0, max_start_idx)
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # Extract sequences from dataset
    # For each starting index i, extract dataset[i:i+context_length]
    input_ids_list = []
    target_ids_list = []
    
    for start_idx in start_indices:
        # Extract input sequence: dataset[start_idx:start_idx+context_length]
        input_seq = dataset[start_idx:start_idx + context_length]
        
        # Extract target sequence: dataset[start_idx+1:start_idx+context_length+1]
        # This gives us the next token for each position (shifted by 1)
        target_seq = dataset[start_idx + 1:start_idx + context_length + 1]
        
        input_ids_list.append(input_seq)
        target_ids_list.append(target_seq)
    
    # Convert to numpy arrays and then to PyTorch tensors
    input_ids = torch.from_numpy(np.array(input_ids_list)).long()
    target_ids = torch.from_numpy(np.array(target_ids_list)).long()
    
    # Move to specified device
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    
    return input_ids, target_ids


class AdamW(Optimizer):
    """AdamW optimizer implementation.
    
    AdamW decouples weight decay from gradient-based updates, unlike Adam.
    This is the optimizer used in most modern transformer training.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
    
    Notes:
        - Should implement the AdamW algorithm from "Decoupled Weight Decay Regularization"
        - Maintain running averages of gradients and squared gradients
        - Apply weight decay separately from gradient updates
        - Handle bias correction for first and second moment estimates
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step.
        
        Args:
            closure: Optional closure that reevaluates the model and returns loss
        
        Returns:
            Loss value if closure is provided, None otherwise
        """
        if closure is not None:
            loss = closure()
        
        # Increment step count
        # Note: We use a shared step count, but PyTorch typically tracks per-parameter steps
        # For simplicity, we'll use a global step count
        step_count = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get optimizer hyperparameters
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                
                # Initialize state for this parameter if needed
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)  # First moment (m)
                    state['exp_avg_sq'] = torch.zeros_like(p)  # Second moment (v)
                
                # Get gradient
                grad = p.grad
                
                # Get current state
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                step_count = state['step']
                
                # Step 1: Apply weight decay (decoupled from gradient)
                # In AdamW, weight decay is applied directly to parameters, not gradients
                p.mul_(1 - lr * weight_decay)
                
                # Step 2: Update biased first moment estimate
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Step 3: Update biased second raw moment estimate
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Step 4: Bias correction
                # m_hat_t = m_t / (1 - beta1^t)
                # v_hat_t = v_t / (1 - beta2^t)
                bias_correction1 = 1 - beta1 ** step_count
                bias_correction2 = 1 - beta2 ** step_count
                
                # Step 5: Compute update
                # p_t = p_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss if closure is not None else None
        
        


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer (with optional momentum).
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0.0 for vanilla SGD)
        weight_decay: Weight decay (L2 penalty) coefficient (default: 0.0)
    
    Notes:
        - Simpler than AdamW but often less effective for transformers
        - Can be useful for fine-tuning or specific training schedules
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        # TODO: Initialize optimizer state
        # Hint: May need to track velocity for momentum
    
    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step.
        
        Args:
            closure: Optional closure that reevaluates the model and returns loss
        
        Returns:
            Loss value if closure is provided, None otherwise
        """
        # TODO: Implement SGD update step
        # Hint:
        # 1. For each parameter group:
        #    - Get learning rate, momentum, weight_decay
        #    - For each parameter:
        #      a. Apply weight decay if specified
        #      b. Update with momentum if momentum > 0
        #      c. Update parameter: p = p - lr * (grad + weight_decay * p)
        raise NotImplementedError


def get_cosine_lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Compute learning rate using cosine annealing with warmup.
    
    Schedule:
    - For iterations < warmup_iters: Linear warmup from 0 to max_learning_rate
    - For iterations >= warmup_iters: Cosine decay from max_learning_rate to min_learning_rate
    
    Args:
        it: Current iteration number (0-indexed)
        max_learning_rate: Maximum learning rate (alpha_max)
        min_learning_rate: Minimum learning rate (alpha_min)
        warmup_iters: Number of warmup iterations (T_w)
        cosine_cycle_iters: Number of cosine annealing iterations (T_c)
    
    Returns:
        Learning rate for the current iteration
    
    Notes:
        - During warmup: lr = max_lr * (it / warmup_iters)
        - After warmup: lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * progress))
        - Progress after warmup: (it - warmup_iters) / cosine_cycle_iters
    """
    # Phase 1: Linear warmup from 0 to max_learning_rate
    if it < warmup_iters:
        # Linear interpolation: lr = max_lr * (it / warmup_iters)
        return max_learning_rate * (it / warmup_iters)
    
    # Phase 2: Cosine annealing from max_learning_rate to min_learning_rate
    # Compute progress through cosine cycle (0 to 1)
    # The cosine cycle goes from iteration warmup_iters to iteration cosine_cycle_iters
    # So the cycle length is (cosine_cycle_iters - warmup_iters)
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    
    # Clamp progress to [0, 1] to handle iterations beyond cosine_cycle_iters
    # This ensures lr stays at min_learning_rate after the cycle completes
    progress = min(progress, 1.0)
    
    # Cosine annealing formula:
    # lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * progress))
    # When progress=0: cos(0)=1, so lr = min_lr + (max_lr-min_lr)*0.5*2 = max_lr
    # When progress=1: cos(π)=-1, so lr = min_lr + (max_lr-min_lr)*0.5*0 = min_lr
    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_factor


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """Clip gradients to have maximum L2 norm.
    
    Args:
        parameters: Iterable of parameters whose gradients to clip
        max_l2_norm: Maximum allowed L2 norm of all gradients
    
    Notes:
        - Computes the L2 norm of all gradients combined
        - If norm > max_l2_norm, scales all gradients by (max_l2_norm / actual_norm)
        - Modifies gradients in-place
        - Formula: If ||g|| > max_l2_norm, then g = g * (max_l2_norm / ||g||)
    """
    # Collect all gradients (skip parameters without gradients)
    grads = []
    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad)
    
    # If no gradients, return early
    if len(grads) == 0:
        return
    
    # Compute the L2 norm of all gradients combined
    # L2 norm = sqrt(sum(g^2 for all gradients))
    # We compute sum of squares first, then take square root
    total_norm_squared = sum(grad.pow(2).sum() for grad in grads)
    total_norm = math.sqrt(total_norm_squared.item())
    
    # If norm exceeds max_l2_norm, scale all gradients
    # Scale factor: clip_coef = max_l2_norm / total_norm
    # Then: grad = grad * clip_coef
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / total_norm
        # Scale all gradients in-place
        for grad in grads:
            grad.mul_(clip_coef)


def train_transformer(
    model: TransformerLM,
    train_dataset: npt.NDArray[np.int64],
    optimizer: Optimizer,
    num_iterations: int,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    max_l2_norm: Optional[float] = None,
    lr_schedule_fn: Optional[callable] = None,
    eval_dataset: Optional[npt.NDArray[np.int64]] = None,
    eval_interval: Optional[int] = None,
    checkpoint_interval: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
) -> dict[str, list]:
    """Main training loop for transformer language model.
    
    Args:
        model: TransformerLM model to train
        train_dataset: Training dataset as 1D array of token IDs
        optimizer: Optimizer (AdamW or SGD)
        num_iterations: Number of training iterations
        batch_size: Batch size
        context_length: Sequence length
        device: Device to train on ('cpu' or 'cuda:0')
        max_l2_norm: Maximum gradient norm for clipping (None = no clipping)
        lr_schedule_fn: Function that takes iteration number and returns learning rate
                        If None, uses fixed learning rate from optimizer
        eval_dataset: Optional validation dataset for evaluation
        eval_interval: Evaluate model every N iterations (None = no evaluation)
        checkpoint_interval: Save checkpoint every N iterations (None = no checkpoints)
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        Dictionary with training history:
        {
            'train_loss': [loss1, loss2, ...],
            'eval_loss': [eval_loss1, eval_loss2, ...] (if eval_interval provided),
            'iterations': [iter1, iter2, ...]
        }
    
    Notes:
        - Uses model.train() and model.eval() appropriately
        - Should handle device placement
        - Should update learning rate if lr_schedule_fn is provided
        - For evaluation, computes loss on eval_dataset without gradients
    """
    model = model.to(device)
    model.train()
    
    history = {
        'train_loss': [],
        'eval_loss': [],
        'iterations': [],
    }
    
    # Training loop
    for iteration in range(num_iterations):
        # 1. Sample batch
        input_ids, target_ids = get_batch(train_dataset, batch_size, context_length, device)
        
        # 2. Forward pass
        logits = model(input_ids)  # (B, T, vocab_size)
        
        # 3. Compute loss
        # For language modeling, we predict the next token at each position
        # get_batch returns:
        #   - input_ids: [x0, x1, ..., xT-1] 
        #   - target_ids: [x1, x2, ..., xT] (already shifted by 1)
        # At position t, logits[:, t, :] predicts the next token after input_ids[:, t]
        # So logits[:, t, :] should predict target_ids[:, t] (which is x_{t+1})
        # We exclude the last position since we don't have a target for it
        loss = cross_entropy_loss(
            logits[:, :-1].reshape(-1, logits.size(-1)), 
            target_ids[:, :-1].reshape(-1)
        )
        
        # 4. Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 5. Gradient clipping (if specified)
        if max_l2_norm is not None:
            clip_gradients(model.parameters(), max_l2_norm)
        
        # 6. Update learning rate (if schedule provided)
        if lr_schedule_fn is not None:
            lr = lr_schedule_fn(iteration)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # 7. Optimizer step
        optimizer.step()
        
        # 8. Logging
        history['train_loss'].append(loss.item())
        history['iterations'].append(iteration)
        
        # 9. Evaluation (if specified)
        if eval_dataset is not None and eval_interval is not None:
            if iteration % eval_interval == 0:
                eval_loss = evaluate_model(model, eval_dataset, batch_size, context_length, device)
                history['eval_loss'].append(eval_loss)
        
        # 10. Checkpointing (if specified)
        if checkpoint_interval is not None and checkpoint_dir is not None:
            if iteration % checkpoint_interval == 0:
                # Create checkpoint filename with iteration number
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt")
                save_checkpoint(model, optimizer, iteration, checkpoint_path)
    
    return history


def evaluate_model(
    model: TransformerLM,
    dataset: npt.NDArray[np.int64],
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    num_eval_batches: int = 100,
) -> float:
    """Evaluate model on a dataset.
    
    Args:
        model: TransformerLM model
        dataset: Evaluation dataset as 1D array of token IDs
        batch_size: Batch size for evaluation
        context_length: Sequence length
        device: Device to evaluate on
        num_eval_batches: Number of batches to evaluate on
    
    Returns:
        Average loss over evaluation batches
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Evaluation loop (no gradients needed)
    with torch.no_grad():
        for _ in range(num_eval_batches):
            # Sample batch
            input_ids, target_ids = get_batch(dataset, batch_size, context_length, device)
            
            # Forward pass
            logits = model(input_ids)  # (B, T, vocab_size)
            
            # Compute loss (same as training)
            loss = cross_entropy_loss(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                target_ids[:, :-1].reshape(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    # Return average loss
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(
    model: TransformerLM,
    optimizer: Optimizer,
    iteration: int,
    checkpoint_path: str | os.PathLike,
) -> None:
    """Save training checkpoint.
    
    Args:
        model: TransformerLM model
        optimizer: Optimizer state
        iteration: Current iteration number
        checkpoint_path: Path to save checkpoint (str or PathLike)
    
    Notes:
        - Saves model state dict, optimizer state dict, and iteration number
        - Uses torch.save() which can handle both file paths and file-like objects
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    model: TransformerLM,
    optimizer: Optimizer,
    checkpoint_path: str | os.PathLike,
) -> int:
    """Load training checkpoint.
    
    Args:
        model: TransformerLM model
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint file (str or PathLike)
    
    Returns:
        Iteration number from checkpoint
    
    Notes:
        - Loads model state dict and optimizer state dict from checkpoint
        - Restores model and optimizer to the saved state
        - Returns the iteration number that was saved
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state dict
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Return iteration number
    return checkpoint['iteration']

