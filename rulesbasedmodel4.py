import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import json
import threading
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from transformers import PreTrainedTokenizerFast, AddedToken
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders
import copy
import random
import math
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import re
import pandas as pd
import typing
from torch.optim import Optimizer
import torch.utils.checkpoint as checkpoint
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import subprocess
import gc

torch.backends.cudnn.enabled = True  # Ensures cuDNN is enabled
torch.backends.cudnn.benchmark = True  # Optimizes performance
# Allow PyTorch to use system RAM when GPU memory overflows

os.environ["NCCL_P2P_DISABLE"] = "1"

def safe_tensor_allocation(tensor):
    try:
        return tensor.to("cuda", non_blocking=True)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("⚠️ GPU memory full! Offloading to CPU RAM.")
            return tensor.to("cpu")
        else:
            raise e

## set threads
torch.set_num_threads(32)
torch.set_num_interop_threads(32)

# Debug for CUDA
print(f"CUDA Available: {torch.cuda.is_available()}")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_LOGINFO_DBG"]= "0"
os.environ["CUBLAS_LOG_LEVEL"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
tokenizer = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

seq_len = 50
pad_token_id = 0
set_number_rules = 10000

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Uses a quintic iteration optimized for stability in low precision.
    """
    #print(f"Before NS: {G.shape}")

    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum Orthogonalized by Newton-Schulz

    A distributed-friendly optimizer that applies momentum-based updates and
    orthogonalization post-processing. Works on multi-GPU setups, but can also run
    in single-GPU mode by bypassing distributed operations.

    Arguments:
        lr: Learning rate.
        weight_decay: Weight decay (L2 regularization).
        momentum: Momentum coefficient.
        nesterov: Use Nesterov-style momentum.
        ns_steps: Number of Newton-Schulz iterations.
        world_size: Number of GPUs used for distributed training.
        rank: Rank of the current process (set automatically in DDP).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5):
        # Detect whether distributed training is initialized
        self.ddp_enabled = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.ddp_enabled else 1
        self.rank = dist.get_rank() if self.ddp_enabled else 0

        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)

        param_groups = []
        for size in {p.numel() for p in params}:
            # 🔹 Only create distributed buffers if DDP is enabled
            if self.ddp_enabled:
                b = torch.empty(self.world_size, size, dtype=torch.bfloat16, device="cuda")
                group = dict(params=[p for p in params if p.numel() == size],
                             update_buffer=b, update_buffer_views=[b[i] for i in range(self.world_size)])
            else:
                group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)

        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params = group["params"]

            if self.ddp_enabled:
                update_buffer: torch.Tensor = group["update_buffer"]
                update_buffer_views: list[torch.Tensor] = group["update_buffer_views"]
                handle = None
                params_world = None

            def update_prev():
                """Distributed update processing (only if DDP is enabled)."""
                if self.ddp_enabled:
                    handle.wait()
                    for p_world, g_world in zip(params_world, update_buffer_views):
                        p_world.mul_(1 - group["lr"] * group["weight_decay"])
                        p_world.add_(g_world.view_as(p_world),
                                    alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)

            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]

                    # Initialize momentum buffer if not already present
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)

                    buf: torch.Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

                    # Handle convolutional filters
                    if g.ndim == 4:
                        g = g.view(len(g), -1)

                    # 🔹 DEBUG: Print before Newton-Schulz
                    #print(f"🔍 Before NS: {g.shape} (Original param shape: {p.shape})")

                    # 🔹 Fix potential reshape issue before NS
                    if g.ndim == 3:
                        g = g.view(g.shape[0], -1, g.shape[-1])  # Reshape 3D to 2D
                    elif g.ndim > 3:
                        g = g.view(g.shape[0], g.shape[1], -1)  # Handle extra dimensions

                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                    #print(f"✅ After NS: {g.shape}")

                else:
                    g = update_buffer_views[self.rank] if self.ddp_enabled else None

                # Handle distributed processing (skip if single GPU)
                if self.ddp_enabled:
                    if base_i > 0:
                        update_prev()
                    handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                    params_world = params[base_i: base_i + self.world_size]
                else:
                    # Apply updates directly if single-GPU
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

            if self.ddp_enabled:
                update_prev()


def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    torch.save(checkpoint, path)
    
def save_checkpoint_rules(model, optimizer, epoch, path, num_saved_rules):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'num_saved_rules': num_saved_rules  # 🔹 Save the number of rules
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    
    # 🔹 Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # 🔹 Load optimizer state if provided
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=False)

    # 🔹 Load number of saved rules and update model
    num_saved_rules = checkpoint.get('num_saved_rules', 1)  # Default to 1 if not found
    if hasattr(model, 'embedding') and isinstance(model.embedding, DynamicRuleEmbedding):
        model.embedding.current_rule_count = num_saved_rules  # 🔹 Restore rule count
        print(f"✅ Loaded model with {num_saved_rules} rules!")

    return checkpoint['epoch'], checkpoint['phase'], num_saved_rules


def init_tokenizer(tokenizer_path):
    global tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    logging.info(f"Tokenizer pad_token set to: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")


def tokenize_chunk(chunk):
    # Tokenizer is now the global variable initialized in each process
    encoded = tokenizer(chunk, return_attention_mask=False, truncation=True, max_length=seq_len)
    return encoded['input_ids']


# In your collate_fn, specify device when creating new tensors:
def collate_fn(batch):
    padded_inputs = []
    padded_targets = []
    fixed_length = seq_len
    for input_ids, tot, tgt in batch:
        combined_target = torch.cat((tot, tgt), dim=0)
        if combined_target.size(0) > fixed_length:
            combined_target = combined_target[:fixed_length]
        else:
            pad_len = fixed_length - combined_target.size(0)
            combined_target = torch.cat(
                (combined_target, torch.full((pad_len,), pad_token_id, dtype=torch.long, device=device)),
                dim=0
            )
        padded_targets.append(combined_target)
        
        if input_ids.size(0) > fixed_length:
            padded_input = input_ids[:fixed_length]
        else:
            pad_len = fixed_length - input_ids.size(0)
            padded_input = torch.cat(
                (input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long, device=device)),
                dim=0
            )
        padded_inputs.append(padded_input)
    
    padded_inputs = torch.stack(padded_inputs).to(device)
    padded_targets = torch.stack(padded_targets).to(device)
    return padded_inputs, padded_targets
    
class ChunkedDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data_path, tokenizer, max_length=seq_len):
        self.tokenized_data_path = tokenized_data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Get a list of chunk files
        self.chunk_files = [os.path.join(self.tokenized_data_path, f) 
                            for f in os.listdir(self.tokenized_data_path) 
                            if f.startswith('chunk_') and f.endswith('.jsonl')]
        self.chunk_files.sort()  # Ensure the chunks are in order

        # Build an index mapping from global indices to (chunk_idx, sample_idx)
        self.index_mapping = []
        for chunk_idx, chunk_file in enumerate(self.chunk_files):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
            self.index_mapping.extend([(chunk_idx, i) for i in range(num_lines)])

        # Initialize current chunk data
        self.current_chunk_idx = -1  # Indicates no chunk is currently loaded
        self.current_chunk_data = []  # Will hold the data from the current chunk

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):

        if idx < 0 or idx >= len(self.index_mapping):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_mapping)}")

        chunk_idx, sample_idx = self.index_mapping[idx]

        # Load the appropriate chunk if not already loaded
        if self.current_chunk_idx != chunk_idx:
            self.load_chunk(chunk_idx)

        record = self.current_chunk_data[sample_idx]
        input_ids = record['input_ids']
        labels = record['labels']

        # Calculate original sequence length before padding
        original_seq_length = min(len(input_ids), self.max_length)
        logging.debug(f"original sequence length = {original_seq_length}")
        # Pad sequences to max_length
        input_ids = input_ids[:self.max_length] + [self.tokenizer.pad_token_id] * max(0, self.max_length - len(input_ids))
        labels = labels[:self.max_length] + [self.tokenizer.pad_token_id] * max(0, self.max_length - len(labels))

        assert isinstance(input_ids, list), "input_ids should be a list"
        assert isinstance(labels, list), "labels should be a list"
        assert all(isinstance(id, int) for id in input_ids), "All input_ids should be integers"
        assert all(isinstance(id, int) for id in labels), "All labels should be integers"
        assert len(input_ids) == self.max_length, "input_ids should be padded to max_length"
        assert len(labels) == self.max_length, "labels should be padded to max_length"
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        seq_lengths = torch.tensor(original_seq_length, dtype=torch.long)

        # Check for empty sequences
        if len(input_ids) == 0:
            logging.error(f"Empty input_ids at index {idx}.")
            raise ValueError(f"Empty input_ids at index {idx}.")
        if len(labels) == 0:
            logging.error(f"Empty labels at index {idx}.")
            raise ValueError(f"Empty labels at index {idx}.")
    
        return input_ids, attention_mask, labels, seq_lengths

    def load_chunk(self, idx):
        if not self.chunk_files:
            raise ValueError("⚠️ No chunk files found!")
        chunk_file = self.chunk_files[idx]
        if not os.path.exists(chunk_file):
            raise FileNotFoundError(f"Chunk file not found: {chunk_file}")
        with open(chunk_file, 'r', encoding='utf-8') as f:
            try:
                self.current_chunk_data = [json.loads(line.strip()) for line in f]
            except json.JSONDecodeError:
                print(f"⚠️ Skipping corrupted chunk file: {chunk_file}")
        self.current_chunk_idx = idx

# Generate src mask function
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_combined_mask(batch_input_ids, pad_token_id):
    """
    Create a combined attention mask that incorporates both the causal (subsequent) mask
    and the padding mask. This function ensures that each row has at least one valid token.
    """
    batch_size, seq_length = batch_input_ids.size()
    device = batch_input_ids.device
    
    # Generate causal (subsequent) mask: shape (seq_len, seq_len)
    causal_mask = generate_square_subsequent_mask(seq_len).to(device)
    logging.debug(f"Shape of causal_mask before expand: {causal_mask.shape}")

    # Expand to batch dimension: (batch_size, seq_len, seq_len)
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    logging.debug(f"Shape of causal_mask after expansion: {causal_mask.shape}")
    # Create padding mask: valid tokens are True, padded tokens are False.
    # Shape: (batch_size, seq_len)
    padding_mask = (batch_input_ids != pad_token_id)
    # Expand padding mask to match the shape (batch_size, seq_len, seq_len)
    # Here we broadcast along one dimension so that we mask out positions in each row.
    padding_mask_expanded = padding_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
    logging.debug(f"Shape of padding_mask after expansion: {padding_mask_expanded.shape}")

    # Combine masks: where padding_mask is False, set to -inf.
    # This keeps the causal structure while ensuring that padded positions are fully masked.
    combined_mask = causal_mask.masked_fill(~padding_mask_expanded, float('-inf'))
    logging.debug(f"Shape of combined_mask after fill: {combined_mask.shape}")

    # Check each row: if an entire row is -inf, force the first token (or a designated position) to be valid.
    for i in range(batch_size):
        for j in range(seq_len):
            if torch.all(combined_mask[i, j] == float('-inf')):
                combined_mask[i, j, 0] = 0.0  # Force at least one valid position
    
    return combined_mask


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent):
        """
        Multi-Head Latent Attention (MHLA)
        - d_model: Input feature dimension
        - num_heads: Number of attention heads
        - d_latent: Compressed latent space dimension
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent

        # Standard attention projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Latent compression & reconstruction
        self.W_down_kv = nn.Linear(d_model, d_latent, bias=False)  # Compress keys/values
        self.W_up_k = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct keys
        self.W_up_v = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct values

    def forward(self, x, memory=None):
        """
        Forward pass with optional memory (for hierarchical tokenization)
        - x: Input tensor (batch, seq_len, d_model)
        - memory: Cached latent state (batch, d_latent) [optional]
        """
        batch_size, seq_len, _ = x.shape

        # Compute queries, keys, values
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Latent compression for keys and values
        latent_kv = self.W_down_kv(k + v)  # Merge and compress
        if memory is not None:
            latent_kv = (latent_kv + memory) / 2  # Combine with previous memory

        # Reconstruct full-size keys and values
        k_reconstructed = self.W_up_k(latent_kv)
        v_reconstructed = self.W_up_v(latent_kv)

        # Multi-head split
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k_reconstructed = k_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v_reconstructed = v_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k_reconstructed.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_reconstructed)

        # Merge attention heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final projection
        output = self.W_o(attn_output)

        return output, latent_kv  # Return output and memory for next layer


class TimeAwareMultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, lambda_decay=0.01):
        """
        Multi-Head Latent Attention (MHLA) with Time-Aware Decay.
        - d_model: Input feature dimension
        - num_heads: Number of attention heads
        - d_latent: Compressed latent space dimension
        - lambda_decay: Controls how quickly attention fades over time
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent
        self.lambda_decay = lambda_decay

        # Standard attention projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Latent compression & reconstruction
        self.W_down_kv = nn.Linear(d_model, d_latent, bias=False)  # Compress keys/values
        self.W_up_k = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct keys
        self.W_up_v = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct values

    def forward(self, x, memory=None):
        """
        Forward pass with optional hierarchical memory.
        - x: Input tensor (batch, seq_len, d_model)
        - memory: Cached latent state (batch, d_latent) [optional]
        """
        batch_size, seq_len, _ = x.shape

        # Compute queries, keys, values
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Latent compression for keys and values
        latent_kv = self.W_down_kv(k + v)  # Merge and compress
        if memory is not None:
            latent_kv = (latent_kv + memory) / 2  # Combine with previous memory

        # Reconstruct full-size keys and values
        k_reconstructed = self.W_up_k(latent_kv)
        v_reconstructed = self.W_up_v(latent_kv)

        # Multi-head split
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k_reconstructed = k_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v_reconstructed = v_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # Compute raw attention scores
        attn_scores = torch.matmul(q, k_reconstructed.transpose(-2, -1)) / math.sqrt(self.d_model)

        # 🔹 Apply time decay to attention scores
        time_matrix = torch.arange(seq_len, device=x.device).float().unsqueeze(0).expand(seq_len, seq_len)
        time_decay = torch.exp(-self.lambda_decay * torch.abs(time_matrix - time_matrix.T))  # e^(-λt)
        attn_scores = attn_scores * time_decay.unsqueeze(0).unsqueeze(0)  # Shape: (batch, heads, seq, seq)

        # Normalize attention scores
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_reconstructed)

        # Merge attention heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final projection
        output = self.W_o(attn_output)

        return output, latent_kv  # Return output and hierarchical memory

class HierarchicalMultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, lambda_decay=0.01, memory_size=16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent
        self.lambda_decay = lambda_decay
        self.memory_size = memory_size  # How many past summaries to retain
        self.memory = []  # Stores hierarchical memory embeddings

        # Ensure `d_model` is evenly divisible by `num_heads`
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.head_dim = d_model // num_heads  # Compute per-head dimension

        # Standard attention components
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Latent compression & reconstruction
        self.W_down_kv = nn.Linear(d_model, d_latent, bias=False)
        self.W_up_k = nn.Linear(d_latent, d_model, bias=False)
        self.W_up_v = nn.Linear(d_latent, d_model, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Compute queries, keys, values
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Latent compression for keys and values
        latent_kv = self.W_down_kv(k + v)

        # 🔹 Fix: Ensure Latent Memory Doesn't Accumulate Unexpectedly
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)  # Keep memory size consistent
        self.memory.append(latent_kv.mean(dim=1))  # Store compressed memory state

        # Reconstruct keys and values
        k_reconstructed = self.W_up_k(latent_kv)
        v_reconstructed = self.W_up_v(latent_kv)

        # 🔹 Fix: Ensure Shape Matches Expected Multi-Head Attention Shape
        try:
            k_reconstructed = k_reconstructed.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v_reconstructed = v_reconstructed.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        except RuntimeError as e:
            print(f"Error reshaping k/v in MHLA: {e}")
            print(f"Shape mismatch: batch={batch_size}, seq_len={seq_len}, num_heads={self.num_heads}, head_dim={self.head_dim}")
            raise e

        # Compute attention
        attn_scores = torch.matmul(q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), 
                                   k_reconstructed.transpose(-2, -1)) / math.sqrt(self.d_model)

        # Apply time decay
        time_matrix = torch.arange(seq_len, device=x.device).float().unsqueeze(0).expand(seq_len, seq_len)
        time_decay = torch.exp(-self.lambda_decay * torch.abs(time_matrix - time_matrix.T))
        attn_scores = attn_scores * time_decay.unsqueeze(0).unsqueeze(0)

        # Normalize and compute attention output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_reconstructed)

        # Merge heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)

        return output

class FourierEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_frequencies=100, device=device):
        """
        Fourier-Based Embedding Layer
        - vocab_size: Number of tokens
        - embedding_dim: Desired embedding size
        - num_frequencies: Number of Fourier components used (must match embedding_dim or be projected)
        - device: Ensures tensors are on the correct device
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_frequencies = num_frequencies
        self.device = device

        # Learnable Fourier coefficients for sine and cosine
        self.a_n = nn.Parameter(torch.randn(vocab_size, num_frequencies, device=device))
        self.b_n = nn.Parameter(torch.randn(vocab_size, num_frequencies, device=device))

        # Frequency scaling factors (move to device)
        self.freqs = torch.linspace(1, num_frequencies, num_frequencies, device=device).view(1, -1)

        # 🔹 Projection layer to ensure output matches `embedding_dim`
        self.projection = nn.Linear(num_frequencies, embedding_dim)

    def forward(self, token_ids):
        """
        Generate embeddings dynamically using Fourier Series
        - token_ids: Tensor of token indices (batch, seq_len)
        """
        batch_size, seq_len = token_ids.shape

        # Normalize token IDs to continuous space
        x = token_ids.float().unsqueeze(-1) / self.vocab_size  # Shape: (batch, seq_len, 1)

        # Ensure `self.freqs` is on the same device as token_ids
        self.freqs = self.freqs.to(token_ids.device)

        # Compute Fourier embedding
        cos_terms = torch.cos(2 * math.pi * self.freqs * x)  # (batch, seq_len, num_frequencies)
        sin_terms = torch.sin(2 * math.pi * self.freqs * x)  # (batch, seq_len, num_frequencies)

        # Multiply by learnable coefficients
        embedding = (self.a_n[token_ids] * cos_terms + self.b_n[token_ids] * sin_terms)  # (batch, seq_len, num_frequencies)

        # 🔹 Ensure output size matches `embedding_dim` by projecting
        embedding = self.projection(embedding)  # (batch, seq_len, embedding_dim)

        return embedding


class RuleBasedEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_rules=set_number_rules, device="cpu"):
        """
        Rule-Based Embedding Layer
        - vocab_size: Number of tokens
        - embedding_dim: Size of each embedding
        - num_rules: Number of transformation rules stored
        - device: Ensures tensors are on the correct device
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_rules = num_rules
        self.device = device

        # 🔹 Base embeddings (randomly initialized, but modifiable)
        self.base_embeddings = nn.Parameter(torch.randn(vocab_size, embedding_dim, device=device))

        # 🔹 Learnable rule transformation matrix
        self.rule_transform = nn.Parameter(torch.randn(num_rules, embedding_dim, embedding_dim, device=device))

        # 🔹 Rule index per token (learns which rule applies to each token)
        self.token_rules = nn.Parameter(torch.randint(0, num_rules, (vocab_size,), device=device), requires_grad=False)

    def forward(self, token_ids):
        """
        Generates embeddings dynamically using transformation rules.
        - token_ids: Tensor of token indices (batch, seq_len)
        """
        batch_size, seq_len = token_ids.shape

        # Fetch base embeddings
        base_embeds = self.base_embeddings[token_ids]  # Shape: (batch, seq_len, embedding_dim)

        # Retrieve rules for each token
        rule_indices = self.token_rules[token_ids]  # Shape: (batch, seq_len)
        rule_matrices = self.rule_transform[rule_indices]  # Shape: (batch, seq_len, embedding_dim, embedding_dim)

        # Apply transformation rules to embeddings
        transformed_embeds = torch.einsum("bsd,bsde->bse", base_embeds, rule_matrices)

        return transformed_embeds

class DynamicRuleEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_frequencies=100, max_rules=set_number_rules, device=device):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_rules = max_rules
        self.num_frequencies = num_frequencies
        self.device = device

        # 🔹 Fourier Base Embeddings
        self.fourier_freqs = torch.linspace(1, num_frequencies, num_frequencies, device=device).view(1, -1)
        self.a_n = nn.Parameter(torch.randn(vocab_size, num_frequencies, device=device))
        self.b_n = nn.Parameter(torch.randn(vocab_size, num_frequencies, device=device))

        # 🔹 Rule Transformation Matrix (Start With 1 Rule, Expand Over Time)
        self.rule_transform = nn.Parameter(torch.randn(1, embedding_dim, embedding_dim, device=device))  # Start with 1 rule

        # 🔹 Store token-rule mappings as a non-trainable buffer
        self.register_buffer("token_rules", torch.randint(0, 1, (vocab_size,), device=device))  # Start with 1 rule index
        self.register_buffer("rule_scores", torch.zeros(1, device=device))  # Start with 1 score
        self.current_rule_count = 1  # Track number of active rules

        # 🔹 Projection layer to ensure Fourier output matches `embedding_dim`
        self.projection = nn.Linear(num_frequencies, embedding_dim)


    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape

        # 🔹 Compute Fourier Base Embedding
        x = token_ids.float().unsqueeze(-1) / self.vocab_size  # Normalize token ID
        cos_terms = torch.cos(2 * torch.pi * self.fourier_freqs * x)
        sin_terms = torch.sin(2 * torch.pi * self.fourier_freqs * x)
        fourier_embedding = (self.a_n[token_ids] * cos_terms + self.b_n[token_ids] * sin_terms)

        # 🔹 Project to `embedding_dim`
        base_embeds = self.projection(fourier_embedding)

        # 🔹 Retrieve rules for each token
        rule_indices = self.token_rules[token_ids]  # Shape: (batch, seq_len)
        rule_matrices = self.rule_transform[rule_indices]  # Shape: (batch, seq_len, embedding_dim, embedding_dim)

        # 🔹 Apply rule transformation
        transformed_embeds = torch.einsum("bsd,bsde->bse", base_embeds, rule_matrices)

        return transformed_embeds

    def update_rule_scores(self, token_ids, loss_diff):
        """
        Updates rule effectiveness scores based on loss reduction.
        - token_ids: Tokens involved in the rule transformation
        - loss_diff: Change in loss after applying rule (scalar)
        """
        rule_indices = self.token_rules[token_ids].detach()
        self.rule_scores[rule_indices] += loss_diff  

    def add_new_rule(self):
        """
        Dynamically manages rule count:
        - If below `max_rules`, add a new rule.
        - If at `max_rules`, replace a randomly chosen low-scoring rule.
        """
        if self.current_rule_count < self.max_rules:  
            # 🔹 Add new rule if under max_rules
            k = min(self.current_rule_count, 10)  # Ensure safe `topk()` selection
            top_rules = torch.topk(self.rule_scores, k, largest=True).indices  # Select top rules
            new_rule = self.rule_transform[top_rules].mean(dim=0, keepdim=True)  # Generate new rule

            self.rule_transform = nn.Parameter(torch.cat([self.rule_transform, new_rule], dim=0))
            self.rule_scores = torch.cat([self.rule_scores, torch.tensor([0.0], device=self.device)])
            self.current_rule_count += 1  # Track number of rules

            #print(f"🆕 Added a new rule! Total rules: {self.current_rule_count}")

        else:
            # 🔹 Pick a random low-scoring rule instead of always replacing the absolute worst
            k = min(self.current_rule_count, 10)  # Ensure we never request more rules than exist
            bottom_k_rules = torch.topk(self.rule_scores, k, largest=False).indices  # Select bottom `k` rules
            worst_rule_idx = bottom_k_rules[random.randint(0, k - 1)].item()  # Randomly select a low-scoring rule

            top_rules = torch.topk(self.rule_scores, k, largest=True).indices
            new_rule = self.rule_transform[top_rules].mean(dim=0, keepdim=True)  # Generate better rule
            
            with torch.no_grad():
                self.rule_transform[worst_rule_idx] = new_rule.squeeze(0)  # Replace the randomly chosen low-scoring rule
                self.rule_scores[worst_rule_idx] = 0  # Reset score

            #print(f"🔄 Replaced rule at index {worst_rule_idx}, now tracking {self.current_rule_count} rules")


class Rule_Transformer_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, seq_length, lambda_decay=0.01, memory_size=16, dropout=0.1, compression_factor=4, num_frequencies=100, max_rules=set_number_rules):
        super(Rule_Transformer_Model, self).__init__()
        self.embed_size = embedding_dim
        self.d_latent = embedding_dim // compression_factor  # Use a compressed latent space

        # 🔹 Use Dynamic Rule-Based Embeddings with Fourier as Base
        self.embedding = DynamicRuleEmbedding(vocab_size, embedding_dim, num_frequencies, max_rules)

        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len=seq_length)

        # Use Hierarchical MHLA
        self.encoder_layers = nn.ModuleList([
            HierarchicalMultiHeadLatentAttention(embedding_dim, num_heads, self.d_latent, lambda_decay, memory_size) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.seq_length = seq_length

    def forward(self, src, tgt):
        src_emb = self.embedding(src)  # 🔹 Generate Fourier + Rule-based embeddings dynamically
        tgt_emb = self.embedding(tgt)

        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        for layer in self.encoder_layers:
            src_emb = layer(src_emb)

        output = self.fc_out(src_emb)
        return output

def pad_mask(mask, target_size):
    """
    Pad the mask to match the required size.
    Args:
        mask (torch.Tensor): Original mask of shape (seq_len-1, seq_len-1)
        target_size (int): The target size to pad to (e.g., seq_len)
    Returns:
        torch.Tensor: Padded mask of shape (target_size, target_size)
    """
    pad_size = target_size - mask.size(0)
    if pad_size > 0:
        # Pad with -inf on the last row and column
        padding = torch.full((mask.size(0), pad_size), float('-inf'), device=mask.device)
        mask = torch.cat([mask, padding], dim=1)
        padding = torch.full((pad_size, target_size), float('-inf'), device=mask.device)
        mask = torch.cat([mask, padding], dim=0)
    return mask

def causal_mask(seq_len):
    """
    Creates a mask to prevent attending to future tokens.
    Args:
        seq_len (int): Length of the sequence
    Returns:
        mask (torch.Tensor): Shape [seq_len, seq_len], lower triangular matrix
    """
    return torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0)  # Add batch dimension

def padding_mask(input_ids, pad_token_id=pad_token_id):
    """
    Creates a mask for padded tokens in a batch.
    Args:
        input_ids (torch.Tensor): Shape [batch_size, seq_len]
        pad_token_id (int): Token ID representing padding (default 0)
    Returns:
        mask (torch.Tensor): Shape [batch_size, seq_len, seq_len]
    """
    mask = (input_ids != pad_token_id).unsqueeze(1).expand(-1, input_ids.size(1), -1)
    return mask


def create_memory_mask(memory, pad_token_id=pad_token_id):
    """
    Creates a memory mask for encoder-decoder attention.
    Masks padding tokens in the encoder output.
    Args:
        memory (torch.Tensor): Shape [batch_size, seq_len, d_model]
        pad_token_id (int): ID representing padding (usually 0)
    Returns:
        mask (torch.Tensor): Shape [batch_size, 1, seq_len]
    """
    return (memory != pad_token_id)  # Shape: [batch_size, 1, seq_len]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # Instead of erroring, simply truncate positional encodings to x.size(1)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
                 
                 
class Transformer_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, seq_length, dropout=0.1):
        super(Transformer_Model, self).__init__()
        self.embed_size = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len=seq_length)
        
        # Using batch_first=True so inputs are (batch, seq_len, embed_size)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.seq_length = seq_length
    
    def forward(self, src, tgt):
        # Ensure inputs require gradients for checkpointing

        logging.debug(f"src Shape: {src.shape}")
        logging.debug(f"tgt Shape: {tgt.shape}")

        # Input Embedding with requires_grad_
        src_emb = self.embedding(src).requires_grad_() * math.sqrt(self.embed_size)
        tgt_emb = self.embedding(tgt).requires_grad_() * math.sqrt(self.embed_size)
        # Input Embedding (Set requires_grad on the embedding, which is float)
        src_emb = self.embedding(src) * math.sqrt(self.embed_size)
        logging.debug(f"src_emb Shape: {src_emb.shape}")
        src_emb = src_emb.requires_grad_()  # Set requires_grad on float embeddings

        tgt_emb = self.embedding(tgt) * math.sqrt(self.embed_size)
        logging.debug(f"tgt_emb Shape: {tgt_emb.shape}")

        tgt_emb = tgt_emb.requires_grad_()  # Set requires_grad on float embeddings
        # Positional Encoding with requires_grad_
        src_emb = self.pos_encoder(src_emb).requires_grad_()
        tgt_emb = self.pos_encoder(tgt_emb).requires_grad_()
        logging.debug(f"src_emb with Positional Encoding Shape: {src_emb.shape}")
        logging.debug(f"tgt_emb with Positional Encoding Shape: {tgt_emb.shape}")

        if self.training:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.seq_length).to(src_emb.device)
        else:
            current_length = src_emb.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_length).to(src_emb.device)

        # Transformer Block with Checkpointing
        output = checkpoint.checkpoint(self.transformer, src_emb, tgt_emb, tgt_mask.to(src_emb.device), use_reentrant=False)
        logging.debug(f"output Shape: {output.shape}")

        # Output Layer
        output = self.fc_out(output)
        return output


def generate_text(model, input_ids, max_length, tokenizer, temperature=1.2, top_k=40, repetition_penalty=1.2):
    model.eval()
    generated = input_ids

    for _ in range(max_length):
        # Forward pass through the model
        outputs = model(generated, generated)
        next_token_logits = outputs[:, -1, :]  # Get logits for the last token
        
        # Apply repetition penalty while excluding special tokens
        for token in set(generated[0].tolist()):
            if token not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
                next_token_logits[0, token] /= repetition_penalty
        
        # Temperature scaling
        next_token_logits /= temperature
        
        # Top-k Sampling
        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        next_token = top_k_indices.gather(dim=-1, index=torch.multinomial(top_k_probs, num_samples=1))
        
        # Append the newly generated token to the sequence
        generated = torch.cat((generated, next_token), dim=1)

        # Stop if end-of-sequence token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated[0].tolist())

class ReasoningModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Reasoning Model GUI")

        # Transformer Parameters
        self.layers = []
        # Model Configuration Variables
        self.model_name = tk.StringVar(value="Reasoning Model")
        self.num_parameters = tk.IntVar(value=1024)
        self.num_heads = tk.IntVar(value=4)
        self.vocab_size = tk.IntVar(value=10000)
        self.hidden_size = tk.IntVar(value=8)
        self.num_layers = tk.IntVar(value=2)

        self.pad_token_id = 0  # Default value, adjust based on your tokenizer setup

        # Device Selection Variable
        self.device_option = tk.StringVar(value="GPU" if torch.cuda.is_available() else "CPU")

        # Dynamically calculate parameters based on other inputs
        self.vocab_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.hidden_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_layers.trace_add("write", lambda *args: self.update_num_parameters())

        # Set initial calculated value
        self.update_num_parameters()

        # Training Parameters
        self.dataset_path = ""
        self.vocab_path = ""
        self.tokenizer_path = ""
        self.batch_size = tk.IntVar(value=1)
        self.learning_rate = tk.DoubleVar(value=0.0005)
        self.epochs = tk.IntVar(value=1)

        # Training Variables
        self.loss_history = []
        self.accuracy_history = []
        self.current_epoch = 0
        self.stop_training = threading.Event()

        # Model and Data Variables
        self.model = None
        self.tokenizer = None
        self.dataset_path = None
        self.vocab_path = None
        self.tokenizer_path = None
        self.model_path = None
        self.train_data = None  # To store the dataset
        self.tokenized_data_path = None  # To store the tokenized data file path
        self.use_genetic_algo = "Genetic Algorithm"  # default to optim
        self.validation_loader = None
        self.num_saved_rules = 1
        
        # Device (CPU or GPU) - Initially set based on device_option
        self.device = torch.device(self.map_device(self.device_option.get()))

        # Select log file path
        self.select_log_file()

        # Setup logging
        logging.basicConfig(filename=self.log_file_path, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info(f"Using device: {self.device}")

        self.create_widgets()

    def map_device(self, selected_device):
        device_mapping = {
            "CPU": "cpu",
            "GPU": "cuda"
        }
        return device_mapping.get(selected_device, "cpu")

    def create_widgets(self):
        # Transformer Construction Frame
        transformer_frame = ttk.LabelFrame(self.root, text="Transformer Construction", padding=(10, 10))
        transformer_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(transformer_frame, text="Number of Parameters:").grid(row=0, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_parameters, state="readonly").grid(row=0, column=1)

        ttk.Label(transformer_frame, text="Number of Heads:").grid(row=1, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_heads).grid(row=1, column=1)
        
        ttk.Label(transformer_frame, text="Vocabulary Size:").grid(row=2, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.vocab_size).grid(row=2, column=1)

        ttk.Label(transformer_frame, text="Hidden Size:").grid(row=3, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.hidden_size).grid(row=3, column=1)

        ttk.Label(transformer_frame, text="Number of Layers:").grid(row=2, column=4, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_layers).grid(row=2, column=5)

        # Device Selection
        ttk.Label(transformer_frame, text="Select Device:").grid(row=4, column=0, sticky="w", pady=(10, 0))
        device_options = ["CPU"]
        if torch.cuda.is_available():
            device_options.append("GPU")
        device_combo = ttk.Combobox(transformer_frame, textvariable=self.device_option, values=device_options, state="readonly")
        device_combo.grid(row=4, column=1, sticky="w", pady=(10, 0))
        device_combo.bind("<<ComboboxSelected>>", self.on_device_change)

        # Attach parameter calculation to variable updates
        self.vocab_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.hidden_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_layers.trace_add("write", lambda *args: self.update_num_parameters())

        # For resuming training
        ttk.Button(transformer_frame, text="Select Model File", command=self.select_model_file).grid(row=3, column=2, pady=5)

        # Architecture selection
        self.architecture = tk.StringVar(value="Rule Transformer")
        ttk.Label(transformer_frame, text="Select Architecture:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(transformer_frame, textvariable=self.architecture, values=["Reasoning Model", "Rule Transformer"], state="readonly").grid(row=0, column=3)

        ttk.Button(transformer_frame, text="Add Layer", command=self.add_layer).grid(row=4, column=0, pady=5)
        ttk.Button(transformer_frame, text="Save Transformer and Model", command=self.save_transformer_and_model).grid(row=1, column=3, pady=5)
        ttk.Button(transformer_frame, text="Load Transformer", command=self.load_transformer).grid(row=1, column=2, pady=5)
        ttk.Button(transformer_frame, text="Initialize/Load Model", command=self.load_model).grid(row=2, column=3, pady=5)
        self.genetic_algo_var = tk.StringVar(value="GHR Optim")
        ttk.Label(transformer_frame, text="Algo:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(transformer_frame, textvariable=self.genetic_algo_var, values=["GHR Optim", "Genetic Algorithm", "Firefly", "NEAT", "NF Hybrid"], state="readonly").grid(row=0, column=4)

        # Data Selection Frame
        data_frame = ttk.LabelFrame(self.root, text="Data Selection", padding=(10, 10))
        data_frame.pack(fill="x", padx=10, pady=5)
        self.use_chunked_dataset = tk.BooleanVar(value=False)
        self.test_bool = tk.BooleanVar(value=False)
        
        ttk.Button(data_frame, text="Select Dataset Directory", command=self.select_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Load Dataset", command=self.load_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Load Tokenizer", command=self.load_tokenizer).pack(pady=5)

        # New buttons for tokenized data
        ttk.Button(data_frame, text="Select/Create Tokenized Data", command=self.select_or_create_tokenized_data).pack(pady=5)
        ttk.Button(data_frame, text="Tokenize Data", command=self.tokenize_data).pack(pady=5)

        # Training Configuration Frame
        train_frame = ttk.LabelFrame(self.root, text="Training Configuration", padding=(10, 10))
        train_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(train_frame, text="Batch Size:").grid(row=0, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.batch_size).grid(row=0, column=1)

        ttk.Label(train_frame, text="Learning Rate:").grid(row=1, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.learning_rate).grid(row=1, column=1)

        ttk.Label(train_frame, text="Epochs:").grid(row=2, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.epochs).grid(row=2, column=1)

        ttk.Button(train_frame, text="Start Training", command=self.start_training).grid(row=3, column=0, pady=5)
        ttk.Button(train_frame, text="Save Model", command=self.save_model).grid(row=3, column=1, pady=5)
        ttk.Button(train_frame, text="Stop Training", command=self.stop_training_command).grid(row=4, column=0, pady=5)
        self.training_mode = tk.StringVar(value="response")  # Default
        training_modes = ["imitation", "completion", "response"]
        ttk.Combobox(data_frame, textvariable=self.training_mode, values=training_modes, state="readonly").pack(pady=5)
        ttk.Button(train_frame, text="Run Validation", command=self.run_validation_button).grid(row=5, column=0, pady=5)
        ttk.Button(train_frame, text="Test Inference", command=self.test_inference).grid(row=5, column=1, pady=5)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(self.root, orient='horizontal', length=400, mode='determinate')
        self.progress_bar.pack(pady=10)
        self.status_label = ttk.Label(self.root, text="Status: Ready")
        self.status_label.pack(pady=5)

    def select_log_file(self):
        self.log_file_path = filedialog.asksaveasfilename(
            title="Select Log File Location",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        if self.log_file_path:
            print(f"Log file will be saved to: {self.log_file_path}")
        else:
            self.log_file_path = 'training_debug.log'  # Default log file
            print(f"No log file selected. Using default: {self.log_file_path}")

    def calculate_parameters(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        embedding_params = vocab_size * embedding_dim * 4  # Quaternion embeddings (4x normal embedding size)
        transformer_params = num_layers * (
            (embedding_dim * hidden_dim * 4) +  # Attention layers
            (hidden_dim * hidden_dim * 4) +  # Feed-forward layers
            (hidden_dim * 4 * embedding_dim * 4)  # Output layers
        )
        output_projection_params = embedding_dim * 4 * vocab_size  # Final projection
        return embedding_params + transformer_params + output_projection_params
        
    def test_inference(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded.")
            return

        # Set the model to evaluation mode
        self.model.eval()
        
        # Prompt the user for input text
        prompt = simpledialog.askstring("Test Inference", "Enter input text:")
        if prompt:
            try:
                if self.architecture.get() == "Rule Transformer":

                    self.model.eval()
                    with torch.no_grad():
                        # Tokenize the prompt and move to the correct device.
                        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
                        # Pad input_ids to the maximum sequence length (512)
                        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                        generated_text = input_ids
                        generated = []
                        logging.debug(f"Padded input_ids Shape: {input_ids.shape}")

                        # Choose a start token for the dummy target.
                        # Here we use tokenizer.eos_token_id if available; otherwise, fallback to tokenizer.pad_token_id.
                        bos_token = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.pad_token_id
                        eos_token = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id
                        eos_token  = torch.tensor([[eos_token]], device=device)

                        tgt_ids = torch.tensor([[bos_token]], device=device)
                        tgt_ids = torch.cat([tgt_ids, input_ids], dim=1)
                        logging.info(f"tgt_ids: {tgt_ids}")

                        # Keep track of the original input length
                        input_length = input_ids.size(1)

                        for _ in range(seq_len - input_ids.size(1)):
                            # Generate the target mask for the current target sequence length.
                            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.size(1)).to(device)
                            if tgt_mask is not None:
                                print("Target Mask:", tgt_mask)
                            # Forward pass through the model
                            outputs = self.model(input_ids, tgt_ids)
                            logging.debug(f"output shape: {outputs.shape}")

                            # Get logits for the last token and apply argmax to get the next token ID
                            next_token_logits = outputs[:, -1, :]  # Get the logits for the last position
                            repetition_penalty = 1.2  # Adjust for stronger penalty
                            # Apply repetition penalty while excluding special tokens like PAD (0)
                            for token in set(generated_text[0].tolist()):
                                if token not in [self.tokenizer.pad_token_id, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]:
                                    next_token_logits[0, token] /= repetition_penalty


                            top_p = 0.9  # Cumulative probability threshold
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0

                            filtered_logits = next_token_logits.clone()
                            filtered_logits[sorted_indices_to_remove] = float('-inf')

                            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                            logging.debug(f"next_token_logits: {next_token_logits}")
                            logging.debug(f"next_token_logits shape: {next_token_logits.shape}")
                            logging.debug(f"next_token_id shape: {next_token_id.shape}")
                            logging.debug(f"next_token_id: {next_token_id}")
                            # Append the new token to the target sequence.
                            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
                            logging.debug(f"tgt_ids: {tgt_ids}")
                            input_ids = input_ids[input_ids != self.tokenizer.pad_token_id].unsqueeze(0)
                            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                            logging.debug(f"input_ids: {input_ids}")
                            generated.append(self.tokenizer.decode(next_token_id[0].tolist()))
                            logging.debug(f"generated_text: {generated_text}")
                            # Stop generation if eos_token is generated
                            # Stop generation if eos_token is generated or max length is reached
                            if next_token_id.item() == self.tokenizer.eos_token_id or tgt_ids.size(1) >= 100:
                                break

                else:

                    self.model.eval()
                    with torch.no_grad():
                        # Tokenize the prompt and move to the correct device.
                        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
                        # Pad input_ids to the maximum sequence length (512)
                        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                        generated_text = input_ids
                        generated = []
                        logging.debug(f"Padded input_ids Shape: {input_ids.shape}")

                        # Choose a start token for the dummy target.
                        # Here we use tokenizer.eos_token_id if available; otherwise, fallback to tokenizer.pad_token_id.
                        bos_token = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.pad_token_id
                        eos_token = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id
                        eos_token  = torch.tensor([[eos_token]], device=device)

                        tgt_ids = torch.tensor([[bos_token]], device=device)
                        tgt_ids = torch.cat([tgt_ids, input_ids], dim=1)
                        logging.info(f"tgt_ids: {tgt_ids}")

                        # Keep track of the original input length
                        input_length = input_ids.size(1)

                        for _ in range(seq_len - input_ids.size(1)):
                            # Generate the target mask for the current target sequence length.
                            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.size(1)).to(device)
                            if tgt_mask is not None:
                                print("Target Mask:", tgt_mask)
                            # Forward pass through the model
                            outputs = self.model(input_ids, tgt_ids)
                            logging.debug(f"output shape: {outputs.shape}")

                            # Get logits for the last token and apply argmax to get the next token ID
                            next_token_logits = outputs[:, -1, :]  # Get the logits for the last position
                            repetition_penalty = 1.2  # Adjust for stronger penalty
                            # Apply repetition penalty while excluding special tokens like PAD (0)
                            for token in set(generated_text[0].tolist()):
                                if token not in [self.tokenizer.pad_token_id, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]:
                                    next_token_logits[0, token] /= repetition_penalty


                            top_p = 0.9  # Cumulative probability threshold
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0

                            filtered_logits = next_token_logits.clone()
                            filtered_logits[sorted_indices_to_remove] = float('-inf')

                            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                            logging.debug(f"next_token_logits: {next_token_logits}")
                            logging.debug(f"next_token_logits shape: {next_token_logits.shape}")
                            logging.debug(f"next_token_id shape: {next_token_id.shape}")
                            logging.debug(f"next_token_id: {next_token_id}")
                            # Append the new token to the target sequence.
                            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
                            logging.debug(f"tgt_ids: {tgt_ids}")
                            input_ids = input_ids[input_ids != self.tokenizer.pad_token_id].unsqueeze(0)
                            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                            logging.debug(f"input_ids: {input_ids}")
                            generated.append(self.tokenizer.decode(next_token_id[0].tolist()))
                            logging.debug(f"generated_text: {generated_text}")
                            print(tgt_ids)
                            # Stop generation if eos_token is generated
                            if next_token_id.item() == eos_token:
                                break

                    
                messagebox.showinfo("Inference Output", generated)
                logging.info(f"Inference Output: {generated}")

            except Exception as e:
                messagebox.showerror("Error", f"Inference failed: {str(e)}")
                logging.error(f"Inference failed: {str(e)}")

        # Optionally, return to train mode if further training is planned
        self.model.train()


    def run_validation(self, validation_loader, loss_fn):
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_input_ids, batch_labels, batch_labels_tot, seq_lengths in validation_loader:
                batch_input_ids = batch_input_ids.to(self.device)
                batch_labels = batch_labels.to(self.device)
                # Here, if your model requires three inputs, you may also send batch_labels_tot
                outputs, _, _ = self.model(batch_input_ids, batch_labels.reshape(-1), batch_labels_tot.reshape(-1))
                # Flatten outputs and targets for loss calculation
                logits = outputs.reshape(-1, outputs.size(-1))
                targets = batch_labels.reshape(-1)
                loss = loss_fn(logits, targets)
                total_val_loss += loss.item()
                num_batches += 1

        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")
        self.model.train()
        return avg_val_loss

    def update_num_parameters(self):
        vocab_size = self.vocab_size.get()
        embed_size = self.hidden_size.get()
        num_layers = self.num_layers.get()
        hidden_size = self.hidden_size.get()

        total_params = self.calculate_parameters(vocab_size, embed_size, num_layers, hidden_size)
        self.num_parameters.set(total_params)

    def on_device_change(self, event):
        selected_device = self.device_option.get()
        if selected_device == "GPU" and not torch.cuda.is_available():
            messagebox.showerror("Error", "GPU selected but CUDA is not available on this system.")
            self.device_option.set("CPU")
            selected_device = "CPU"
        device_str = self.map_device(selected_device)
        self.device = torch.device(device_str)
        logging.info(f"Device changed to: {self.device}")
        messagebox.showinfo("Device Selection", f"Computation device set to: {selected_device}")

    def resize_checkpoint_weights(self, state_dict, new_vocab_size, embed_size):
        """
        Resize checkpoint weights to match the current model's dimensions.
        """
        # This method may need to be updated depending on the model's state_dict keys
        return state_dict

    def select_model_file(self):
        self.model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", "*.pth;*.json"), ("All files", "*.*")]
        )
        if self.model_path:
            if self.model_path.endswith('.json'):
                # Load model configuration
                with open(self.model_path, 'r') as f:
                    config = json.load(f)
                # Update GUI parameters
                self.vocab_size.set(config.get("vocab_size", self.vocab_size.get()))
                self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                self.num_heads.set(config.get("num_heads", self.num_heads.get()))
                self.architecture.set(config.get("architecture", self.architecture.get()))
                messagebox.showinfo("Success", f"Model configuration loaded from: {self.model_path}")
            elif self.model_path.endswith('.pth'):
                # Load model weights
                config_directory = os.path.dirname(self.model_path)
                config_path = os.path.join(config_directory, 'model_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    # Update GUI parameters
                    self.vocab_size.set(config.get("vocab_size", self.vocab_size.get()))
                    self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                    self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                    self.num_heads.set(config.get("num_heads", self.num_heads.get()))
                    self.architecture.set(config.get("architecture", self.architecture.get()))
                    # Load the model
                    self.load_model()
                    # Load model state
                    state_dict = torch.load(self.model_path, map_location=self.device)
                    if self.architecture == "Rule Transformer":
                        # 🔹 Extract the number of rules from the saved model
                        self.num_saved_rules = state_dict["embedding.rule_transform"].shape[0]
                    self.model.load_state_dict(state_dict, strict=False)
                    messagebox.showinfo("Success", f"Model weights and configuration loaded from: {self.model_path}")
                else:
                    messagebox.showwarning("Warning", "Model configuration file not found. Please ensure the configuration is set correctly.")
            else:
                messagebox.showerror("Error", "Unsupported file format selected.")

    def save_transformer_and_model(self):
        if not self.model:
            messagebox.showerror("Error", "Model has not been initialized. Please initialize the model first.")
            return
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer has not been initialized. Please load a tokenizer first.")
            return

        transformer_data = {
            "vocab_size": self.vocab_size.get(),
            "embed_size": self.hidden_size.get(),
            "hidden_size": self.hidden_size.get(),
            "num_layers": self.num_layers.get(),
            "architecture": self.architecture.get(),
            "num_heads": self.num_heads.get(),
            "num_parameters": self.num_parameters.get(),
            "layers": self.layers
        }

        directory = filedialog.askdirectory(title="Select Save Directory")
        if directory:
            # Save configuration
            config_path = os.path.join(directory, "model_config.json")
            with open(config_path, "w") as file:
                json.dump(transformer_data, file, indent=4)

            # Save weights
            if self.architecture.get() == "Reasoning Model":
                model_file_name = 'reasoning_model.pth'
            elif self.architecture.get() == "Rule Transformer":
                model_file_name = 'rule_transformer.pth'
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            model_path = os.path.join(directory, model_file_name)
            torch.save(self.model.state_dict(), model_path)

            # Save tokenizer
            self.tokenizer.save_pretrained(directory)

            messagebox.showinfo("Success", "Model, tokenizer, and configuration saved successfully!")
            logging.info("Model, tokenizer, and configuration saved successfully.")

    def select_dataset(self):
        self.dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        if self.dataset_path:
            messagebox.showinfo("Success", f"Dataset directory selected: {self.dataset_path}")

    def select_vocab(self):
        self.vocab_path = filedialog.askopenfilename(
            title="Select Vocabulary File",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if self.vocab_path:
            messagebox.showinfo("Success", f"Vocabulary file selected: {self.vocab_path}")

    def select_tokenizer(self):
        self.tokenizer_path = filedialog.askopenfilename(
            title="Select Tokenizer File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if self.tokenizer_path:
            messagebox.showinfo("Success", f"Tokenizer file selected: {self.tokenizer_path}")

    def test_tokenizer(self):
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer not loaded.")
            return
        sample_text = simpledialog.askstring("Test Tokenizer", "Enter a sample text to tokenize:")
        if sample_text:
            tokens = self.tokenizer.tokenize(sample_text)
            token_ids = self.tokenizer.encode(sample_text)
            logging.info(f"Sample Text: {sample_text}")
            logging.info(f"Tokens: {tokens}")
            logging.info(f"Token IDs: {token_ids}")
            messagebox.showinfo("Tokenizer Test", f"Tokens: {tokens}\nToken IDs: {token_ids}")

    def save_dataset_as_text(self):
        if not hasattr(self, 'text_data') or not self.text_data:
            messagebox.showerror("Error", "No dataset loaded or processed to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Dataset as Text File",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    for line in self.text_data:
                        f.write(line + '\n')
                messagebox.showinfo("Success", f"Dataset saved to {save_path}")
                logging.info(f"Dataset saved to {save_path}")
            except Exception as e:
                logging.error(f"Failed to save dataset: {e}")
                messagebox.showerror("Error", f"Failed to save dataset: {e}")
                
    def create_tokenizer_from_vocab(self):
        try:
            # Ask the user to select the vocabulary file (our generated tokenizer.json)
            vocab_path = filedialog.askopenfilename(
                title="Select Vocabulary File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not vocab_path:
                messagebox.showerror("Error", "No vocabulary file selected.")
                return

            # Load the vocab from the JSON.
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)

            if "token_to_id" not in vocab_data:
                raise ValueError("The JSON file does not contain a 'token_to_id' key.")

            vocab = vocab_data["token_to_id"]

            # Check if merges exist in the file.
            if "merges" in vocab_data:
                merges = vocab_data["merges"]
                # Create a BPE model if merges are available.
                model = models.BPE(vocab=vocab, merges=merges, unk_token="<UNK>")
            else:
                # Fallback: use a WordLevel model if no merges are found.
                model = models.WordLevel(vocab=vocab, unk_token="<UNK>")

            # Create the tokenizer with the selected model.
            tokenizer = Tokenizer(model)

            # Set normalizer to NFKC for Unicode normalization.
            tokenizer.normalizer = normalizers.NFKC()
            # Use ByteLevel pre-tokenizer for byte-level tokenization.
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            # Use ByteLevel decoder for correct reconstruction of text.
            tokenizer.decoder = decoders.ByteLevel()

            # Wrap with PreTrainedTokenizerFast for HF integration.
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token="<UNK>",
                pad_token="<PAD>",
                bos_token="<BOS>",
                eos_token="<EOS>",
                model_max_length=seq_len  # Ensure seq_len is defined in your context.
            )

            # Ensure special tokens are added.
            self.tokenizer.add_special_tokens({
                'unk_token': '<UNK>',
                'pad_token': '<PAD>',
                'bos_token': '<BOS>',
                'eos_token': '<EOS>'
            })

            # Save the tokenizer.
            save_directory = filedialog.askdirectory(title="Select Directory to Save Tokenizer")
            if save_directory:
                os.makedirs(save_directory, exist_ok=True)
                self.tokenizer.save_pretrained(save_directory)
                self.tokenizer_path = os.path.join(save_directory, 'tokenizer.json')
                messagebox.showinfo("Success", f"Tokenizer saved to {self.tokenizer_path}")
                logging.info(f"Tokenizer saved to {self.tokenizer_path}")
            else:
                messagebox.showerror("Error", "No save directory selected for tokenizer.")
                return

            # Test the tokenizer.
            test_text = "Hello World!\nThis is a test.\tLet's remove line breaks and tabs."
            tokens = self.tokenizer.tokenize(test_text)
            logging.info(f"Test tokenization of '{test_text}': {tokens}")

            tokenizer_vocab = self.tokenizer.get_vocab()
            sorted_vocab = dict(sorted(tokenizer_vocab.items(), key=lambda item: item[1]))
            logging.info(f"Sorted Tokenizer Vocabulary: {sorted_vocab}")

            logging.info("Tokenizer created and saved successfully")
        except Exception as e:
            logging.error(f"Failed to create tokenizer: {str(e)}")
            messagebox.showerror("Error", f"Failed to create tokenizer: {str(e)}")
            raise


    def load_tokenizer(self):
        try:
            self.tokenizer_path = filedialog.askopenfilename(
                title="Select Tokenizer File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not self.tokenizer_path or not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError("Tokenizer file not selected or does not exist.")

            # Load the PreTrainedTokenizerFast from file.
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_path)
            logging.info(f"Tokenizer loaded from {self.tokenizer_path}")

            # If a special tokens map exists, load and add them.
            special_tokens_path = os.path.join(os.path.dirname(self.tokenizer_path), "special_tokens_map.json")
            if os.path.exists(special_tokens_path):
                with open(special_tokens_path, "r", encoding="utf-8") as file:
                    special_tokens = json.load(file)
                # Convert nested dicts to AddedToken if needed.
                for key, value in special_tokens.items():
                    if isinstance(value, dict):
                        special_tokens[key] = AddedToken(value["content"],
                                                        lstrip=value.get("lstrip", False),
                                                        rstrip=value.get("rstrip", False))
                    elif not isinstance(value, (str, AddedToken)):
                        raise ValueError(f"Invalid token format for key {key}: {value}")
                self.tokenizer.add_special_tokens(special_tokens)
                logging.info(f"Special tokens added: {special_tokens}")

            # Load tokenizer configuration if available.
            tokenizer_config_path = os.path.join(os.path.dirname(self.tokenizer_path), "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path, "r", encoding="utf-8") as file:
                    tokenizer_config = json.load(file)
                    self.tokenizer.init_kwargs.update(tokenizer_config)
                    if "model_max_length" in tokenizer_config:
                        self.tokenizer.model_max_length = tokenizer_config["model_max_length"]
                    logging.info(f"Tokenizer configuration loaded: {tokenizer_config}")

            # Ensure a reasonable model_max_length is set.
            if not hasattr(self.tokenizer, "model_max_length") or self.tokenizer.model_max_length > 1024 * 1024:
                self.tokenizer.model_max_length = seq_len  # Default value; ensure seq_len is defined
            logging.info(f"Model max length set to: {self.tokenizer.model_max_length}")

            # Log the vocabulary size.
            tokenizer_vocab_size = len(self.tokenizer)
            logging.info(f"Loaded tokenizer vocabulary size: {tokenizer_vocab_size}")
            self.vocab_size.set(tokenizer_vocab_size)

            # Ensure special tokens are correctly set.
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = "<PAD>"
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<PAD>")
                logging.warning("Pad token was not set. Defaulting to <PAD>.")
            if not self.tokenizer.unk_token:
                self.tokenizer.unk_token = "<UNK>"
                self.tokenizer.unk_token_id = self.tokenizer.convert_tokens_to_ids("<UNK>")
                logging.warning("UNK token was not set. Defaulting to <UNK>.")
            if not self.tokenizer.bos_token:
                self.tokenizer.bos_token = "<BOS>"
                self.tokenizer.bos_token_id = self.tokenizer.convert_tokens_to_ids("<BOS>")
                logging.warning("BOS token was not set. Defaulting to <BOS>.")
            if not self.tokenizer.eos_token:
                self.tokenizer.eos_token = "<EOS>"
                self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("<EOS>")
                logging.warning("EOS token was not set. Defaulting to <EOS>.")

            print("Special tokens map:", self.tokenizer.special_tokens_map)
            print("Pad token ID:", self.tokenizer.pad_token_id)
            print("Model max length:", self.tokenizer.model_max_length)

        except Exception as e:
            logging.error(f"Failed to load tokenizer: {str(e)}")
            messagebox.showerror("Error", f"Failed to load tokenizer: {str(e)}")


    def select_or_create_tokenized_data(self):
        use_chunked = self.use_chunked_dataset.get()
        answer = messagebox.askyesno("Select or Create Tokenized Data", "Do you want to use existing tokenized data?")
        
        if answer:
            if use_chunked:
                # User wants to use existing chunked tokenized data, select a directory
                self.tokenized_data_path = filedialog.askdirectory(
                    title="Select Tokenized Data Directory",
                    mustexist=True
                )
                if self.tokenized_data_path:
                    messagebox.showinfo("Success", f"Tokenized data directory selected: {self.tokenized_data_path}")
            else:
                # User wants to use existing single tokenized data file, select a file
                self.tokenized_data_path = filedialog.askopenfilename(
                    title="Select Tokenized Data File",
                    filetypes=[("JSON Lines files", "*.jsonl"), ("All files", "*.*")]
                )
                if self.tokenized_data_path:
                    # Attempt to load the file to validate its content
                    try:
                        with open(self.tokenized_data_path, 'r', encoding='utf-8') as f:
                            self.input_ids, self.labels, self.labels_tot = [], [], []
                            for line in f:
                                record = json.loads(line)
                                self.input_ids.append(record['input_ids'])
                                self.labels.append(record['labels'])
                                self.labels_tot.append(record['labels_tot'])
                        messagebox.showinfo("Success", f"Tokenized data file loaded: {self.tokenized_data_path}")
                        logging.info(f"Tokenized data file loaded successfully with {len(self.input_ids)} entries.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to load tokenized data file: {str(e)}")
        else:
            if use_chunked:
                # User wants to create new chunked tokenized data, select a directory to save
                self.tokenized_data_path = filedialog.askdirectory(
                    title="Select Directory to Save Tokenized Data"
                )
                if self.tokenized_data_path:
                    os.makedirs(self.tokenized_data_path, exist_ok=True)  # Ensure directory is created
                    messagebox.showinfo("Success", f"Tokenized data will be saved to directory: {self.tokenized_data_path}")
            else:
                # User wants to create new single tokenized data file, select a file path
                self.tokenized_data_path = filedialog.asksaveasfilename(
                    title="Save Tokenized Data As",
                    defaultextension=".jsonl",
                    filetypes=[("JSON Lines files", "*.jsonl"), ("All files", "*.*")]
                )
                if self.tokenized_data_path:
                    messagebox.showinfo("Success", f"Tokenized data will be saved to file: {self.tokenized_data_path}")

            
    def tokenize_data(self):
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer not loaded.")
            return
        if not hasattr(self, 'query_target_pairs') or not self.query_target_pairs:
            messagebox.showerror("Error", "No query-target pairs loaded. Please load the dataset first.")
            return
        if not self.tokenized_data_path:
            messagebox.showerror("Error", "Tokenized data path not set. Please select or create tokenized data.")
            return

        # Select training mode
        training_mode = self.training_mode.get()  # "imitation", "completion", "response"
        self.input_ids = []  # Initialize for unchunked dataset
        self.labels = []  # Initialize for unchunked dataset
        self.labels_tot = [] #for train of thought labels
        
        try:
            use_chunked = self.use_chunked_dataset.get()
            if use_chunked:
                #create path if none
                os.makedirs(self.tokenized_data_path, exist_ok=True)
                chunk_size = 32
                num_chunks = (len(self.query_target_pairs) + chunk_size - 1) // chunk_size

                for chunk_idx in range(num_chunks):
                    chunk_pairs = self.query_target_pairs[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]
                    chunk_file_path = os.path.join(self.tokenized_data_path, f'chunk_{chunk_idx}.jsonl')

                    with open(chunk_file_path, 'w', encoding='utf-8') as f:
                        for query, target in chunk_pairs:
                            input_ids, labels, labels_tot = self._generate_training_pairs(query, target, tot_target, training_mode)
                            if input_ids and labels and labels_tot:
                                record = {'input_ids': input_ids, 'labels': labels, 'labels_tot': labels_tot}
                                f.write(json.dumps(record) + '\n')
                logging.info(f"Chunk {chunk_idx} tokenized and saved to {chunk_file_path}")

                messagebox.showinfo("Success", f"Data tokenized into {num_chunks} chunks and saved successfully to {self.tokenized_data_path}.")
                logging.info(f"Data tokenized into {num_chunks} chunks and saved successfully to {self.tokenized_data_path}.")
            else:
                with open(self.tokenized_data_path, 'w', encoding='utf-8') as f:
                    for query, target, tot_target in self.query_target_pairs:
                        input_ids, labels, labels_tot = self._generate_training_pairs(query, target, tot_target, training_mode)

                        if input_ids and labels and labels_tot:
                            self.input_ids.append(input_ids)  # Store for training
                            self.labels.append(labels)  # Store for training
                            self.labels_tot.append(labels_tot)  # Store for training
                            record = {'input_ids': input_ids, 'labels': labels, 'labels_tot': labels_tot}


                            f.write(json.dumps(record) + '\n')
                logging.info(f"Input IDs: {len(self.input_ids)} sequences loaded.")
                logging.info(f"Labels: {len(self.labels)} sequences loaded.")
                logging.info(f"Labels_tot: {len(self.labels_tot)} sequences loaded.")
                logging.info(f"Input_ids sample: {self.input_ids[0][:10]}...")  # Shows only first 10 tokens
                logging.info(f"Labels sample: {self.labels[0][:10]}...")  # Shows only first 10 tokens
                logging.info(f"Labels_tot sample: {self.labels_tot[0][:10]}...")  # First 10 tokens for ToT

                messagebox.showinfo("Success", f"Data tokenized and saved successfully to {self.tokenized_data_path}.")
                logging.info(f"Data tokenized and saved successfully to {self.tokenized_data_path}.")
        except Exception as e:
            logging.error(f"Tokenization failed: {str(e)}")
            messagebox.showerror("Error", f"Tokenization failed: {str(e)}")

    def _generate_training_pairs(self, query, target, tot_target, training_mode):
        # Debugging logs
        logging.debug(f"Generating Training Pairs - Query: {query}")
        logging.debug(f"Generating Training Pairs - Target: {target}")
        logging.debug(f"Generating Training Pairs - ToT: {tot_target}")

        # Ensure inputs are valid strings before tokenization
        query_ids = self.tokenizer.encode(str(query) if query else "", truncation=True, max_length=seq_len)
        target_ids = self.tokenizer.encode(str(target) if target else "", truncation=True, max_length=seq_len)
        tot_target_ids = self.tokenizer.encode(str(tot_target) if tot_target else "", truncation=True, max_length=seq_len)

        if training_mode == "imitation":
            input_ids = [self.tokenizer.bos_token_id] + query_ids + [self.tokenizer.eos_token_id] 
            labels = [self.tokenizer.bos_token_id] + query_ids + [self.tokenizer.eos_token_id] 
        elif training_mode == "completion":
            partial_length = len(query_ids) // 2
            partial_input = query_ids[:partial_length]
            input_ids = [self.tokenizer.bos_token_id] + partial_input + [self.tokenizer.eos_token_id]
            labels = [self.tokenizer.bos_token_id] + query_ids + [self.tokenizer.eos_token_id]  
        else:  # response mode
            input_ids = [self.tokenizer.bos_token_id] + query_ids + [self.tokenizer.eos_token_id]
            labels = [self.tokenizer.bos_token_id] + target_ids + [self.tokenizer.eos_token_id]
            labels_tot = [self.tokenizer.bos_token_id] + tot_target_ids + [self.tokenizer.eos_token_id]

        return input_ids, labels, labels_tot


    def add_layer(self):
        layer_type = simpledialog.askstring("Layer Type", "Enter layer type (e.g., attention, feed_forward)")
        if layer_type:
            layer_config = {
                "type": layer_type,
                "parameters": {}  # Placeholder for future parameter configuration
            }
            self.layers.append(layer_config)
            messagebox.showinfo("Layer Added", f"Layer of type '{layer_type}' added.")

    def save_transformer(self):
        transformer_data = {
            "num_parameters": self.num_parameters.get(),
            "layers": self.layers
        }

        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as file:
                json.dump(transformer_data, file, indent=4)
            messagebox.showinfo("Save", "Transformer saved successfully!")
            logging.info(f"Number of layers in the model: {len(self.model.transformer_encoder.layers)}")

    def load_transformer(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as file:
                transformer_data = json.load(file)
            self.num_parameters.set(transformer_data["num_parameters"])
            self.num_heads.set(transformer_data["num_heads"])
            self.layers = transformer_data["layers"]
            messagebox.showinfo("Success", "Transformer loaded successfully")

    def load_model(self):
        try:
            if not self.tokenizer:
                vocab_size = self.vocab_size.get()
            else:
                vocab_size = len(self.tokenizer)

            # Log and validate vocab size
            logging.info(f"Tokenizer vocabulary size: {vocab_size}")
            self.vocab_size.set(vocab_size)

            # Initialize the model based on architecture
            if self.architecture.get() == "Reasoning Model":
                self.model = Transformer_Model(
                    vocab_size=vocab_size,
                    embedding_dim=self.hidden_size.get(),
                    hidden_dim=self.hidden_size.get(),
                    num_layers=self.num_layers.get(),
                    num_heads=self.num_heads.get(),

                    seq_length=seq_len
                )
            elif self.architecture.get() == "Rule Transformer":
                self.model = Rule_Transformer_Model(
                    vocab_size=vocab_size,
                    embedding_dim=self.hidden_size.get(),
                    num_layers=self.num_layers.get(),
                    num_heads=self.num_heads.get(),
                    seq_length=seq_len,
                    max_rules=self.num_saved_rules  
                )
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            # Move the entire model to the selected device
            self.model.to(device)
            logging.info(f"Model moved to device: {self.device}")

            # Load checkpoint if a model file is selected
            if self.model_path and self.model_path.endswith('.pth'):
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                logging.info("Model weights loaded and resized successfully.")

            logging.info(f"Model initialized on device: {self.device}")
            messagebox.showinfo("Success", "Model initialized successfully.")

        except Exception as e:
            logging.error(f"Failed to initialize model: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize model: {str(e)}")

    def load_trained_model(self):
        trained_model_path = "trained_model.pth"
        if os.path.exists(trained_model_path):
            self.model_path=trained_model_path
            self.load_model()  # Load updated model
            # Load model state
            state_dict = torch.load(self.model_path, map_location=self.device)
            if self.architecture == "Rule Transformer":
                # 🔹 Extract the number of rules from the saved model
                self.num_saved_rules = state_dict["embedding.rule_transform"].shape[0]
            self.model.load_state_dict(state_dict, strict=False)
            print("✅ Loaded updated model into GUI.")

    def calculate_learning_rate(self, total_params):
        # Calculate learning rate based on total parameters using the derived formula
        # LR = 17.38 * (Model Size)^-0.424
        lr = 17.38 * (total_params ** -0.424)
        return lr

    def start_training(self):
        # Start training in a separate thread to keep the GUI responsive
        self.stop_training.clear()
        model_path = "saved_model.pth"
        torch.save(self.model.state_dict(), model_path)
        # ✅ Save model parameters dynamically
        model_config = {
            "seq_length": seq_len,
            "vocab_size": self.vocab_size.get(),
            "embed_size": self.hidden_size.get(),
            "hidden_size": self.hidden_size.get(),
            "num_layers": self.num_layers.get(),
            "architecture": self.architecture.get(),
            "num_parameters": self.num_parameters.get(),
            "num_heads": self.num_heads.get(),
            "layers": self.layers,
            "genetic_algo_var": self.genetic_algo_var.get(),
            "use_chunked_dataset": self.use_chunked_dataset.get(),
            "tokenized_data_path": self.tokenized_data_path,
            "tokenizer_path": self.tokenizer_path,
            "batch_size": self.batch_size.get(),
            "epochs": self.epochs.get(),
            "learning_rate": self.learning_rate.get(),
            "num_saved_rules": self.num_saved_rules,
            "log_file_path": self.log_file_path            
        }

        config_path = "model_config.json"
        with open(config_path, "w") as f:
            json.dump(model_config, f)
        # ✅ Delete any existing model in memory (if applicable)
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        training_thread = subprocess.Popen(["torchrun", "--nnodes=1", "--nproc_per_node=2","--master_port=29500", "training_loop.py", model_path, config_path])
        # Wait for training to complete before proceeding
        training_thread.wait()  # ✅ Ensures training fully finishes before loading the updated model

        # Load the trained model back into the GUI
        self.load_trained_model()

    def update_progress(self, progress_value):
        self.progress_bar['value'] = progress_value

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def update_progress(self):
        """Reads the progress file and updates the progress bar in the GUI."""
        progress_file = "training_progress.json"
        epoch = 0
        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r") as f:
                    data = json.load(f)
                    progress_value = data.get("progress", 0)
                    
                    # ✅ Update progress bar
                    self.root.after(0, self.progress_bar.set, progress_value)
            except json.JSONDecodeError:
                pass  # Ignore incomplete writes

        # ✅ Keep checking progress every second
        self.root.after(1000, self.update_progress)
        if progress_value == 100:
            epoch = epoch+1
            self.update_status(f"Epoch {epoch} completed.")

        
    def validate_training_parameters(self):
        # Validate batch size
        try:
            batch_size = int(self.batch_size.get())
            if batch_size <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logging.error(f"Invalid batch size: {self.batch_size.get()}")
            messagebox.showerror("Error", "Batch size must be a positive integer.")
            return False

        # Validate epochs
        try:
            epochs = int(self.epochs.get())
            if epochs <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logging.error(f"Invalid epochs value: {self.epochs.get()}")
            messagebox.showerror("Error", "Epochs must be a positive integer.")
            return False

        if not self.tokenized_data_path or not os.path.exists(self.tokenized_data_path):
            logging.error("Tokenized data path is invalid or does not exist.")
            messagebox.showerror("Error", "Tokenized data is not selected or does not exist.")
            return False

        if not hasattr(self.tokenizer, 'pad_token_id') or self.tokenizer.pad_token_id is None:
            logging.error("Tokenizer pad_token_id is not set.")
            messagebox.showerror("Error", "Tokenizer is missing pad_token_id.")
            return False

        return True

    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.pi * progress)))
            
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def save_model(self):
        if not self.model:
            messagebox.showerror("Error", "Model has not been initialized. Cannot save.")
            logging.error("Attempted to save model but model was not initialized.")
            return
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer has not been initialized. Cannot save.")
            logging.error("Attempted to save model but tokenizer was not initialized.")
            return

        save_directory = filedialog.askdirectory(title="Select Save Directory")
        if save_directory:
            config = {
            "vocab_size": self.vocab_size.get(),
            "embed_size": self.hidden_size.get(),
            "hidden_size": self.hidden_size.get(),
            "num_layers": self.num_layers.get(),
            "architecture": self.architecture.get(),
            "num_parameters": self.num_parameters.get(),
            "num_heads": self.num_heads.get(),
            "layers": self.layers
        }
            config_path = os.path.join(save_directory, 'model_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)

            # Save the model state dictionary
            if self.architecture.get() == "Reasoning Model":
                model_file_name = 'reasoning_model.pth'
            elif self.architecture.get() == "Rule Transformer":
                model_file_name = 'rule_transformer.pth'
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            model_path = os.path.join(save_directory, model_file_name)
            torch.save(self.model.state_dict(), model_path)

            # Save the tokenizer
            self.tokenizer.save_pretrained(save_directory)

            messagebox.showinfo("Success", "Model, tokenizer, and config saved successfully.")
            logging.info("Model, tokenizer, and config saved successfully.")

    def stop_training(self):
        """Stops the training subprocess."""
        self.stop_training.set()
        if hasattr(self, "training_process"):
            self.training_process.terminate()
            self.training_process.wait()
            print("Training stopped.")
        messagebox.showinfo("Stop Training", "Training stopped.")
        logging.info("Training stopped by user.")

    def stop_training_command(self):
        """Stops the training subprocess."""
        self.stop_training.set()
        if hasattr(self, "training_process"):
            self.training_process.terminate()
            self.training_process.wait()
            print("Training stopped.")
        messagebox.showinfo("Stop Training", "Training stopped.")
        logging.info("Training stopped by user.")
        
    def expand_transformer(self):
        # Placeholder method; not used in current implementation
        pass

    
    def load_dataset(self):
        if self.use_chunked_dataset.get():
            # Load data from chunked files
            self.tokenized_data_path = filedialog.askdirectory(
                title="Select Tokenized Data Directory"
            )
            if not self.tokenized_data_path:
                messagebox.showerror("Error", "No tokenized data directory selected.")
                return

            # Check if directory contains chunked data files
            chunk_files = [f for f in os.listdir(self.tokenized_data_path) if f.startswith('chunk_') and f.endswith('.jsonl')]
            if not chunk_files:
                messagebox.showerror("Error", "No chunked data files found in the selected directory.")
                return

            self.chunked_files = [os.path.join(self.tokenized_data_path, f) for f in chunk_files]
            messagebox.showinfo("Success", f"Loaded chunked dataset with {len(self.chunked_files)} files.")
            logging.info(f"Loaded chunked dataset with {len(self.chunked_files)} files.")
        else:
            # Load standard dataset
            if not self.dataset_path:
                messagebox.showerror("Error", "No dataset directory selected.")
                return

            dataset_files = os.listdir(self.dataset_path)
            self.query_target_pairs = []

            for file in dataset_files:
                file_path = os.path.join(self.dataset_path, file)
                if file.endswith('.json') or file.endswith('.jsonl'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if file.endswith('.jsonl'):
                                for line in f:
                                    conversation = json.loads(line.strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target, tot_target = self.query_target_pairs[i]

                            elif file.endswith('.parquet'):
                                try:
                                    df = pd.read_parquet(file_path)
                                    
                                    if "conversations" not in df.columns:
                                        messagebox.showerror("Error", f"Parquet file '{file}' does not contain a 'conversation' column.")
                                        continue

                                    conversations = df["conversations"].dropna().tolist()  # Drop NaN values
                                    
                                    # Parse the JSON strings in the column (if necessary)
                                    parsed_conversations = []
                                    for conv in conversations:
                                        try:
                                            parsed_conversations.append(json.loads(conv))
                                        except json.JSONDecodeError:
                                            messagebox.showwarning("Warning", f"Skipping non-JSON conversation entry in '{file}'.")
                                    
                                    self.query_target_pairs.extend(self.extract_query_target_pairs(parsed_conversations))

                                except Exception as e:
                                    messagebox.showerror("Error", f"Failed to read Parquet file '{file}': {str(e)}")

                            else:
                                data = json.load(f)
                                self.query_target_pairs.extend(self.extract_query_target_pairs(data)) 
                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target, tot_target = self.query_target_pairs[i]
                               
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read JSON file '{file}': {str(e)}")
                else:
                    messagebox.showwarning("Warning", f"Unsupported file format: '{file}'")

            if not self.query_target_pairs:
                messagebox.showerror("Error", "No valid query/target pairs found in the dataset.")
                return

            # Store text data for saving as a text file
            self.text_data = []
            for query, target, tot_target in self.query_target_pairs:
                self.text_data.append(f"User: {query}\nAssistant: {target}\nReasoning: {tot_target}")

            messagebox.showinfo("Success", f"Loaded dataset with {len(self.query_target_pairs)} query/target pairs.")
            logging.info(f"Loaded dataset with {len(self.query_target_pairs)} query/target pairs.")

    def extract_query_target_pairs(self, data):
        query_target_pairs = []

        for conversation in data:
            if conversation.get("messages"):
                messages = conversation.get("messages", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant":
                        query = messages[i].get("content") or messages[i].get("value", "")
                        target = messages[i + 1].get("content") or messages[i + 1].get("value", "")
                        tot_target = self.extract_thinking(messages[i + 1])  # Extract reasoning
                        query_target_pairs.append((query.strip(), target.strip(), tot_target))

                    elif messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        tot_target = self.extract_thinking(messages[i + 1])
                        query_target_pairs.append((query.strip(), target.strip(), tot_target))

            elif conversation.get("conversations"):
                messages = conversation.get("conversations", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        tot_target = self.extract_thinking(messages[i + 1])
                        query_target_pairs.append((query.strip(), target.strip(), tot_target))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        tot_target = self.extract_thinking(messages[i + 1])
                        query_target_pairs.append((query.strip(), target.strip(), tot_target))
            elif conversation.get("text"):
                messages = conversation.get("text", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        tot_target = self.extract_thinking(messages[i + 1])
                        query_target_pairs.append((query.strip(), target.strip(), tot_target))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        tot_target = self.extract_thinking(messages[i + 1])
                        query_target_pairs.append((query.strip(), target.strip(), tot_target))
            else:
                user_messages = conversation.get("user", [])
                assistant_messages = conversation.get("assistant", [])
                for i in range(min(len(user_messages), len(assistant_messages))):
                    query = user_messages[i].replace('\n', ' ').strip()
                    target = assistant_messages[i].replace('\n', ' ').strip()
                    tot_target = self.extract_thinking(assistant_messages[i])
                    query_target_pairs.append((query, target, tot_target))

        return query_target_pairs

    def extract_query_target_pairs_json(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                f.seek(0)  # Reset file pointer
                
                if first_line.startswith('['):
                    data = json.load(f)  # JSON array format
                else:
                    data = [json.loads(line.strip()) for line in f]  # JSONL format

            return self.extract_query_target_pairs(data)

        except Exception as e:
            logging.error(f"Failed to load JSON file: {e}")
            return []

    def extract_thinking(self, assistant_response):
        """
        Extracts reasoning or 'train of thought' from the assistant's response if present.
        """
        tot_target = None
        if isinstance(assistant_response, dict):
            response_text = assistant_response.get("value", "") or assistant_response.get("content", "")
        else:
            response_text = assistant_response  # If it's already a string

        if "<think>" in response_text and "</think>" in response_text:
            tot_target = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
        elif "<thinking>" in response_text and "</thinking>" in response_text:
            tot_target = re.search(r"<thinking>(.*?)</thinking>", response_text, re.DOTALL)
        elif "<|begin_of_thought|>" in response_text and "<|end_of_thought|>" in response_text:
            tot_target = re.search(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", response_text, re.DOTALL)
        elif "longCOT" in assistant_response:
            tot_target = assistant_response["longCOT"]
        elif assistant_response.get("role") == "reasoning":
            tot_target = assistant_response.get("content")
        elif "thinking" in assistant_response:
            tot_target = assistant_response["thinking"]

        return tot_target.group(1).strip() if tot_target else ""  # 🔹 Ensure no `None` values


    def extract_query_target_pairs_parquet(self, file_path):
        try:
            df = pd.read_parquet(file_path)
            query_target_pairs = []

            for _, row in df.iterrows():
                user_query = row.get("question") or row.get("input")
                assistant_response = row.get("answer") or row.get("response")
                tot_target = self.extract_thinking(row)

                if user_query and assistant_response:
                    query_target_pairs.append((user_query.strip(), assistant_response.strip(), tot_target.strip() if tot_target else None))

            return query_target_pairs

        except Exception as e:
            logging.error(f"Failed to load Parquet file: {e}")
            return []
        
    def create_validation_loader(self):
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else pad_token_id
        # If using a chunked dataset for validation
        if self.use_chunked_dataset.get():
            if hasattr(self, 'validation_tokenized_data_path') and self.validation_tokenized_data_path:
                dataset = ChunkedDataset(
                    tokenized_data_path=self.validation_tokenized_data_path,
                    tokenizer=self.tokenizer,
                    max_length=seq_len
                )
            else:
                messagebox.showerror("Error", "No chunked validation data directory selected.")
                return None
        else:
            # Using an unchunked (in-memory) validation dataset
            if not (hasattr(self, 'validation_input_ids') and hasattr(self, 'validation_labels') and hasattr(self, 'validation_labels_tot')):
                messagebox.showerror("Error", "Validation data not loaded. Please load validation data first.")
                return None

            # Convert the lists of token IDs to tensors with proper padding
            input_ids = torch.stack([
                torch.tensor(tokens + [pad_token_id] * (seq_len - len(tokens)), dtype=torch.long, device=self.device)[:seq_len]
                for tokens in self.validation_input_ids
            ])
            labels = torch.stack([
                torch.tensor(tokens + [pad_token_id] * (seq_len - len(tokens)), dtype=torch.long, device=self.device)[:seq_len]
                for tokens in self.validation_labels
            ])
            labels_tot = torch.stack([
                torch.tensor(tokens + [pad_token_id] * (seq_len - len(tokens)), dtype=torch.long, device=self.device)[:seq_len]
                for tokens in self.validation_labels_tot
            ])
            seq_lengths = torch.tensor(
                [min(len(tokens), seq_len) for tokens in self.validation_input_ids],
                dtype=torch.long, device=self.device
            )
            dataset = torch.utils.data.TensorDataset(input_ids, labels, labels_tot, seq_lengths)

        loader = DataLoader(dataset, batch_size=self.batch_size.get(), shuffle=False, collate_fn=collate_fn)
        return loader

        
    def select_validation_dataset(self):
        # Ask the user to select the validation dataset directory
        self.validation_dataset_path = filedialog.askdirectory(title="Select Validation Dataset Directory")
        if not self.validation_dataset_path:
            messagebox.showerror("Error", "No validation dataset directory selected.")
            return

        # Load validation query/target pairs similar to your load_dataset implementation
        dataset_files = os.listdir(self.validation_dataset_path)
        self.validation_query_target_pairs = []
        
        for file in dataset_files:
            file_path = os.path.join(self.validation_dataset_path, file)
            if file.endswith('.json') or file.endswith('.jsonl'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if file.endswith('.jsonl'):
                            for line in f:
                                conversation = json.loads(line.strip())
                                self.validation_query_target_pairs.extend(self.extract_query_target_pairs([conversation]))
                        else:
                            data = json.load(f)
                            self.validation_query_target_pairs.extend(self.extract_query_target_pairs(data))
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load {file}: {str(e)}")
        
        if not self.validation_query_target_pairs:
            messagebox.showerror("Error", "No valid query/target pairs found in the validation dataset.")
            return

        # Tokenize validation data similar to your _generate_training_pairs method
        self.validation_input_ids = []
        self.validation_labels = []
        self.validation_labels_tot = []
        
        # Use the same training mode as set in the GUI (imitation, completion, or response)
        training_mode = self.training_mode.get()
        
        for query, target, tot_target in self.validation_query_target_pairs:
            input_ids, labels, labels_tot = self._generate_training_pairs(query, target, tot_target, training_mode)
            self.validation_input_ids.append(input_ids)
            self.validation_labels.append(labels)
            self.validation_labels_tot.append(labels_tot)
        
        # Create tensors with proper padding
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else pad_token_id
        input_ids_tensor = torch.stack([
            torch.tensor(ids + [pad_token_id] * (seq_len - len(ids)), dtype=torch.long, device=self.device)[:seq_len]
            for ids in self.validation_input_ids
        ])
        labels_tensor = torch.stack([
            torch.tensor(ids + [pad_token_id] * (seq_len - len(ids)), dtype=torch.long, device=self.device)[:seq_len]
            for ids in self.validation_labels
        ])
        labels_tot_tensor = torch.stack([
            torch.tensor(ids + [pad_token_id] * (seq_len - len(ids)), dtype=torch.long, device=self.device)[:seq_len]
            for ids in self.validation_labels_tot
        ])
        seq_lengths_tensor = torch.tensor(
            [min(len(ids), seq_len) for ids in self.validation_input_ids],
            dtype=torch.long, device=self.device
        )
        
        # Create a TensorDataset and DataLoader for validation
        validation_dataset = torch.utils.data.TensorDataset(input_ids_tensor, labels_tensor, labels_tot_tensor, seq_lengths_tensor)
        self.validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size.get(), shuffle=False, collate_fn=collate_fn)
        
        messagebox.showinfo("Success", f"Validation dataset loaded with {len(validation_dataset)} samples.")


    def run_validation(self, validation_loader, loss_fn):
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch_input_ids, batch_labels, batch_labels_tot, seq_lengths in validation_loader:
                batch_input_ids = batch_input_ids.to(self.device)
                batch_labels = batch_labels.to(self.device)
                # Adjust the forward call as needed if your model requires three inputs.
                outputs, _, _ = self.model(batch_input_ids, batch_labels.reshape(-1), batch_labels_tot.reshape(-1))
                # Flatten outputs and targets for loss calculation
                logits = outputs.reshape(-1, outputs.size(-1))
                targets = batch_labels.reshape(-1)
                loss = loss_fn(logits, targets)
                total_val_loss += loss.item()
                num_batches += 1
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")
        self.model.train()
        return avg_val_loss

    def run_validation_button(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded.")
            return
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else pad_token_id
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        validation_loader = self.create_validation_loader()
        if validation_loader is None:
            return
        val_loss = self.run_validation(validation_loader, loss_fn)
        messagebox.showinfo("Validation", f"Validation Loss: {val_loss:.4f}")


# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()


    app = ReasoningModelGUI(root)
    root.mainloop()
