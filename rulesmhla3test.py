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
import gc
import time
import pdfplumber


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return full_text

def safe_tensor_allocation(tensor):
    try:
        return tensor.to("cuda", non_blocking=True)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("⚠️ GPU memory full! Offloading to CPU RAM.")
            return tensor.to("cpu")
        else:
            raise e
        
def setup_ddp():
    """Setup distributed training only if multiple GPUs are available."""
    if torch.cuda.device_count() > 2:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")

        world_size = torch.cuda.device_count()
        rank = int(os.environ.get("RANK", "0"))  # Rank should be set by torchrun

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        print(f"✅ DDP Initialized: {world_size} GPUs, Rank: {rank}")
    else:
        print("⚠️ Running on a single GPU (DDP disabled).")

setup_ddp()

## set threads
#torch.set_num_threads(32)
#torch.set_num_interop_threads(32)

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

seq_len = 120
pad_token_id = 0
set_number_rules = 1000

scaler = torch.amp.GradScaler()  # Initialize GradScaler

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
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-4)

    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    del A, B, b, c
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
                    #assert g is not None
                    if g is None:
                        continue  # skip this param

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
                    torch.cuda.empty_cache()
                else:
                    # Apply updates directly if single-GPU
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
                    torch.cuda.empty_cache()

            if self.ddp_enabled:
                update_prev()
                
def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    torch.save(checkpoint, path)
    
def save_checkpoint_rules(model, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    
    # 🔹 Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # 🔹 Load optimizer state if provided
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=False)


    return checkpoint['epoch'], checkpoint['phase']


def init_tokenizer(tokenizer_path):
    global tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    logging.info(f"Tokenizer pad_token set to: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")


def tokenize_chunk(chunk):
    # Tokenizer is now the global variable initialized in each process
    encoded = tokenizer(chunk, return_attention_mask=False, truncation=True, max_length=seq_len)
    return encoded['input_ids']

device_cpu = 'cpu'
def collate_fn(batch):
    """
    Collate function for standard seq2seq data. Each sample is a tuple (input_ids, target).
    Both sequences are padded/truncated to a fixed length.
    """
    input_ids = []
    labels = []
    seq_lengths = []
    if len(batch[0]) == 2:
        for query, target in batch:
            input_ids.append(query)
            labels.append(target)
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        return input_ids, labels
    if len(batch[0]) == 3:
        # Dataset returns: input_ids, labels, seq_lengths
        input_ids, labels, seq_lengths = zip(*batch)
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        return input_ids, labels, seq_lengths

class ChunkedDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data_path, tokenizer, max_length=seq_len):
        self.tokenized_data_path = tokenized_data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # ✅ Get list of chunk files
        self.chunk_files = sorted([
            os.path.join(self.tokenized_data_path, f) 
            for f in os.listdir(self.tokenized_data_path) 
            if f.startswith('chunk_') and f.endswith('.jsonl')
        ])

        # ✅ Build index mapping (ensures proper indexing across chunks)
        self.index_mapping = []
        self.chunk_sizes = {}  # Store chunk sizes for quick lookup

        for chunk_idx, chunk_file in enumerate(self.chunk_files):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
            self.chunk_sizes[chunk_idx] = num_lines
            self.index_mapping.extend([(chunk_idx, i) for i in range(num_lines)])

        # ✅ Ensure dataset length is correct
        self.dataset_length = len(self.index_mapping)

        # ✅ Preload first chunk (optimization)
        self.current_chunk_idx = None
        self.current_chunk_data = []

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.dataset_length:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.dataset_length}")

        chunk_idx, sample_idx = self.index_mapping[idx]

        # ✅ Load correct chunk if necessary
        if self.current_chunk_idx != chunk_idx:
            self.load_chunk(chunk_idx)

        record = self.current_chunk_data[sample_idx]
        input_ids = record['input_ids']
        labels = record['labels']

        # ✅ Proper padding (ensuring sequence alignment)
        input_ids = self.pad_sequence(input_ids, self.max_length)
        labels = self.pad_sequence(labels, self.max_length // 2)  # Prevent out-of-bounds

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def load_chunk(self, chunk_idx):
        """
        ✅ Efficient chunk loading with error handling
        """
        chunk_file = self.chunk_files[chunk_idx]
        with open(chunk_file, 'r', encoding='utf-8') as f:
            self.current_chunk_data = [json.loads(line.strip()) for line in f]
        self.current_chunk_idx = chunk_idx

    def pad_sequence(self, sequence, length):
        """
        ✅ Proper padding function to avoid mismatch issues.
        """
        if len(sequence) >= length:
            return sequence[:length]
        else:
            pad_len = length - len(sequence)
            return sequence + [self.tokenizer.pad_token_id] * pad_len

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
                                   k_reconstructed.transpose(-2, -1)) / (math.sqrt(self.d_model) + 1e-8)
        attn_scores = torch.nan_to_num(attn_scores, nan=-1e9)
        attn_scores = torch.clamp(attn_scores, -1e4, 1e4)

        # Apply time decay
        time_matrix = torch.arange(seq_len, device=x.device).float().unsqueeze(0).expand(seq_len, seq_len)
        time_decay = torch.exp(-self.lambda_decay * torch.abs(time_matrix - time_matrix.T))
        attn_scores = attn_scores * time_decay.unsqueeze(0).unsqueeze(0)
        attn_scores = torch.nan_to_num(attn_scores, nan=-1e9)
        attn_scores = torch.nan_to_num(attn_scores, nan=-1e9)
        attn_scores = torch.clamp(attn_scores, -1e4, 1e4)

        # Normalize and compute attention output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_reconstructed)

        # Merge heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)

        return output

class TimeAwareMultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, lambda_decay=0.01, use_wallclock_time=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent
        self.lambda_decay = lambda_decay
        self.use_wallclock_time = use_wallclock_time
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads

        # Attention weights
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Latent compression
        self.W_down_kv = nn.Linear(d_model, d_latent, bias=False)
        self.W_mem_kv = nn.Linear(d_model, d_latent, bias=False)
        self.W_up_k = nn.Linear(d_latent, d_model, bias=False)
        self.W_up_v = nn.Linear(d_latent, d_model, bias=False)

    def get_timestamps(self, x):
        """
        Create a timestamp tensor matching the sequence length.
        Returns: shape (batch, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        current_time = time.time()
        # simulate per-token arrival spaced by 0.01s (or real timestamps if known)
        return torch.tensor([[current_time + i * 0.01 for i in range(seq_len)]
                             for _ in range(batch_size)], device=x.device)

    def forward(self, x, memory=None, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.W_q(x)

        if memory is None:
            # Self-attention
            k = self.W_k(x)
            v = self.W_v(x)
        else:
            # Cross-attention with encoder output
            k = self.W_k(memory)
            v = self.W_v(memory)

        latent_kv = self.W_down_kv(k + v)
        # Just use memory directly:
        if memory is not None:
            latent_kv = self.W_down_kv(memory)
        else:
            latent_kv = self.W_down_kv(k + v)

        k_reconstructed = self.W_up_k(latent_kv)
        v_reconstructed = self.W_up_v(latent_kv)

        # Reshape
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_reconstructed = k_reconstructed.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_reconstructed = v_reconstructed.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Matrix multiply (batch, heads, q_seq, k_seq)
        attn_scores = torch.matmul(q, k_reconstructed.transpose(-2, -1)) / (math.sqrt(self.d_model) + 1e-8)

        # 🔹 Apply attention mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_scores = torch.nan_to_num(attn_scores, nan=-1e9)
        attn_scores = torch.clamp(attn_scores, -1e4, 1e4)

        if self.use_wallclock_time:
            if memory is None:
                # Self-attention: time decay between tokens in x
                token_timestamps = self.get_timestamps(x)  # shape: (batch, seq_len)
                time_diffs = torch.abs(token_timestamps.unsqueeze(2) - token_timestamps.unsqueeze(1))  # (batch, seq, seq)
                time_decay = torch.exp(-self.lambda_decay * time_diffs)  # (batch, seq, seq)
                time_decay = time_decay.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (batch, heads, seq, seq)
                attn_scores = attn_scores * time_decay

            else:
                # Cross-attention: decay between decoder tokens (q) and encoder memory tokens (k)
                q_timestamps = self.get_timestamps(x)               # shape: (batch, seq_q)
                k_timestamps = self.get_timestamps(memory)          # shape: (batch, seq_k)
                time_diffs = torch.abs(q_timestamps.unsqueeze(2) - k_timestamps.unsqueeze(1))  # (batch, seq_q, seq_k)
                time_decay = torch.exp(-self.lambda_decay * time_diffs)  # (batch, seq_q, seq_k)
                time_decay = time_decay.unsqueeze(1).expand(-1, self.num_heads, -1, -1)         # (batch, heads, seq_q, seq_k)
                attn_scores = attn_scores * time_decay
        attn_scores = torch.nan_to_num(attn_scores, nan=-1e9)
        attn_scores = torch.clamp(attn_scores, -1e4, 1e4)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_reconstructed)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_output), latent_kv



class TimeAwarePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048, lambda_time=0.01, use_wallclock_time=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lambda_time = lambda_time
        self.use_wallclock_time = use_wallclock_time

        # 🔹 Precompute sinusoidal positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("positional_encoding", pe.unsqueeze(0))  # (1, max_len, d_model)

    def get_timestamps(self, seq_len, batch_size, device):
        now = time.time()
        timestamps = [[now + i * 0.01 for i in range(seq_len)] for _ in range(batch_size)]
        return torch.tensor(timestamps, device=device).unsqueeze(-1)  # (batch, seq_len, 1)

    def forward(self, x, token_timestamps=None):
        """
        x: (batch, seq_len, d_model)
        token_timestamps: optional (batch, seq_len)
        """
        batch_size, seq_len, d_model = x.size()

        # 🔹 Get standard sinusoidal PE
        pos_pe = self.positional_encoding[:, :seq_len, :]  # (1, seq_len, d_model)
        x = x + pos_pe

        # 🔹 Add time-based information
        if self.use_wallclock_time:
            if token_timestamps is None:
                token_timestamps = self.get_timestamps(seq_len, batch_size, x.device)  # (batch, seq, 1)

            # Time signal scaled to match PE shape
            time_signal = token_timestamps * self.lambda_time  # Shape: (batch, seq_len, 1)
            time_embedding = torch.sin(time_signal * math.pi).expand(-1, -1, d_model)  # broadcast over features

            x = x + time_embedding

        return self.dropout(x)

    
    
class RuleTransform(nn.Module):
    def __init__(self,vocab_size, embedding_dim, max_rules, device):
        super().__init__()
        self.rule_transform = nn.Parameter(torch.randn(max_rules, embedding_dim), requires_grad=True)
        self.rule_mat = nn.Parameter(torch.randn(embedding_dim, max_rules), requires_grad=True)
        self.conv = nn.Linear(embedding_dim, embedding_dim)
        self.rule_scores = torch.zeros(max_rules, device=device)
        self.register_buffer("token_rules", torch.randint(0, max_rules, (vocab_size,)))
        self.embed_dim = embedding_dim

    def forward(self, hidden_states):
        # token_ids: (batch, seq)
        # hidden_states: (batch, seq, embedding_dim)
        #rule_matrices = (self.rule_mat*self.rule_transform).to(device)  # (batch, seq, d, d)
        rule_matrices = torch.tensordot(self.rule_mat, self.rule_transform, dims=1)

        rule_matrices = self.conv(rule_matrices)
        rule_matrices = F.tanh(self.conv(rule_matrices)).to(device)
        #print(rule_matrices.shape)
        rule_matrices = torch.einsum("bsd,de->bse", hidden_states, rule_matrices).to(device)
        #print(rule_matrices.shape)
        return rule_matrices

    def replace_rule(self):
        """
        Replace a low-scoring rule using crossover + occasional mutation.
        """
        k = min(self.max_rules, 10)
        bottom_k = torch.topk(self.rule_scores, k, largest=False).indices.to(device)
        worst_rule_idx = bottom_k[torch.randint(0, k, (1,))].item().to(device)

        top_k = torch.topk(self.rule_scores, k, largest=True).indices.to(device)
        r1 = self.rule_transform[top_k[torch.randint(0, k, (1,))]].to(device)
        r2 = self.rule_transform[top_k[torch.randint(0, k, (1,))]].to(device)
        new_rule = (r1 + r2) / 2

        # 1% chance of mutation
        if random.random() < 0.01:
            new_rule += torch.randn_like(new_rule) * 0.01

        with torch.no_grad():
            self.rule_transform[worst_rule_idx] = new_rule
            self.rule_scores[worst_rule_idx] = 0.0

        return worst_rule_idx
    def update_rule_scores(self, token_ids, loss_diff):
        rule_indices = self.token_rules[token_ids].detach().to(device)
        #self.rule_scores.index_add_(0, rule_indices.view(-1), torch.full_like(rule_indices.view(-1).float(), loss_diff))
        self.rule_scores[rule_indices] += loss_diff  
    def validate_and_replace_rule(self, model, val_input, val_target, decoder_input, loss_fn, k=5):
        """
        Uses local validation to evolve and evaluate crossover/mutated rules before replacing weak ones.
        """
        worst_k = torch.topk(self.rule_scores, k, largest=False).indices
        top_k = torch.topk(self.rule_scores, k, largest=True).indices

        # Create a candidate via crossover
        r1 = self.rule_transform[top_k[torch.randint(0, k, (1,))]]
        r2 = self.rule_transform[top_k[torch.randint(0, k, (1,))]]
        candidate_rule = (r1 + r2) / 2

        # Optional 2% mutation chance
        if random.random() < 0.02:
            candidate_rule += torch.randn_like(candidate_rule) * 0.1
        candidate_rule = candidate_rule.detach()

        best_loss = float('inf')
        best_idx = None

        for idx in worst_k:
            original = self.rule_transform[idx].clone()
            with torch.no_grad():
                self.rule_transform[idx] = candidate_rule

            with torch.no_grad():
                output = model(val_input, decoder_input)
                loss = loss_fn(output.view(-1, output.shape[-1]), val_target.view(-1))

            if loss < best_loss:
                best_loss = loss
                best_idx = idx

            with torch.no_grad():
                self.rule_transform[idx] = original

        if best_idx is not None:
            with torch.no_grad():
                self.rule_transform[best_idx] = candidate_rule
                self.rule_scores[best_idx] = 0.0
            print(f"✅ Validated and replaced rule at index {best_idx} with loss={best_loss:.4f}")

    def optimized_rule_replacement(self, model, val_input, val_target, decoder_input, loss_fn, k=5):
        """
        Optimizes rule_transform updates by evaluating candidates trained with different learning rates.
        """
        worst_k = torch.topk(self.rule_scores, k, largest=False).indices

        # Base candidate to start optimization
        learning_rates = [1e-5, 1e-6, 1e-4]
        candidates = []
        best_idx = None
        for idx in worst_k:
            base_rule = self.rule_transform[idx]

                
            for lr in learning_rates:
                candidate = base_rule.detach().clone().requires_grad_(True)
                optimizer = torch.optim.AdamW([candidate], lr=lr)

                # One step of gradient descent using validation batch
                output = model(val_input, decoder_input)
                loss = loss_fn(output.view(-1, output.shape[-1]), val_target.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if best_idx is None or loss < best_loss:
                    best_loss = loss
                    candidate = self.rule_transform[idx].detach().clone().requires_grad_(True)
                    best_idx = self.rule_transform[idx]
               
                candidates.append((candidate.detach(), loss.item()))

            # Select the best candidate based on validation loss
            best_candidate, best_loss = min(candidates, key=lambda x: x[1])
            if best_idx is not None:
                with torch.no_grad():
                    self.rule_transform[idx] = best_candidate
                    self.rule_scores[idx] = 0.0
                print(f"✅ Replaced rule at index {idx} using optimizer-based update with val_loss={best_loss:.4f}")
                


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, dropout=0.1, lambda_decay=0.01):
        super().__init__()
        self.self_attn = TimeAwareMultiHeadLatentAttention(d_model, num_heads, d_latent, lambda_decay)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        ffn_out = torch.clamp(ffn_out, -5.0, 5.0)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, dropout=0.1, lambda_decay=0.01):
        super().__init__()
        self.self_attn = TimeAwareMultiHeadLatentAttention(d_model, num_heads, d_latent, lambda_decay)
        self.cross_attn = TimeAwareMultiHeadLatentAttention(d_model, num_heads, d_latent, lambda_decay)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, mask=None):
        attn_out, _ = self.self_attn(x, mask=mask)  # Decoder self-attention
        x = self.norm1(x + self.dropout(attn_out))

        cross_out, _ = self.cross_attn(x, memory=encoder_output)  # Cross-attention
        x = self.norm2(x + self.dropout(cross_out))

        ffn_out = self.ffn(x)
        ffn_out = torch.clamp(ffn_out, -5.0, 5.0)

        x = self.norm3(x + self.dropout(ffn_out))
        return x

class Rule_Transformer_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, seq_length, lambda_decay=0.01, dropout=0.1, compression_factor=4, num_frequencies=100, max_rules=set_number_rules):
        super().__init__()
        self.embed_size = embedding_dim
        self.d_latent = embedding_dim // compression_factor
        self.vocab_size = vocab_size
        self.seq_length = seq_length

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = TimeAwarePositionalEncoding(embedding_dim, max_len=seq_length)

        self.encoder_layers = nn.ModuleList([
            EncoderBlock(embedding_dim, num_heads, self.d_latent, dropout, lambda_decay)
            for _ in range(num_layers)
        ])

        self.rule_transform = RuleTransform(vocab_size, embedding_dim, max_rules, device=device)

        self.decoder_layers = nn.ModuleList([
            DecoderBlock(embedding_dim, num_heads, self.d_latent, dropout, lambda_decay)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src_ids, tgt_ids):
        # --- INPUT EMBEDDING ---
        src_emb = self.token_embedding(src_ids)
        src_emb = self.pos_encoder(src_emb)
        for layer in self.encoder_layers:
            src_emb = layer(src_emb)

        # --- RULE TRANSFORM ---
        rule_encoded = self.rule_transform(src_emb)

        # --- DECODER PROCESS ---
        tgt_emb = self.token_embedding(tgt_ids)
        tgt_emb = self.pos_encoder(tgt_emb)
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, rule_encoded)

        return self.fc_out(tgt_emb)
    
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
                 
class FourierPositionalEncoding(nn.Module):
    """
    Standard Fourier-based positional encoding
    to replace learned positional embeddings.
    """

    def __init__(self, embedding_dim, max_length=seq_len):
        super().__init__()

        # Positional encoding using sine/cosine functions
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))

        pos_enc = torch.zeros(max_length, embedding_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        """Apply Fourier positional encoding."""
        return x + self.pos_enc[: x.shape[1], :].unsqueeze(0)

class HierarchicalMultiHeadLatentAttention2(nn.Module):
    """
    MultiHead Latent Attention modified to process hierarchical summary embeddings.
    - Operates on different levels (sentence, paragraph, document).
    """

    def __init__(self, embedding_dim, num_heads, latent_dim):
        super().__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim

        # Linear transformations for Query, Key, Value
        self.q_linear = nn.Linear(embedding_dim, latent_dim * num_heads)
        self.k_linear = nn.Linear(embedding_dim, latent_dim * num_heads)
        self.v_linear = nn.Linear(embedding_dim, latent_dim * num_heads)
        self.out_linear = nn.Linear(latent_dim * num_heads, embedding_dim)

        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, num_chunks, embed_dim = x.shape

        # Compute Q, K, V
        q = self.q_linear(x).view(batch_size, num_chunks, self.num_heads, self.latent_dim)
        k = self.k_linear(x).view(batch_size, num_chunks, self.num_heads, self.latent_dim)
        v = self.v_linear(x).view(batch_size, num_chunks, self.num_heads, self.latent_dim)

        # Compute scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.latent_dim ** 0.5)
        attn_scores = torch.nan_to_num(attn_scores, nan=-1e9)
        attn_scores = torch.clamp(attn_scores, -1e4, 1e4)

        attn_weights = self.softmax(attn_scores)
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate attention heads and pass through output layer
        attn_output = attn_output.view(batch_size, num_chunks, self.num_heads * self.latent_dim)
        return self.out_linear(attn_output)


class FourierSummaryEmbedding(nn.Module):
    """
    Generates summary embeddings for hierarchical levels using:
    - Fourier-based encoding for positional awareness.
    - Learnable summary embeddings for hierarchical abstraction.
    """

    def __init__(self, embedding_dim, max_levels, max_length=seq_len):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_levels = max_levels

        # Fourier Positional Encoding
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))

        pos_enc = torch.zeros(max_length, embedding_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pos_enc", pos_enc)

        # Learnable Summary Embeddings
        self.summary_embeddings = nn.Embedding(max_levels, embedding_dim)

    def forward(self, x, level):
        """Apply Fourier encoding + learned summary embedding based on level."""
        pos_encoded = x + self.pos_enc[: x.shape[1], :]#.unsqueeze(0)
        #print(pos_encoded.shape)
        level = torch.tensor(level).to(device)
        #print(self.summary_embeddings(level).shape)
        level_embedding = self.summary_embeddings(level)#.unsqueeze(1)
        #print(level_embedding.shape)

        return pos_encoded + level_embedding  # 🔹 Mix Fourier and learned embeddings

class HierarchicalMultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, lambda_decay=0.01, memory_size=8):
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
                                   k_reconstructed.transpose(-2, -1)) / (math.sqrt(self.d_model) + 1e-8)
        attn_scores = torch.nan_to_num(attn_scores, nan=-1e9)
        attn_scores = torch.clamp(attn_scores, -1e4, 1e4)

        # Apply time decay
        time_matrix = torch.arange(seq_len, device=x.device).float().unsqueeze(0).expand(seq_len, seq_len)
        time_decay = torch.exp(-self.lambda_decay * torch.abs(time_matrix - time_matrix.T))
        attn_scores = attn_scores * time_decay.unsqueeze(0).unsqueeze(0)
        attn_scores = torch.nan_to_num(attn_scores, nan=-1e9)
        attn_scores = torch.clamp(attn_scores, -1e4, 1e4)
        # Normalize and compute attention output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_reconstructed)

        # Merge heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)

        return output
    
chunksize=32
class Transformer_Model(nn.Module):
    """
    Transformer model that processes input hierarchically,
    summarizing lower levels into Fourier-based embeddings,
    and applying MultiHead Latent Attention (MHLA).
    """

    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, seq_length, hierarchy_levels=3, chunk_size=chunksize):
        super().__init__()
        self.embed_size = embedding_dim
        self.hierarchy_levels = hierarchy_levels
        self.chunk_size = chunk_size

        # 🔹 Token Embedding (Standard Transformer Style)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 🔹 Fourier Summary Embeddings for hierarchical levels
        self.hierarchical_embedding = FourierSummaryEmbedding(embedding_dim, hierarchy_levels)
        self.summarize = nn.Linear(embedding_dim*chunk_size, embedding_dim)

        # 🔹 MultiHead Latent Attention at each level
        self.mhla_layers = nn.ModuleList([
            HierarchicalMultiHeadLatentAttention(embedding_dim, num_heads, embedding_dim // 2) for _ in range(num_layers)

        ])
        self.extrapolation = nn.Linear(embedding_dim, chunk_size*embedding_dim)
        # 🔹 Output Projection
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src, level=0):
        """
        Processes input hierarchically:
        - src: input token sequences.
        - level: hierarchy level.
        """
        batch_size, seq_len = src.shape

        # 🔹 Token Embeddings
        token_embeddings = self.embedding(src)
        logging.debug(f"tokenembeddings Shape: {token_embeddings.shape}")

        # 🔹 Chunking input into hierarchical units
        num_chunks = (seq_len // self.chunk_size) 
        logging.debug(f"numchunks: {num_chunks}")

        hierarchical_chunks = token_embeddings.view(batch_size, num_chunks,self.chunk_size*self.embed_size)
        logging.debug(f"hierarchical_chunks Shape: {hierarchical_chunks.shape}")

        # 🔹 Generate summary embeddings per chunk
        summaries = self.summarize(hierarchical_chunks)#.squeeze(-1)  # 🔹 Average embeddings per chunk
        logging.debug(f"summaries Shape: {summaries.shape}")

        hierarchical_embeddings = self.hierarchical_embedding(summaries, level)  # 🔹 Apply Fourier encoding
        logging.debug(f"hierarchicalembeddings Shape1: {hierarchical_embeddings.shape}")

        # 🔹 Apply MultiHead Latent Attention
        for mhla in self.mhla_layers:
            hierarchical_embeddings = mhla(hierarchical_embeddings)
        logging.debug(f"hierarchicalembeddings Shape2: {hierarchical_embeddings.shape}")

        hierarchical_embeddings = self.extrapolation(hierarchical_embeddings)
        logging.debug(f"hierarchicalembeddings Shape3: {hierarchical_embeddings.shape}")

        # 🔹 Final projection
        output = self.fc_out(hierarchical_embeddings.view(batch_size, seq_len, self.embed_size))
        logging.debug(f"output Shape: {output.shape}")

        return output

def sample_gumbel(shape, eps=1e-20, device='cpu'):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution """
    gumbel_noise = sample_gumbel(logits.shape, device=logits.device)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [*, num_classes] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but still use softmax gradients
    Returns:
        [*, num_classes] sample from the Gumbel-Softmax distribution.
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        # Straight-through trick: make hard one-hot output, but keep soft gradients
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        # Set gradients of y_hard equal to those of y
        y = (y_hard - y).detach() + y
    logging.debug(f"Gumbel shape: {y.shape}") 

    return y

def greedy_sample(logits):
    """ Converts raw model outputs into discrete tokens using greedy sampling. """
    probs = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities
    return torch.argmax(probs, dim=-1)  # Select the most probable token


def propagate_embedding_size(new_model, new_dim):
    """
    Propagates the new embedding size throughout the model's layers.
    """
    # Update positional encoding if it exists
    if hasattr(new_model, 'pos_encoder'):
        new_model.pos_encoder = PositionalEncoding(new_dim, dropout=0.1, max_len=seq_len)
    
    # Update Transformer layers
    if hasattr(new_model, 'transformer') and isinstance(new_model.transformer, nn.Transformer):
        # Reinitialize the transformer with the new dimension
        new_model.transformer = nn.Transformer(
            d_model=new_dim,
            nhead=new_model.transformer.encoder.layers[0].self_attn.num_heads,
            num_encoder_layers=len(new_model.transformer.encoder.layers),
            num_decoder_layers=len(new_model.transformer.decoder.layers),
            dim_feedforward=new_model.transformer.encoder.layers[0].linear1.out_features,
            dropout=new_model.transformer.encoder.layers[0].dropout.p,
            batch_first=True
        )
    
    # Update output projection layer
    if hasattr(new_model, 'fc_out'):
        new_model.fc_out = nn.Linear(new_dim, new_model.fc_out.out_features)

    # Update all MultiheadAttention layers
    if hasattr(new_model.transformer.encoder, 'layers'):
        for layer in new_model.transformer.encoder.layers:
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, nn.MultiheadAttention):
                layer.self_attn.embed_dim = new_dim
                layer.self_attn.kdim = new_dim
                layer.self_attn.vdim = new_dim
                layer.self_attn.q_proj_weight = nn.Parameter(torch.randn(new_dim, new_dim))
                layer.self_attn.k_proj_weight = nn.Parameter(torch.randn(new_dim, new_dim))
                layer.self_attn.v_proj_weight = nn.Parameter(torch.randn(new_dim, new_dim))
    if hasattr(new_model.transformer.decoder, 'layers'):
        for layer in new_model.transformer.decoder.layers:
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, nn.MultiheadAttention):
                layer.self_attn.embed_dim = new_dim
                layer.self_attn.kdim = new_dim
                layer.self_attn.vdim = new_dim
                layer.self_attn.q_proj_weight = nn.Parameter(torch.randn(new_dim, new_dim))
                layer.self_attn.k_proj_weight = nn.Parameter(torch.randn(new_dim, new_dim))
                layer.self_attn.v_proj_weight = nn.Parameter(torch.randn(new_dim, new_dim))


class GeneticAlgorithm:
    def __init__(self, model, mutation_rate, population_size=10):
        self.model = model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [self._randomize_weights() for _ in range(population_size)]

    def _randomize_weights(self):
        new_model = copy.deepcopy(self.model)
        for param in new_model.parameters():
            param.data += torch.randn_like(param) * self.mutation_rate  # Mutate weights
        return new_model

    def select_best(self, loss_fn, inputs, target_labels, decoder_input, architecture):
        best_model = None
        best_loss = float('inf')
        n=0
        for model in self.population:
            loss = 0
            if architecture == "Rule Transformer":

                output = self.model(inputs, decoder_input)

            else:
                output = self.model(inputs)          
                
            output = output.reshape(-1, output.shape[-1])
            logging.debug(f"output reshaped Shape: {output.shape}")
            target_labels_reshaped = target_labels.reshape(-1)
            logging.debug(f"target reshaped Labels Shape: {target_labels_reshaped.shape}")
            loss = loss_fn(output, target_labels_reshaped)
            if loss < best_loss:
                    best_loss = loss
                    n=n+1
                    print(f"Best model iteration {n}, Loss: {loss.item()}")
                    best_model = model
            
            else:
                loss = 0

                if architecture == "Rule Transformer":

                    output = self.model(inputs, decoder_input)

                else:
                    output = self.model(inputs)
                # Flatten logits and targets:
                output = output.reshape(-1, output.shape[-1])
                logging.debug(f"output reshaped Shape: {output.shape}")
                target_labels_reshaped = target_labels.reshape(-1)
                logging.debug(f"target reshaped Labels Shape: {target_labels_reshaped.shape}")
                loss = loss_fn(output, target_labels_reshaped)
                n=n+1
                print(f"Iteration {n}, Loss: {loss}")
                if loss < best_loss:
                        best_loss = loss
                        n=n+1
                        print(f"Best model iteration {n}, Loss: {loss.item()}")
                        best_model = model
        return best_model

    def evolve(self, loss_fn, inputs, target_labels, decoder_input, architecture):
        best_model = self.select_best(loss_fn, inputs, target_labels, decoder_input, architecture)
        self.population = [copy.deepcopy(best_model) for _ in range(self.population_size)]
        for model in self.population:
            for param in model.parameters():
                param.data += torch.randn_like(param) * self.mutation_rate  # Apply mutation
        # Return the best model from the new population.
        return self.select_best(loss_fn, inputs, target_labels, decoder_input, architecture)
    
class FireflyOptimizer:
    def __init__(self, model, num_fireflies=5, alpha=0.1, beta=0.5, crossover_rate=0.05, embed_growth_rate=0.02):
        self.population = [copy.deepcopy(model) for _ in range(num_fireflies)]
        self.alpha = alpha  # Random movement magnitude
        self.beta = beta    # Attraction towards brighter fireflies
        self.crossover_rate = crossover_rate
        self.embed_growth_rate = embed_growth_rate

    def move_towards(self, firefly1, firefly2):
        for p1, p2 in zip(firefly1.parameters(), firefly2.parameters()):
            # Attraction towards brighter firefly + Random perturbation
            p1.data += self.beta * (p2.data - p1.data) + self.alpha * torch.randn_like(p1)

    def mutate_topology(self, model):
        new_model = copy.deepcopy(model)
        
        # Weighted probability for embedding growth or layer addition
        
        # Preferentially grow embedding size
        if random.random() < self.embed_growth_rate:
            if hasattr(new_model, 'embedding') and isinstance(new_model.embedding, nn.Embedding):
                new_dim = int(new_model.embedding.embedding_dim * 1.1) + 1
                num_embeddings = new_model.embedding.num_embeddings
                new_weight = torch.cat(
                    [new_model.embedding.weight, 
                    torch.randn((num_embeddings, new_dim - new_model.embedding.embedding_dim)).to(new_model.embedding.weight.device)],
                    dim=1
                )
                new_model.embedding = nn.Embedding(num_embeddings, new_dim, _weight=new_weight)
                print(f"Embedding dimension increased to: {new_dim}")
            return new_model
        
        # Add new Transformer Encoder Layer
        if random.random() < 0.003:  # 30% chance of adding a new layer
            if hasattr(new_model, 'transformer') and isinstance(new_model.transformer, nn.Transformer):
                # Access encoder layers
                encoder_layers = new_model.transformer.encoder.layers
                if isinstance(encoder_layers, nn.ModuleList):
                    layer_choice = random.choice(encoder_layers)
                    new_layer = nn.TransformerEncoderLayer(
                        d_model=layer_choice.self_attn.embed_dim,
                        nhead=layer_choice.self_attn.num_heads,
                        dim_feedforward=layer_choice.linear1.out_features,
                        activation='relu'
                    )
                    encoder_layers.insert(random.randint(0, len(encoder_layers)), new_layer)
                    print("New Transformer Encoder layer added.")
            
            # Access decoder layers
            decoder_layers = new_model.transformer.decoder.layers
            if isinstance(decoder_layers, nn.ModuleList):
                layer_choice = random.choice(decoder_layers)
                new_layer = nn.TransformerDecoderLayer(
                    d_model=layer_choice.self_attn.embed_dim,
                    nhead=layer_choice.self_attn.num_heads,
                    dim_feedforward=layer_choice.linear1.out_features,
                    activation='relu'
                )
                decoder_layers.insert(random.randint(0, len(decoder_layers)), new_layer)
                print("New Transformer Decoder layer added.")
            
        return new_model


    def crossover(self, parent1, parent2):
        child = copy.deepcopy(parent1)
        for child_param, param1, param2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):

            crossover_mask = (torch.rand_like(child_param) < self.crossover_rate).float()
            child_param.data = param1.data * crossover_mask + param2.data * (1 - crossover_mask)
        return child

    def optimize(self, loss_fn, inputs, targets, architecture):
        fitness = [self.calculate_fitness(m, loss_fn, inputs, targets, architecture) for m in self.population]
        best_idx = torch.argmax(torch.tensor(fitness))
        best_firefly = self.population[best_idx]

        # Move each firefly towards the brightest one
        for i in range(len(self.population)):
            if i != best_idx:
                self.move_towards(self.population[i], best_firefly)
        
        # Crossover for enhanced diversity
        next_population = []
        for _ in range(len(self.population) // 2):
            parent1, parent2 = random.sample(self.population, 2)
            child = self.crossover(parent1, parent2)
            next_population.append(child)

        # Mutate copies of the best firefly
        for _ in range(len(self.population) - len(next_population)):
            mutated = self.mutate_topology(copy.deepcopy(best_firefly))
            next_population.append(mutated)
        
        # Update population
        self.population = next_population
        return best_firefly
    
    def calculate_fitness(self, model, loss_fn, inputs, targets, decoder_input, architecture):

        total_loss = 0
        n=0
        with torch.no_grad():
                if architecture == "Rule Transformer":
                    decoder_input = targets[:, :-1]
                    output = model(inputs, decoder_input)
                else:
                    output = model(inputs, decoder_input)
                logging.debug(f"Shape of outputs: {output.shape}")
                # Assume batch_labels are tensors of shape [batch_size, seq_len, vocab_size]

                output = output.reshape(-1, output.shape[-1])
                logging.debug(f"output reshaped Shape: {output.shape}")
                targets = targets.reshape(-1)
                logging.debug(f"target reshaped Labels Shape: {targets.shape}")
                loss = loss_fn(output, targets)
                n+=1
                print(f"Iteration: {n} loss:{loss.item()}")
                total_loss += loss.item()
        return 1.0 / (total_loss + 1e-8)  # Inverse of loss as fitness
    
class NEAT:
    def __init__(self, model, population_size=5, mutation_rate=0.1, crossover_rate=0.1, embed_growth_rate=0.01):
        self.population = [copy.deepcopy(model) for _ in range(population_size)]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.embed_growth_rate = embed_growth_rate

    def mutate_topology(self, model):
        new_model = copy.deepcopy(model)
        
        ### Step 1: Embedding Growth ###
        if random.random() < self.embed_growth_rate:
            if hasattr(new_model, 'embedding') and isinstance(new_model.embedding, nn.Embedding):
                # Calculate the new dimension by 10% increase and ensure it's even
                new_dim = int(new_model.embedding.embedding_dim * 1.1) + 1
                # Force even embedding size
                if new_dim % 2 != 0:
                    new_dim += 1
                
                # Ensure divisibility by nhead
                if new_dim % new_model.transformer.encoder.layers[0].self_attn.num_heads != 0:
                    new_dim += new_model.transformer.encoder.layers[0].self_attn.num_heads - (new_dim % new_model.transformer.encoder.layers[0].self_attn.num_heads)
                
                num_embeddings = new_model.embedding.num_embeddings
                new_weight = torch.cat(
                    [new_model.embedding.weight, 
                    torch.randn((num_embeddings, new_dim - new_model.embedding.embedding_dim)).to(new_model.embedding.weight.device)],
                    dim=1
                )
                new_model.embedding = nn.Embedding(num_embeddings, new_dim, _weight=new_weight)
                print(f"Embedding dimension increased to: {new_dim}")

                # Propagate embedding size change
                propagate_embedding_size(new_model, new_dim)

            return new_model

        
        ### Step 2: Add New Encoder Layer ###
        if random.random() < 0.001:
            if hasattr(new_model, 'transformer') and isinstance(new_model.transformer, nn.Transformer):
                # Access encoder layers
                encoder_layers = new_model.transformer.encoder.layers
                if isinstance(encoder_layers, nn.ModuleList):
                    layer_choice = random.choice(encoder_layers)
                    new_layer = nn.TransformerEncoderLayer(
                        d_model=layer_choice.self_attn.embed_dim,
                        nhead=layer_choice.self_attn.num_heads,
                        dim_feedforward=layer_choice.linear1.out_features,
                        activation='relu'
                    )
                    encoder_layers.append(new_layer)  # Use append instead of insert
                    # Re-register all layers after mutation
                    new_model.transformer.encoder.layers = nn.ModuleList(new_model.transformer.encoder.layers)
                    new_model.transformer.decoder.layers = nn.ModuleList(new_model.transformer.decoder.layers)
                    print("New Transformer Encoder layer added and registered.")
                    propagate_embedding_size(new_model, layer_choice.self_attn.embed_dim)
                    # After mutation, propagate size change to the population

        ### Step 3: Add New Decoder Layer ###
        if random.random() < 0.001:
            if hasattr(new_model, 'transformer') and isinstance(new_model.transformer, nn.Transformer):
                decoder_layers = new_model.transformer.decoder.layers
                if isinstance(decoder_layers, nn.ModuleList):
                    layer_choice = random.choice(decoder_layers)
                    new_layer = nn.TransformerDecoderLayer(
                        d_model=layer_choice.self_attn.embed_dim,
                        nhead=layer_choice.self_attn.num_heads,
                        dim_feedforward=layer_choice.linear1.out_features,
                        activation='relu'
                    )
                    decoder_layers.append(new_layer)  # Use append instead of insert
                    # Re-register all layers after mutation
                    new_model.transformer.encoder.layers = nn.ModuleList(new_model.transformer.encoder.layers)
                    new_model.transformer.decoder.layers = nn.ModuleList(new_model.transformer.decoder.layers)
                    print("New Transformer DEcoder layer added and registered.")
                    propagate_embedding_size(new_model, layer_choice.self_attn.embed_dim)
                    # After mutation, propagate size change to the population

        return new_model

    def crossover(self, parent1, parent2):
        child = copy.deepcopy(parent1)
        for child_param, param1, param2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            # Check if dimensions match
            if child_param.size() == param1.size() == param2.size():
                crossover_mask = (torch.rand_like(child_param) < self.crossover_rate).float()
                child_param.data = param1.data * crossover_mask + param2.data * (1 - crossover_mask)
            else:
                print("Dimension mismatch during crossover. Adjusting...")
                # Adjust to smallest common dimension
                min_shape = torch.Size([min(s1, s2, s3) for s1, s2, s3 in zip(child_param.size(), param1.size(), param2.size())])
                child_param.data = param1.data[:min_shape[0], :min_shape[1]] * 0.5 + param2.data[:min_shape[0], :min_shape[1]] * 0.5
                
                # Reshape child_param to the adjusted size
                child_param.data = child_param.data.view(min_shape)
        print("DEBUG - Crossover complete")
        return child


    def select_best(self, loss_fn, inputs, target_labels, decoder_input, architecture):
        best_model = None
        best_loss = float('inf')
        inputs.to(device)
        target_labels.to(device)
        decoder_input.to(device)
        for model in self.population:
            n = 0
            model.to(device)
            with torch.no_grad():
                loss = 0
                if architecture == "Rule Transformer":
                    output = model(inputs, decoder_input)
                else:
                    output = model(inputs, decoder_input)
                    
                output = output.reshape(-1, output.shape[-1])
                target_labels_reshaped = target_labels.reshape(-1)
                loss = loss_fn(output, target_labels_reshaped)
                n+=1
                print(f"Iteration: {n} loss:{loss.item()}")

                if loss < best_loss:
                    best_loss = loss
                    best_model = model
                    
        return best_model

    def evolve(self, loss_fn, inputs, target_labels, decoder_input, architecture):
        # Select the best model
        best_model = self.select_best(loss_fn, inputs, target_labels, decoder_input, architecture)
        best_model.to(device)
        # Create the next generation
        next_population = []
        
        # Perform crossover for half of the population
        for _ in range(len(self.population) // 2):
            parent1 = random.choice(self.population)
            parent2 = random.choice(self.population)
            child = self.crossover(parent1, parent2)
            logging.debug("before  next_population append crossover neat")
            next_population.append(child)
            logging.debug("after next_population append crossover neat")
    
        propagate_population_size(best_model, next_population)

        # Retain some mutated copies of the best model
        for _ in range(len(self.population) - len(next_population)):
            mutated_model = self.mutate_topology(copy.deepcopy(best_model))
            logging.debug("before next_population append mutate neat")
            next_population.append(mutated_model)
            logging.debug("after next_population append mutate neat")

        propagate_population_size(best_model, next_population)

        # Update population
        self.population = next_population
        
        # Return the best model from the new population
        return self.select_best(loss_fn, inputs, target_labels, decoder_input, architecture)


def propagate_population_size(new_model, population):
    """
    Ensures consistent dimensions across the population after mutation.
    """
    for model in population:
        if hasattr(model, 'embedding') and hasattr(new_model, 'embedding'):
            new_dim = new_model.embedding.embedding_dim
            model.embedding = nn.Embedding(new_model.embedding.num_embeddings, new_dim)
            propagate_embedding_size(model, new_dim)


class HybridNEATFireflyOptimizer:
    def __init__(self, model, loss_fn, population_size=5, mutation_rate=0.1, crossover_rate=0.1, embed_growth_rate=0.001, alpha=0.1, beta=0.5, learning_rate=0.001):
        self.population = [copy.deepcopy(model) for _ in range(population_size)]
        self.loss_fn = loss_fn
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.embed_growth_rate = embed_growth_rate
        self.alpha = alpha  # Random movement magnitude
        self.beta = beta    # Attraction towards brighter fireflies
        self.learning_rate = learning_rate
        
        # Initialize optimizers for each model
        self.optimizers = [torch.optim.AdamW(m.parameters(), lr=self.learning_rate, momentum=0.9) 
                   for m in self.population]
        self.fitness_history = []

    def calculate_fitness(self, loss_fn, model, inputs, targets, decoder_input, architecture):
        total_loss = 0

        with torch.no_grad():
                logging.debug(f"calculate fitness")
                inputs.to(device)
                targets.to(device)
                model.to(device)
                if architecture == "Rule Transformer":
                    decoder_input = targets[:, :-1]
                    output = model(inputs, decoder_input)
                else:
                    output = model(inputs, decoder_input)
                logging.debug(f"Shape of outputs: {output.shape}")
                # Assume batch_labels are tensors of shape [batch_size, seq_len, vocab_size]

                output = output.reshape(-1, output.shape[-1])
                logging.debug(f"output reshaped Shape: {output.shape}")
                targets = targets.reshape(-1)
                logging.debug(f"target reshaped Labels Shape: {targets.shape}")
                loss = loss_fn(output, targets)
                total_loss += loss.item()
        return 1.0 / (total_loss + 1e-8)  # Inverse of loss as fitness

    def mutate_topology(self, model, fitness):
        new_model = copy.deepcopy(model)
        
        # Weighted probability for embedding growth or layer addition
        embed_growth_prob = self.embed_growth_rate * fitness
        layer_add_prob = (1 - self.embed_growth_rate) * fitness
        
        ### Step 1: Embedding Growth ###
        if random.random() < embed_growth_prob:
            if hasattr(new_model, 'embedding') and isinstance(new_model.embedding, nn.Embedding):
                # Calculate the new dimension by 10% increase and ensure it's even
                new_dim = int(new_model.embedding.embedding_dim * 1.1) + 1
                # Force even embedding size
                if new_dim % 2 != 0:
                    new_dim += 1
                
                # Ensure divisibility by nhead
                if new_dim % new_model.transformer.encoder.layers[0].self_attn.num_heads != 0:
                    new_dim += new_model.transformer.encoder.layers[0].self_attn.num_heads - (new_dim % new_model.transformer.encoder.layers[0].self_attn.num_heads)
                
                num_embeddings = new_model.embedding.num_embeddings
                new_weight = torch.cat(
                    [new_model.embedding.weight, 
                    torch.randn((num_embeddings, new_dim - new_model.embedding.embedding_dim)).to(new_model.embedding.weight.device)],
                    dim=1
                )
                new_model.embedding = nn.Embedding(num_embeddings, new_dim, _weight=new_weight)
                print(f"Embedding dimension increased to: {new_dim}")

                # Propagate embedding size change
                propagate_embedding_size(new_model, new_dim)

            return new_model

        ### Step 2: Add New Encoder Layer ###
        if random.random() < layer_add_prob:
            if hasattr(new_model, 'transformer') and isinstance(new_model.transformer, nn.Transformer):
                # Access encoder layers
                encoder_layers = new_model.transformer.encoder.layers
                if isinstance(encoder_layers, nn.ModuleList):
                    layer_choice = random.choice(encoder_layers)
                    new_layer = nn.TransformerEncoderLayer(
                        d_model=layer_choice.self_attn.embed_dim,
                        nhead=layer_choice.self_attn.num_heads,
                        dim_feedforward=layer_choice.linear1.out_features,
                        activation='relu'
                    )
                    encoder_layers.append(new_layer)  # Use append instead of insert
                    # Re-register all layers after mutation
                    new_model.transformer.encoder.layers = nn.ModuleList(new_model.transformer.encoder.layers)
                    new_model.transformer.decoder.layers = nn.ModuleList(new_model.transformer.decoder.layers)
                    print("New Transformer Encoder layer added and registered.")
                    propagate_embedding_size(new_model, layer_choice.self_attn.embed_dim)
                    # After mutation, propagate size change to the population

        ### Step 3: Add New Decoder Layer ###
        if random.random() < layer_add_prob:
            if hasattr(new_model, 'transformer') and isinstance(new_model.transformer, nn.Transformer):
                decoder_layers = new_model.transformer.decoder.layers
                if isinstance(decoder_layers, nn.ModuleList):
                    layer_choice = random.choice(decoder_layers)
                    new_layer = nn.TransformerDecoderLayer(
                        d_model=layer_choice.self_attn.embed_dim,
                        nhead=layer_choice.self_attn.num_heads,
                        dim_feedforward=layer_choice.linear1.out_features,
                        activation='relu'
                    )
                    decoder_layers.append(new_layer)  # Use append instead of insert
                    # Re-register all layers after mutation
                    new_model.transformer.encoder.layers = nn.ModuleList(new_model.transformer.encoder.layers)
                    new_model.transformer.decoder.layers = nn.ModuleList(new_model.transformer.decoder.layers)
                    print("New Transformer DEcoder layer added and registered.")
                    propagate_embedding_size(new_model, layer_choice.self_attn.embed_dim)
                    # After mutation, propagate size change to the population

        return new_model


    def crossover(self, parent1, parent2):
        print("DEBUG - Crossover attempt")
        child = copy.deepcopy(parent1)
        for child_param, param1, param2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            # Check if dimensions match
            if child_param.size() == param1.size() == param2.size():
                crossover_mask = (torch.rand_like(child_param) < self.crossover_rate).float()
                child_param.data = param1.data * crossover_mask + param2.data * (1 - crossover_mask)
            else:
                print("Dimension mismatch during crossover. Skipping crossover for this layer.")
                continue  # Skip layers with dimension mismatches
        print("DEBUG - Crossover complete")
        return child


    def move_towards(self, firefly1, firefly2, fitness1, fitness2):
        for p1, p2 in zip(firefly1.parameters(), firefly2.parameters()):
            attraction_strength = self.beta * (fitness2 - fitness1)
            p1.data += attraction_strength * (p2.data - p1.data) + self.alpha * torch.randn_like(p1)

    def optimize(self, loss_fn, inputs, targets, decoder_input, architecture):
        # Calculate fitness for each model
        fitness_scores = [self.calculate_fitness(loss_fn, m, inputs, targets,decoder_input, architecture) for m in self.population]
        self.fitness_history.append(fitness_scores)
        logging.debug("fitness calc completed")
        
        # Identify the best model (brightest firefly)
        best_idx = torch.argmax(torch.tensor(fitness_scores))
        best_model = self.population[best_idx.to(device)]
        
        next_population = []

        # Step 1: Perform Mutation First
        for _ in range(len(self.population)):
            mutated_model = self.mutate_topology(copy.deepcopy(best_model), fitness_scores[best_idx])
            next_population.append(mutated_model)

        # Step 2: Propagate size change to the population
        propagate_population_size(best_model, next_population)
        logging.debug("mutate completed")

        # Step 3: Then Perform Crossover
        for i in range(len(next_population) // 2):
            parent1, parent2 = random.sample(next_population, 2)


            child = self.crossover(parent1, parent2)
            next_population.append(child)
        logging.debug("crossover completed")

        # Step 4: Firefly Attraction - Move towards brighter models
        for i in range(len(next_population)):
            if i != best_idx:  # Skip the best model itself
                self.move_towards(next_population[i].to(device), best_model.to(device), fitness_scores[i], fitness_scores[best_idx.to(device)])
        logging.debug("firefly completed")

        # Step 5: Isolate Optimizers and Clear Gradients
        for model in next_population:
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

            optimizer.zero_grad()
        
        n=0
        # Backpropagation Update
        for model, optimizer in zip(next_population, optimizer):
                model.train().to(device)
                logging.debug("training hybrid.")
                inputs.to(device)
                targets.to(device)
                optimizer.zero_grad()
                if architecture == "Rule Transformer":
                    decoder_input = targets[:, :-1]
                    output = model(inputs, decoder_input)
                else:
                    output = model(inputs, decoder_input)
                logging.debug(f"Shape of outputs: {output.shape}")
                # Assume batch_labels are tensors of shape [batch_size, seq_len, vocab_size]

                output = output.reshape(-1, output.shape[-1])
                logging.debug(f"output reshaped Shape: {output.shape}")
                targets = targets.reshape(-1)
                logging.debug(f"target reshaped Labels Shape: {targets.shape}")
                loss = loss_fn(output, targets)
                n+=1
                print(f"Iteration: {n} loss:{loss.item()}")

                loss.backward()
                optimizer.step()

        # Update population
        self.population = next_population
        return best_model


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

def prepare_decoder_input_and_target(target):
    """
    Prepares inputs and targets for teacher forcing when <BOS> is auto-generated by the tokenizer.
    - target: Tensor of shape (batch_size, seq_len)
    Returns:
    - decoder_input: Shifted target, including <BOS>
    - target_output: Original target
    """
    # Shift target to the right to form the decoder input
    decoder_input = torch.zeros_like(target)
    decoder_input[:, 1:] = target[:, :-1]  # Shift right
    decoder_input[:, 0] = target[:, 0]     # Copy the <BOS> from the target

    # The output is the target sequence itself (including <EOS>)
    target_output = target
    
    return decoder_input, target_output

def build_custom_validation_batch(tokenizer, seq_len=seq_len, device='cuda', batch_size=14):
    query_strings = [
        "1. What is 17 + 35?",
        "2. Solve for x: 2x + 5 = 13",
        "3. What is the derivative of x^2?",
        "4. What is the integral of x dx?",
        "5. What is the plural of 'analysis'?",
        "6. Is this sentence correct? 'He go to school every day.'",
        "7. What is the first law of Robotics?",
        "8. What is the secpnd law of robotics?",
        "9. What is the third law of robotics?,",
        "10. What is the zeroth law of robotics?",
        "11. What does this Python function return? def square(x): return x * x",
        "12. Write a function in Python that checks if a number is prime.",
        "13. What is the derivative of a function x^3 according to calculus?",
        "14. Describe the integral of a function x^3 according to calculus, please."
    ]

    target_strings = [
        "1. 52",
        "2. x = 4",
        "3. 2x",
        "4. (1/2)x^2 + C",
        "5. analyses",
        "6. No, it should be: 'He goes to school every day.'",
        "7. 1. A robot may not injure a human being or, through inaction, allow a human being to come to harm.",
        "8. 2. A robot must obey orders given by humans except where such orders would conflict with the First Law.",
        "9. 3. A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.",
        "10. 0. A robot may not harm humanity, or, by inaction, allow humanity to come to harm.",
        "11. It returns the square of x.",
        "12. def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "13. The derivative of x^3 by the power law for derivatives would be 3x^2.",
        "14. According to the integral power law the integral of x^3 would be (x^2)/2."
    ]

    input_ids, target_ids = [], []
    for query, target in zip(query_strings, target_strings):
        q_ids = tokenizer.encode(query, max_length=seq_len, truncation=True, padding='max_length')
        a_ids = tokenizer.encode(target, max_length=seq_len, truncation=True, padding='max_length')

        input_ids.append(q_ids)
        target_ids.append(a_ids)

    input_tensor = torch.tensor(input_ids[:batch_size], device=device)
    target_tensor = torch.tensor(target_ids[:batch_size], device=device)
    return input_tensor, target_tensor

class ReasoningModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Reasoning Model GUI")

        # Transformer Parameters
        self.layers = []
        # Model Configuration Variables
        self.model_name = tk.StringVar(value="Reasoning Model")
        self.num_parameters = tk.IntVar(value=1024)
        self.num_heads = tk.IntVar(value=12)
        self.vocab_size = tk.IntVar(value=10000)
        self.hidden_size = tk.IntVar(value=768)
        self.num_layers = tk.IntVar(value=12)

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
        self.learning_rate = tk.DoubleVar(value=0.0001)
        self.epochs = tk.IntVar(value=10)

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
        ttk.Checkbutton(data_frame, text="Use Chunked Dataset", variable=self.use_chunked_dataset).pack(pady=5)

        ttk.Button(data_frame, text="Select Dataset Directory", command=self.select_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Load Dataset", command=self.load_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Load Tokenizer", command=self.load_tokenizer).pack(pady=5)
        #ttk.Button(data_frame, text="Save Dataset as Text File", command=self.save_dataset_as_text).pack(pady=5)

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
        #ttk.Button(train_frame, text="Run Validation", command=self.run_validation_button).grid(row=5, column=0, pady=5)
        ttk.Button(train_frame, text="Test Inference", command=self.test_inference).grid(row=4, column=1, pady=5)

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
                    max_generated = 50
                    generated_tokens = []
                    generated = []
                    repetition_penalty = 1.2  # Adjust for stronger penalty
                    top_p = 0.9  # Cumulative probability threshold

                    self.model.eval()
                    tokenizer = self.tokenizer
                    with torch.no_grad():
                        # Tokenize prompt → fixed encoder input
                        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                        encoder_input_len = input_ids.size(1)

                        # Pad encoder input to max model length
                        if encoder_input_len < seq_len:
                            pad_len = seq_len - encoder_input_len
                            pad_token_id = tokenizer.pad_token_id or 0
                            padding = torch.full((1, pad_len), pad_token_id, dtype=torch.long).to(device)
                            input_ids = torch.cat([input_ids, padding], dim=1)
                        else:
                            input_ids = input_ids[:, :seq_len]

                        # Encoder is static throughout generation
                        encoder_input_ids = input_ids

                        # Setup initial decoder input
                        bos_token_id = tokenizer.bos_token_id or tokenizer.pad_token_id or 0
                        tgt_ids = torch.tensor([[bos_token_id]], device=device)

                        for _ in range(max_generated):
                            # Forward pass through model
                            outputs = self.model(encoder_input_ids, tgt_ids)
                            logits = outputs[:, -1, :]  # (batch, vocab)

                            # Repetition penalty
                            for token in set(tgt_ids[0].tolist()):
                                if token not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
                                    logits[0, token] /= repetition_penalty

                            # Top-p sampling
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            filtered_logits = logits.clone()
                            filtered_logits[0, sorted_indices[0][sorted_indices_to_remove[0]]] = float('-inf')

                            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

                            # Stop at EOS
                            if next_token_id.item() == tokenizer.eos_token_id:
                                break

                            # Append and continue
                            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
                            print(tgt_ids)
                            # Pad if needed to align with model
                            if tgt_ids.size(1) > seq_len:
                                tgt_ids = tgt_ids[:, -seq_len:]
                            generated.append(self.tokenizer.decode(next_token_id[0].tolist()))

                else:

                    self.model.eval()
                    with torch.no_grad():
                        # Tokenize the prompt and move to the correct device.
                        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
                        # Pad input_ids to the maximum sequence length (512)
                        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                        generated_text = input_ids
                        generated = []
                        # Ensure input length is multiple of chunk size
                        original_length = input_ids.shape[1]
                        pad_length = (chunksize - (original_length % chunksize)) % chunksize  # Only pad if needed
                        if pad_length > 0:
                            pad_tokens = torch.full((1, pad_length), pad_token_id, dtype=torch.long).to(device)
                            input_ids = torch.cat([input_ids, pad_tokens], dim=1)
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
                            outputs = self.model(input_ids)
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
                            input_ids = input_ids[input_ids != pad_token_id].unsqueeze(0)
                            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                            logging.debug(f"input_ids: {input_ids}")
                            original_length = input_ids.shape[1]
                            pad_length = (chunksize - (original_length % chunksize)) % chunksize  # Only pad if needed
                            if pad_length > 0:
                                pad_tokens = torch.full((1, pad_length), pad_token_id, dtype=torch.long).to(device)
                                input_ids = torch.cat([input_ids, pad_tokens], dim=1)
                            logging.debug(f"input_ids padded: {input_ids}")                                
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
            for batch_input_ids, batch_labels, seq_lengths in validation_loader:
                batch_input_ids = batch_input_ids.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_input_ids, batch_labels.reshape(-1))
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
                            self.input_ids, self.labels = [], []
                            for line in f:
                                record = json.loads(line)
                                self.input_ids.append(record['input_ids'])
                                self.labels.append(record['labels'])
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
                            input_ids, labels = self._generate_training_pairs(query, target, training_mode)
                            if input_ids and labels:
                                record = {'input_ids': input_ids, 'labels': labels}
                                f.write(json.dumps(record) + '\n')
                logging.info(f"Chunk {chunk_idx} tokenized and saved to {chunk_file_path}")

                messagebox.showinfo("Success", f"Data tokenized into {num_chunks} chunks and saved successfully to {self.tokenized_data_path}.")
                logging.info(f"Data tokenized into {num_chunks} chunks and saved successfully to {self.tokenized_data_path}.")
            else:
                with open(self.tokenized_data_path, 'w', encoding='utf-8') as f:
                    for query, target in self.query_target_pairs:
                        input_ids, labels = self._generate_training_pairs(query, target, training_mode)

                        if input_ids and labels:
                            self.input_ids.append(input_ids)  # Store for training
                            self.labels.append(labels)  # Store for training
                            record = {'input_ids': input_ids, 'labels': labels}


                            f.write(json.dumps(record) + '\n')
                logging.info(f"Input IDs: {len(self.input_ids)} sequences loaded.")
                logging.info(f"Labels: {len(self.labels)} sequences loaded.")
                logging.info(f"Input_ids sample: {self.input_ids[0][:10]}...")  # Shows only first 10 tokens
                logging.info(f"Labels sample: {self.labels[0][:10]}...")  # Shows only first 10 tokens

                messagebox.showinfo("Success", f"Data tokenized and saved successfully to {self.tokenized_data_path}.")
                logging.info(f"Data tokenized and saved successfully to {self.tokenized_data_path}.")
        except Exception as e:
            logging.error(f"Tokenization failed: {str(e)}")
            messagebox.showerror("Error", f"Tokenization failed: {str(e)}")

    def _generate_training_pairs(self, query, target, training_mode):
        # Debugging logs
        logging.debug(f"Generating Training Pairs - Query: {query}")
        logging.debug(f"Generating Training Pairs - Target: {target}")

        # Ensure inputs are valid strings before tokenization
        query_ids = self.tokenizer.encode(str(query) if query else "", truncation=True, max_length=seq_len)
        target_ids = self.tokenizer.encode(str(target) if target else "", truncation=True, max_length=seq_len)

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

        return input_ids, labels


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
                    seq_length=seq_len
                )
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            # Move the entire model to the selected device
            self.model.to(device)  # ✅ Reduces memory usage

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


    def calculate_learning_rate(self, total_params):
        # Calculate learning rate based on total parameters using the derived formula
        # LR = 17.38 * (Model Size)^-0.424
        lr = 17.38 * (total_params ** -0.424)
        return lr

    def start_training(self):
        # Start training in a separate thread to keep the GUI responsive
        self.stop_training.clear()
        training_thread = threading.Thread(target=self.training_loop)
        training_thread.start()

    def update_progress(self, progress_value):
        self.progress_bar['value'] = progress_value

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

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

    def training_loop(self):
        if not self.validate_training_parameters():
            return

        logging.info("All training parameters and data are properly initialized.")
        if not self.model:
            logging.error("Model not initialized before training")
            return
        self.use_genetic_algo = self.genetic_algo_var.get()

        try:

            # Convert LayerNorm to float32 (if applicable)
            #for module in self.model.modules():
            #    if isinstance(module, nn.LayerNorm):
            #        module.float()
            self.model.to(device)  # Move model to GPU
            if self.use_chunked_dataset.get():
                # Initialize the ChunkedDataset
                dataset = ChunkedDataset(
                    tokenized_data_path=self.tokenized_data_path,
                    tokenizer=self.tokenizer,
                    max_length=seq_len
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size.get(),
                    shuffle=True,
                    num_workers=2,    # Uses multiple CPU threads for data loading
                    prefetch_factor=1, # Prefetches data ahead of time
                    pin_memory=True,
                    collate_fn=collate_fn
                )
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else pad_token_id  # Default to global if tokenizer isn't set      
            else:
                # Initialize the standard dataset and dataloader
                device_cpu = 'cpu'
                # Ensure the tokenizer is loaded and has a valid pad_token_id
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else pad_token_id  # Default to global if tokenizer isn't set      
                max_length = seq_len  # Adjust as needed
                logging.info("max_length set")
                # Convert lists of token IDs to tensors and calculate original sequence lengths
                input_ids = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device_cpu)[:max_length]
                    for tokens in self.input_ids
                ]
                logging.info("input ids torched to tensor")

                labels = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device_cpu)[:max_length]
                    for tokens in self.labels
                ]
                logging.info("labels torched to tensor")

                # Stack tensors
                input_ids = torch.stack(input_ids).to(device_cpu)
                labels = torch.stack(labels).to(device_cpu)
                logging.info("datas stacked and torched")


                dataset = torch.utils.data.TensorDataset(input_ids, labels)
                logging.info("dataset torched")
                dataloader = DataLoader(
                    dataset,
                    batch_size=int(self.batch_size.get()),
                    shuffle=True,
                    num_workers=2,    # Uses multiple CPU threads for data loading
                    prefetch_factor=1, # Prefetches data ahead of time
                    pin_memory=True,
                    collate_fn=collate_fn
                )
                logging.info("dataloader defined")
            ##chunked vs. standard else complete

            # Adjust learning rate based on architecture
            total_params = self.num_parameters.get()
            lr = self.learning_rate.get()
            logging.info(f"Learning Rate: {lr} for total parameters: {total_params}")

            # Learning rate scheduler
            total_steps = self.epochs.get() * len(dataloader)
            logging.info(f"Total training steps: {total_steps}")
            # Separate parameters based on their shape.

            # Create two optimizers:
            #Enable for standard optimizer/scheduler
            #num_warmup_steps = total_steps // 10  # Warmup for 10% of training
            #scheduler = self.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)

            #optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
            #optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=lr, eps=1e-8)

            # 🔹 Find ≥2D parameters for Muon
            muon_params = [
                p for name, p in self.model.named_parameters()
                if p.ndim >= 2 and "rule_transform" not in name
            ]

            adamw_params = [
                p for name, p in self.model.named_parameters()
                if p.ndim < 2 or "bias" in name or "rule_scores" in name
                or ("embedding" in name) or  ("rule_transform" in name)
                or ("pos_encoder" in name)
            ]

            print("🧪 Muon param count:", sum(p.numel() for p in muon_params))
            print("🧪 AdamW param count:", sum(p.numel() for p in adamw_params))

            # 🔹 Create optimizers

            optimizers = [
                Muon(muon_params, lr=0.001, momentum=0.95),  
                torch.optim.AdamW(adamw_params, lr=1e-4, betas=(0.90, 0.95), weight_decay=0.01)
            ]

            logging.info("Optimizers defined")

            #loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01, ignore_index=pad_token_id)
            loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

            self.model.train()
            logging.info("Model set to training mode")
            progress_step = 0
            n = 0
            accumulation_steps = 1
            step_count = 0

            for epoch in range(self.epochs.get()):
                if self.stop_training.is_set():
                    logging.info("Training stopped by user.")
                    messagebox.showinfo("Info", "Training stopped by user.")
                    break
                for opt in optimizers:
                    opt.zero_grad()
                logging.debug("Optimizer gradients zeroed")
                self.accumulated_loss = 0
                epoch_loss = 0
                logging.info(f"Epoch {epoch+1} started")
                torch.cuda.empty_cache()

                # Training loop
                for batch_idx, (batch_input_ids, batch_labels) in enumerate(dataloader):
                    if self.stop_training.is_set():
                            logging.info("Training stopped by user.")
                            messagebox.showinfo("Info", "Training stopped by user.")
                            return

                        # Move batches and targets to the correct device 
                    batch_input_ids = batch_input_ids.to(device, non_blocking=True)
                    batch_labels = batch_labels.to(device, non_blocking=True)

                        # Logging epoch and batch info
                    logging.debug(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}')
                    logging.debug(f'Batch input_ids shape: {batch_input_ids.shape}')  # (batch_size, seq_len)
                    logging.debug(f'Using device: {self.device}')

                     # Prepare inputs and targets for teacher forcing
                    decoder_input, target_labels = prepare_decoder_input_and_target(batch_labels)
                    del batch_labels


                        # Log the shape of the combined mask
                    logging.debug(f'Decoder input shape: {decoder_input.shape}')  # (batch_size, seq_len)
                    logging.debug(f'Target labels shape: {target_labels.shape}')  # (batch_size, seq_len)
                    architecture = self.architecture.get()

                    with torch.amp.autocast(device, dtype=torch.float16):  # Enable mixed precision

                            # Check the flag and run evolution once per epoch if requested:
                        if self.use_genetic_algo == "Genetic Algorithm":
                                logging.info("Applying genetic algorithm evolution step...")
                                qga = GeneticAlgorithm(self.model, lr)
                                # Evolve using the same loss function and dataloader (or a validation subset)
                                self.model = qga.evolve(loss_fn, batch_input_ids, target_labels, decoder_input, architecture)
                                #Remove optimizer steps and gradient code enable this for Quaternion NeuroEvolution of Augmenting Topologies (NEAT)
                        elif self.use_genetic_algo == "NEAT":
                                neat = NEAT(self.model)
                                self.model = neat.evolve(loss_fn, batch_input_ids, target_labels, decoder_input, architecture)
                        elif self.use_genetic_algo == "Firefly":
                                #Remove optimizer steps and gradient lines to enable this for Quaternion Firefly Algo
                                firefly_optimizer = FireflyOptimizer(self.model)
                                self.model = firefly_optimizer.optimize(loss_fn, batch_input_ids, target_labels, decoder_input, architecture)
                        elif self.use_genetic_algo == "NF Hybrid":
                            # Initialize the Hybrid NEAT-Firefly Optimizer
                            hybrid_optimizer = HybridNEATFireflyOptimizer(
                                model=self.model,
                                loss_fn=loss_fn,
                                population_size=5,       # Number of models in the population
                                mutation_rate=0.1,       # Mutation rate for topology changes
                                crossover_rate=0.5,      # Crossover rate for combining models
                                embed_growth_rate=0.2,   # Probability of growing embedding size
                                alpha=0.1,               # Random movement magnitude
                                beta=0.5,                # Attraction strength towards better models
                                learning_rate=lr      # Learning rate for backpropagation updates
                            )
                            # Optimize and evolve the population
                            self.model = hybrid_optimizer.optimize(loss_fn, batch_input_ids, target_labels, decoder_input, architecture)

                            # Calculate fitness and log progress
                            fitness_scores = hybrid_optimizer.fitness_history[-1]
                            avg_fitness = sum(fitness_scores) / len(fitness_scores)
                            print(f"Average Fitness: {avg_fitness:.4f}")
                        else:
                            # Forward pass
                            try:
                                if architecture == "Rule Transformer":

                                    output = self.model(batch_input_ids, decoder_input)

                                else:
                                    output = checkpoint.checkpoint(self.model, batch_input_ids,  use_reentrant=False)
                            except Exception as e:
                                    raise ValueError(f"forward pass failed for {str(e)}")

                            logging.debug(f"Shape of outputs: {output.shape}")
                                # Assume batch_labels are tensors of shape [batch_size, seq_len, vocab_size]
                            output = output.reshape(-1, output.shape[-1])
                            logging.debug(f"output reshaped Shape: {output.shape}")
                            target_labels = target_labels.reshape(-1)
                            logging.debug(f"target reshaped Labels Shape: {target_labels.shape}")
                            
                            if architecture == "Rule Transformer":

                                loss = loss_fn(output, target_labels)
                                logging.debug(f"loss debug rule transformer: {loss}")

                            else:
                                loss = loss_fn(output, target_labels)
                        
                            logging.info(f"Loss computed: {loss.item()}")
                            self.accumulated_loss = self.accumulated_loss + loss
                            # ✅ Check for NaNs in loss
                            if torch.isnan(loss).any():
                                logging.warning("⚠️ Skipping optimizer step due to NaN loss.")
                            # Backward pass and optimization
                            scaler.scale(loss).backward()
                            # 🔹 Track how rules affected loss
                            prev_loss = loss.item()
                            logging.info("Loss backward computed")
                                
                                # Check for NaN or Inf in gradients
                            for name, param in self.model.named_parameters():
                                    if param.grad is not None:
                                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                            logging.error(f"Gradient for {name} contains NaN or Inf.")
                                        
                            for name, param in self.model.named_parameters():
                                if param.grad is not None:
                                        logging.debug(f"Gradient for {name}: mean={param.grad.mean().item():.4f}, max={param.grad.max().item():.4f}, min={param.grad.min().item():.4f}")
                                else:
                                        logging.debug(f"Gradient for {name} is None")

                            total_norm = 0.0
                            for p in self.model.parameters():
                                    if p.grad is not None:
                                        total_norm += p.grad.data.norm(2).item() ** 2
                            total_norm = total_norm ** 0.5
                            logging.info(f"Gradient norm: {total_norm}")

                            for name, param in self.model.named_parameters():
                                if param.grad is not None:
                                    logging.debug(f"Gradients for {name}: {param.grad}")
                                else:
                                    logging.debug(f"No gradients found for {name}.")
                            
                                
                            n+=1
                            print(f"Iteration {n}, Loss: {loss.item()}")
                            del loss
                                # Before optimizer step
                            for name, param in self.model.named_parameters():
                                    if param.requires_grad:
                                        logging.debug(f"Before step - {name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")
                            if (batch_idx + 1) % accumulation_steps == 0:     
                                    avg_loss = self.accumulated_loss / accumulation_steps  # Compute avg loss across accumulated steps

                                    # ✅ Unscale **all** optimizers before updating
                                    for opt in optimizers:
                                        scaler.unscale_(opt)
                                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # ✅ Clip gradients for stability

                                        # Log gradients for debugging
                                    for opt in optimizers:
                                        scaler.step(opt)
                                    logging.info("Optimizer step update completed")

                                    # After optimizer step
                                    for name, param in self.model.named_parameters():
                                        if param.requires_grad:
                                            logging.debug(f"After step - {name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")
                                    for opt in optimizers:
                                        opt.zero_grad()
                                    logging.debug("Optimizer gradients zeroed")
                                    scaler.update()
                                    if architecture == "Rule Transformer":
                                        logging.debug("Rule validation")
                                        #Test rules and generate new ones           
                                        with torch.no_grad():
                                            output_new = self.model(batch_input_ids, decoder_input)
                                            #output_new = model(src)
                                            new_loss = loss_fn(output_new[:, :seq_len, :].reshape(-1, output_new.shape[-1]), 
                                                                target_labels.reshape(-1)).item()
                                            loss_diff = prev_loss - new_loss  # Negative means rule improved loss
                                            #Test rules and generate new ones                          
                                            self.model.rule_transform.update_rule_scores(batch_input_ids.to(device), loss_diff)
                                            
                                            val_input, val_target = build_custom_validation_batch(self.tokenizer, device=device)
                                            val_dec_input, val_target = prepare_decoder_input_and_target(val_target)
                                            self.model.rule_transform.validate_and_replace_rule(self.model, val_input, val_target, val_dec_input, loss_fn)               
                                        logging.debug("Rule replaced")
                            if (batch_idx + 1) % 10 == 0:     
                                    if architecture == "Rule Transformer":
                                        logging.debug("Rule validation")
                                        self.model.rule_transform.optimized_rule_replacement(self.model, val_input, val_target, val_dec_input, loss_fn) 
                                    self.accumulated_loss = 0  # Reset loss tracking
                    del batch_input_ids, decoder_input, target_labels, output
                    gc.collect()
                    torch.cuda.empty_cache()
                    step_count += 1

                                                    
                    progress_step += 1
                    progress_value = (progress_step / total_steps) * 100
                    self.root.after(0, self.update_progress, progress_value)

                # Log epoch loss
                logging.info(f"Epoch {epoch + 1}/{self.epochs.get()} completed")
                self.root.after(0, self.update_status, f"Epoch {epoch + 1}/{self.epochs.get()} completed. Current LR")

        except Exception as e:
            logging.error(f"An error occurred during training: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during training: {str(e)}")

    def improved_collate_fn(self, batch):
        input_ids, attention_masks, labels, seq_lengths = zip(*batch)
        
        # Convert sequences to tensors if they aren't already
        input_ids = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in input_ids]
        attention_masks = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in attention_masks]
        labels = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in labels]
        
        # Find max length in batch
        max_len = seq_len
        
        # Pad sequences using torch operations
        def pad_sequence(sequences, max_len, pad_value):
            return torch.stack([
                torch.cat([
                    seq,
                    torch.full((max_len - len(seq),), pad_value, dtype=seq.dtype, device=seq.device)
                ]) if len(seq) < max_len else seq[:max_len]
                for seq in sequences
            ])
        
        # Pad all sequences
        padded_input_ids = pad_sequence(input_ids, max_len, self.tokenizer.pad_token_id)
        padded_attention_masks = pad_sequence(attention_masks, max_len, 0)
        padded_labels = pad_sequence(labels, max_len, self.tokenizer.pad_token_id)
        
        # Convert sequence lengths to tensor
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        
        return padded_input_ids, padded_attention_masks, padded_labels, seq_lengths

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

            # Ensure embeddings match tokenizer
            tokenizer_vocab_size = len(self.tokenizer)

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

    def stop_training_command(self):
        self.stop_training.set()
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
                if file.endswith('.csv'):
                    try:
                        df = pd.read_csv(file_path)
                        text_data = list
                        if 'text' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df.strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                        elif 'instruct' in df.columns and 'output' in df.columns:
                            # Handle 'instruct' and 'output' columns
                            df = df.dropna(subset=['instruct', 'output'])
                            query = df['instruct'].astype(str).tolist()
                            target = df['output'].astype(str).tolist()
                        else:
                            messagebox.showerror(
                                "Error", f"CSV file '{file}' missing 'text' or 'instruct' and 'output' columns."
                            )
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read CSV file '{file}': {str(e)}")
                elif file.endswith('.json'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if file.endswith('.jsonl'):
                                for line in f:
                                    conversation = json.loads(line.strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                            else:
                                data = json.load(f)
                                self.query_target_pairs.extend(self.extract_query_target_pairs(data)) 
                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                               
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read JSON file '{file}': {str(e)}")
                elif file.endswith('.parquet'):
                    try:
                        df = pd.read_parquet(file_path)
                        if 'text' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['text'].strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                        elif 'TEXT' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['TEXT'].strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                        elif 'messages' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['messages'].strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                        elif 'instruct' in df.columns and 'output' in df.columns:
                            # Handle 'instruct' and 'output' columns
                            df = df.dropna(subset=['instruct', 'output'])
                            query = df['instruct'].astype(str).tolist()
                            target = df['output'].astype(str).tolist()
                        else:
                            messagebox.showerror(
                                "Error", f"Parquet file '{file}' missing 'text' or 'instruct' and 'output' columns."
                            )
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read Parquet file '{file}': {str(e)}")
                
                elif file.endswith('.txt'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        text_data.append(text)
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read text file '{file}': {str(e)}")
                elif file.endswith('.pdf'):
                    try:
                        text = []
                        text = extract_text_from_pdf(file_path)
                        
                        # Break into query/target pairs
                        data = []
                        for i in range(0, len(text) - seq_len, seq_len):
                            query = text[i:i + seq_len]
                            target = text[i + 1:i + seq_len + 1]
                            self.query_target_pairs.append({query, target})
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read text file '{file}': {str(e)}")
                else:
                    messagebox.showwarning("Warning", f"Unsupported file format: '{file}'")

            if not self.query_target_pairs:
                messagebox.showerror("Error", "No valid query/target pairs found in the dataset.")
                return

            # Store text data for saving as a text file
            self.text_data = []
            for query, target in self.query_target_pairs:
                self.text_data.append(f"User: {query}\nAssistant: {target}")

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
                        query_target_pairs.append((query.strip(), target.strip()))

                    elif messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))

            elif conversation.get("conversations"):
                messages = conversation.get("conversations", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
            elif conversation.get("text"):
                messages = conversation.get("text", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
            else:
                user_messages = conversation.get("user", [])
                assistant_messages = conversation.get("assistant", [])
                for i in range(min(len(user_messages), len(assistant_messages))):
                    query = user_messages[i].replace('\n', ' ').strip()
                    target = assistant_messages[i].replace('\n', ' ').strip()
                    query_target_pairs.append((query, target))
            # Final fallback: split everything into sequence-length chunks for predictive text
            if not query_target_pairs:
                all_text = " ".join([m.get("text", "") for conversation in data for m in conversation])
                tokenized_text = tokenizer.encode(all_text, truncation=False)
                query_target_pairs = [
                    {"query": tokenized_text[i:i+seq_len], "target": tokenized_text[i:i+seq_len]}
                    for i in range(0, len(tokenized_text), seq_len)
                ]

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

    def extract_query_target_pairs_parquet(self, file_path):
        try:
            df = pd.read_parquet(file_path)
            query_target_pairs = []

            for _, row in df.iterrows():
                user_query = row.get("question") or row.get("input")
                assistant_response = row.get("answer") or row.get("response")

                if user_query and assistant_response:
                    query_target_pairs.append((user_query.strip(), assistant_response.strip()))

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
            if not (hasattr(self, 'validation_input_ids') and hasattr(self, 'validation_labels')):
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
            seq_lengths = torch.tensor(
                [min(len(tokens), seq_len) for tokens in self.validation_input_ids],
                dtype=torch.long, device=self.device
            )
            dataset = torch.utils.data.TensorDataset(input_ids, labels, seq_lengths)

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
        
        # Use the same training mode as set in the GUI (imitation, completion, or response)
        training_mode = self.training_mode.get()
        
        for query, target in self.validation_query_target_pairs:
            input_ids, labels = self._generate_training_pairs(query, target, training_mode)
            self.validation_input_ids.append(input_ids)
            self.validation_labels.append(labels)
        
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
        seq_lengths_tensor = torch.tensor(
            [min(len(ids), seq_len) for ids in self.validation_input_ids],
            dtype=torch.long, device=self.device
        )
        
        # Create a TensorDataset and DataLoader for validation
        validation_dataset = torch.utils.data.TensorDataset(input_ids_tensor, labels_tensor, seq_lengths_tensor)
        self.validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size.get(), shuffle=False, collate_fn=collate_fn)
        
        messagebox.showinfo("Success", f"Validation dataset loaded with {len(validation_dataset)} samples.")


    def run_validation(self, validation_loader, loss_fn):
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch_input_ids, batch_labels, seq_lengths in validation_loader:
                batch_input_ids = batch_input_ids.to(self.device)
                batch_labels = batch_labels.to(self.device)
                # Adjust the forward call as needed if your model requires three inputs.
                outputs, _, _ = self.model(batch_input_ids, batch_labels.reshape(-1))
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
