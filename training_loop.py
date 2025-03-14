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
import sys
import torch
import torch.nn as nn
from rulesbasedmodel4 import Rule_Transformer_Model  # Import the model class
import gc

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_BUFFSIZE"] = "16777216"
torch.backends.cudnn.enabled = True  # Ensures cuDNN is enabled
torch.backends.cudnn.benchmark = True  # Optimizes performance
# Allow PyTorch to use system RAM when GPU memory overflows

def safe_tensor_allocation(tensor):
    try:
        return tensor.to("cuda", non_blocking=True)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("⚠️ GPU memory full! Offloading to CPU RAM.")
            return tensor.to("cpu")
        else:
            raise e

def cleanup():
    dist.destroy_process_group()
    
def setup_ddp():
    """Setup Distributed Data Parallel (DDP) properly."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")  # Change if needed

    world_size = torch.cuda.device_count()
    rank = int(os.environ["LOCAL_RANK"])  # Use LOCAL_RANK for proper GPU assignment

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

    # Assign correct GPU based on rank
    torch.cuda.set_device(rank)

    print(f"✅ DDP Initialized: {world_size} GPUs, Rank: {rank} using GPU: {torch.cuda.current_device()}")

setup_ddp()

# Debug for CUDA
print(f"CUDA Available: {torch.cuda.is_available()}")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_LOGINFO_DBG"]= "1"
os.environ["CUBLAS_LOGDEST_DBG"] = "cublas.log"

os.environ["TORCH_USE_CUDA_DSA"] = "1"

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
                output = self.model(inputs, target_labels)          
                
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
                    output = self.model(inputs, target_labels)
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
                # Assume batch_labels and batch_labels_tot are tensors of shape [batch_size, seq_len, vocab_size]

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
        self.optimizers = [Muon(m.parameters(), lr=self.learning_rate, momentum=0.9) 
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
                # Assume batch_labels and batch_labels_tot are tensors of shape [batch_size, seq_len, vocab_size]

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
            optimizer = Muon(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

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
                # Assume batch_labels and batch_labels_tot are tensors of shape [batch_size, seq_len, vocab_size]

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

def select_or_create_tokenized_data(use_chunked_dataset, tokenized_data_path):
            if use_chunked_dataset is True:
                if tokenized_data_path:
                    messagebox.showinfo("Success", f"Tokenized data directory selected: {tokenized_data_path}")
            else:
                # User wants to use existing single tokenized data file, select a file
                if tokenized_data_path is not None:
                    # Attempt to load the file to validate its content
                    try:
                        with open(tokenized_data_path, 'r', encoding='utf-8') as f:
                            input_ids, labels, labels_tot = [], [], []
                            for line in f:
                                record = json.loads(line)
                                input_ids.append(record['input_ids'])
                                labels.append(record['labels'])
                                labels_tot.append(record['labels_tot'])
                        logging.info(f"Tokenized data file loaded successfully with {len(input_ids)} entries.")
                        return input_ids, labels, labels_tot
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to load tokenized data file: {str(e)}")


def load_tokenizer(tokenizer_path):
        try:
            if not tokenizer_path or not os.path.exists(tokenizer_path):
                raise FileNotFoundError("Tokenizer file not selected or does not exist.")

            # Load the PreTrainedTokenizerFast from file.
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
            logging.info(f"Tokenizer loaded from {tokenizer_path}")

            # If a special tokens map exists, load and add them.
            special_tokens_path = os.path.join(os.path.dirname(tokenizer_path), "special_tokens_map.json")
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
                tokenizer.add_special_tokens(special_tokens)
                logging.info(f"Special tokens added: {special_tokens}")

            # Load tokenizer configuration if available.
            tokenizer_config_path = os.path.join(os.path.dirname(tokenizer_path), "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path, "r", encoding="utf-8") as file:
                    tokenizer_config = json.load(file)
                    tokenizer.init_kwargs.update(tokenizer_config)
                    if "model_max_length" in tokenizer_config:
                        tokenizer.model_max_length = tokenizer_config["model_max_length"]
                    logging.info(f"Tokenizer configuration loaded: {tokenizer_config}")

            # Ensure a reasonable model_max_length is set.
            if not hasattr(tokenizer, "model_max_length") or tokenizer.model_max_length > 1024 * 1024:
                tokenizer.model_max_length = seq_len  # Default value; ensure seq_len is defined
            logging.info(f"Model max length set to: {tokenizer.model_max_length}")

            # Log the vocabulary size.
            tokenizer_vocab_size = len(tokenizer)
            logging.info(f"Loaded tokenizer vocabulary size: {tokenizer_vocab_size}")

            # Ensure special tokens are correctly set.
            if not tokenizer.pad_token:
                tokenizer.pad_token = "<PAD>"
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<PAD>")
                logging.warning("Pad token was not set. Defaulting to <PAD>.")
            if not tokenizer.unk_token:
                tokenizer.unk_token = "<UNK>"
                tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids("<UNK>")
                logging.warning("UNK token was not set. Defaulting to <UNK>.")
            if not tokenizer.bos_token:
                tokenizer.bos_token = "<BOS>"
                tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<BOS>")
                logging.warning("BOS token was not set. Defaulting to <BOS>.")
            if not tokenizer.eos_token:
                tokenizer.eos_token = "<EOS>"
                tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<EOS>")
                logging.warning("EOS token was not set. Defaulting to <EOS>.")

            print("Special tokens map:", tokenizer.special_tokens_map)
            print("Pad token ID:", tokenizer.pad_token_id)
            print("Model max length:", tokenizer.model_max_length)
            return tokenizer

        except Exception as e:
            logging.error(f"Failed to load tokenizer: {str(e)}")
            messagebox.showerror("Error", f"Failed to load tokenizer: {str(e)}")


def validate_training_parameters(batch_size, epochs, tokenized_data_path, tokenizer):
        # Validate batch size
        try:
            batch_size = int(batch_size)
            if batch_size <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logging.error(f"Invalid batch size: {batch_size}")
            messagebox.showerror("Error", "Batch size must be a positive integer.")
            return False

        # Validate epochs
        try:
            epochs = int(epochs)
            if epochs <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logging.error(f"Invalid epochs value: {epochs}")
            messagebox.showerror("Error", "Epochs must be a positive integer.")
            return False

        if not tokenized_data_path or not os.path.exists(tokenized_data_path):
            logging.error("Tokenized data path is invalid or does not exist.")
            messagebox.showerror("Error", "Tokenized data is not selected or does not exist.")
            return False

        if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
            logging.error("Tokenizer pad_token_id is not set.")
            messagebox.showerror("Error", "Tokenizer is missing pad_token_id.")
            return False

        return True

def update_training_progress(progress):
    """Writes progress percentage to a file for the GUI to read."""
    with open("training_progress.json", "w") as f:
        json.dump({"progress": progress}, f)
        

def training_loop(model, epochs, batch_size, genetic_algo_var, use_chunked_dataset, tokenized_data_path, tokenizer_path, seq_len, learning_rate, architecture):

        stop_training = False
        validation_loader = None

        def stop_training():
                """Stops the training subprocess."""
                stop_training= True
                messagebox.showinfo("Stop Training", "Training stopped.")
                logging.info("Training stopped by user.")
        tokenizer = load_tokenizer(tokenizer_path)
        input_ids, labels, labels_tot = select_or_create_tokenized_data(use_chunked_dataset, tokenized_data_path)

        if not validate_training_parameters(batch_size, epochs, tokenized_data_path, tokenizer):
            return

        logging.info("All training parameters and data are properly initialized.")
        if not model:
            logging.error("Model not initialized before training")
            return
        use_genetic_algo = genetic_algo_var

        try:
            
            rank = dist.get_rank()  # Get process rank
            device = torch.device(f"cuda:{rank}")  # Assign each rank a GPU
            model.to(device)  # Move model to GPU
            torch.cuda.set_per_process_memory_fraction(0.98, device=rank)
            if use_chunked_dataset:
                # Initialize the ChunkedDataset
                dataset = ChunkedDataset(
                    tokenized_data_path=tokenized_data_path,
                    tokenizer=tokenizer,
                    max_length=seq_len
                )
                if device == 'cpu':
                    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=16,
                        pin_memory = True,
                        collate_fn=collate_fn
                    )
                else:
                    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory = False,
                        collate_fn=collate_fn
                    )
                # Initialize the standard dataset and dataloader
            else:
                # Ensure the tokenizer is loaded and has a valid pad_token_id
                pad_token_id = tokenizer.pad_token_id if tokenizer else pad_token_id  # Default to global if tokenizer isn't set      
                max_length = seq_len  # Adjust as needed
                logging.info("max_length set")
                # Convert lists of token IDs to tensors and calculate original sequence lengths
                input_ids = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device)[:max_length]
                    for tokens in input_ids
                ]
                logging.info("input ids torched to tensor")

                labels = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device)[:max_length]
                    for tokens in labels
                ]
                logging.info("labels torched to tensor")

                labels_tot = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device)[:max_length]
                    for tokens in labels_tot
                ]
                logging.info("labels_tot torched to tensor")


                # Stack tensors
                input_ids = torch.stack(input_ids)
                labels = torch.stack(labels)
                labels_tot = torch.stack(labels_tot)
                logging.info("datas stacked and torched")


                dataset = torch.utils.data.TensorDataset(input_ids, labels_tot, labels)
                logging.info("dataset torched")
                if device == 'cpu':
                    dataloader = DataLoader(
                        dataset,
                        batch_size=int(batch_size),
                        shuffle=True,
                        num_workers=16,  # Set to 0 to prevent multiple workers from loading chunks simultaneously
                        pin_memory=True,
                        collate_fn=collate_fn
                    )
                else:
                    dataloader = DataLoader(
                        dataset,
                        batch_size=int(batch_size),
                        shuffle=True,
                        num_workers=0,  # Set to 0 to prevent multiple workers from loading chunks simultaneously
                        pin_memory=False,
                        collate_fn=collate_fn
                    )
                logging.info("dataloader defined")
            ##chunked vs. standard else complete

            # Adjust learning rate based on architecture
            lr = learning_rate
            logging.info(f"Learning Rate: {lr}")

            # Learning rate scheduler
            total_steps = epochs * len(dataloader)
            logging.info(f"Total training steps: {total_steps}")
            # Separate parameters based on their shape.

            # Create two optimizers:
            #Enable for standard optimizer/scheduler
            #num_warmup_steps = total_steps // 10  # Warmup for 10% of training
            #scheduler = self.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)

            #optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

            # 🔹 Find ≥2D parameters for Muon
            muon_params = [
                p for name, p in model.named_parameters()
                if p.ndim >= 2 and ("fc_out" in name or "encoder_layers" in name)
            ]

            adamw_params = [
                p for name, p in model.named_parameters()
                if p.ndim < 2 or "rule_scores" in name or "embedding" in name or "bias" in name
            ]


            # 🔹 Create optimizers

            optimizers = [
                Muon(muon_params, lr=0.02, momentum=0.95),  
                torch.optim.AdamW(adamw_params, lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)
            ]

            logging.info("Optimizers defined")

            #loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01, ignore_index=pad_token_id)
            loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

            model.train()
            logging.info("Model set to training mode")
            progress_step = 0
            n = 0
            accumulation_steps = 4
            

            for epoch in range(epochs):
                if stop_training == True:
                    logging.info("Training stopped by user.")
                    messagebox.showinfo("Info", "Training stopped by user.")
                    break
                
                for opt in optimizers:
                    opt.zero_grad()
                    logging.debug("Optimizer gradients zeroed")
                accumulated_loss = 0
                epoch_loss = 0
                logging.info(f"Epoch {epoch+1} started")
                torch.cuda.empty_cache()

                # Training loop
                for batch_idx, (batch_input_ids, batch_labels) in enumerate(dataloader):
                    if stop_training == True:
                            logging.info("Training stopped by user.")
                            messagebox.showinfo("Info", "Training stopped by user.")
                            return

                        # Move batches and targets to the correct device 
                    batch_input_ids = batch_input_ids.to(device)
                    batch_labels = batch_labels.to(device)

                        # Logging epoch and batch info
                    logging.debug(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}')
                    logging.debug(f'Batch input_ids shape: {batch_input_ids.shape}')  # (batch_size, 1024)
                    logging.debug(f'Using device: {device}')


                     # Prepare inputs and targets for teacher forcing
                    decoder_input, target_labels = prepare_decoder_input_and_target(batch_labels)
                
                        # Log the shape of the combined mask
                    logging.debug(f'Decoder input shape: {decoder_input.shape}')  # (batch_size, 1024)
                    logging.debug(f'Target labels shape: {target_labels.shape}')  # (batch_size, 1024)
                    architecture = architecture


                        # Check the flag and run evolution once per epoch if requested:
                    if use_genetic_algo == "Genetic Algorithm":
                            logging.info("Applying genetic algorithm evolution step...")
                            qga = GeneticAlgorithm(model, lr)
                            # Evolve using the same loss function and dataloader (or a validation subset)
                            model = qga.evolve(loss_fn, batch_input_ids, target_labels, decoder_input, architecture)
                            #Remove optimizer steps and gradient code enable this for Quaternion NeuroEvolution of Augmenting Topologies (NEAT)
                    elif use_genetic_algo == "NEAT":
                            neat = NEAT(model)
                            model = neat.evolve(loss_fn, batch_input_ids, target_labels, decoder_input, architecture)
                    elif use_genetic_algo == "Firefly":
                            #Remove optimizer steps and gradient lines to enable this for Quaternion Firefly Algo
                            firefly_optimizer = FireflyOptimizer(model)
                            model = firefly_optimizer.optimize(loss_fn, batch_input_ids, target_labels, decoder_input, architecture)
                    elif use_genetic_algo == "NF Hybrid":
                        # Initialize the Hybrid NEAT-Firefly Optimizer
                        hybrid_optimizer = HybridNEATFireflyOptimizer(
                            model=model,
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
                        model = hybrid_optimizer.optimize(loss_fn, batch_input_ids, target_labels, decoder_input, architecture)

                        # Calculate fitness and log progress
                        fitness_scores = hybrid_optimizer.fitness_history[-1]
                        avg_fitness = sum(fitness_scores) / len(fitness_scores)
                        print(f"Average Fitness: {avg_fitness:.4f}")
                    else:
                        # Forward pass
                        try:
                            if architecture == "Rule Transformer":

                                output = model(batch_input_ids, decoder_input)

                            else:
                                output = model(batch_input_ids, decoder_input)
                        except Exception as e:
                                raise ValueError(f"forward pass failed for {str(e)}")

                        logging.debug(f"Shape of outputs: {output.shape}")
                            # Assume batch_labels and batch_labels_tot are tensors of shape [batch_size, seq_len, vocab_size]
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


                            # Backward pass and optimization
                        loss.backward(retain_graph=True)
                        logging.info("Loss backward computed")
                            
                            # Check for NaN or Inf in gradients
                        for name, param in model.named_parameters():
                                if param.grad is not None:
                                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                        logging.error(f"Gradient for {name} contains NaN or Inf.")
                                        continue
                                    
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                    logging.debug(f"Gradient for {name}: mean={param.grad.mean().item():.4f}, max={param.grad.max().item():.4f}, min={param.grad.min().item():.4f}")
                            else:
                                    logging.debug(f"Gradient for {name} is None")

                        total_norm = 0.0
                        for p in model.parameters():
                                if p.grad is not None:
                                    total_norm += p.grad.data.norm(2).item() ** 2
                        total_norm = total_norm ** 0.5
                        logging.info(f"Gradient norm: {total_norm}")

                            ###Uncomment these for gradient clipping
                            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            

                            # Log gradients for debugging
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                logging.debug(f"Gradients for {name}: {param.grad}")
                            else:
                                logging.debug(f"No gradients found for {name}.")
                        
                            
                        n+=1
                        print(f"Iteration {n}, Loss: {loss.item()}")
        
                            # Before optimizer step
                        for name, param in model.named_parameters():
                                if param.requires_grad:
                                    logging.debug(f"Before step - {name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")
                        if (batch_idx + 1) % accumulation_steps == 0:     
                                avg_loss = accumulated_loss / accumulation_steps  # Compute avg loss across accumulated steps
                                for opt in optimizers:
                                    opt.step()
                                    logging.info("Optimizer step update completed")
                                torch.cuda.empty_cache()

                                # After optimizer step
                                for name, param in model.named_parameters():
                                    if param.requires_grad:
                                        logging.debug(f"After step - {name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")
                                for opt in optimizers:
                                    opt.zero_grad()
                                    logging.debug("Optimizer gradients zeroed")
                                if architecture == "Rule Transformer":
                                    if hasattr(model, 'embedding') and isinstance(model.embedding, DynamicRuleEmbedding):
                                        model.embedding.update_rule_scores(batch_input_ids, avg_loss)
                                        model.embedding.add_new_rule()  # Dynamically add/replace rules
                                        num_saved_rules = model.embedding.current_rule_count or 1

                                accumulated_loss = 0  # Reset loss tracking

                                                    
                    progress_step += 1
                    progress_value = (progress_step / total_steps) * 100
                    # ✅ Write to shared file
                    update_training_progress(progress_value)
                    # Save checkpoint at specified intervals
                    save_interval = 1  # Save every 1%
                    progress_percentage = (batch_idx + 1) / len(dataloader) * 100
                    if abs(progress_percentage % save_interval) < 1e-6:  # Avoid floating-point issues
                        torch.save(model.state_dict(), "trained_model.pth")  # Save trained model
                        logging.info(f"Checkpoint saved at epoch {epoch}, batch {batch_idx}, progress: {progress_percentage:.2f}%")
                    
                    # perform validation after specified progress steps
                    validation_interval = 5  # Save every 25%
                    progress_percentage = (batch_idx + 1) / len(dataloader) * 100
                    if abs(progress_percentage % validation_interval) < 1e-6:  # Avoid floating-point issues

                        if validation_loader is not None:  
                            val_loss = run_validation(validation_loader, loss_fn)
                            print(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")
                            logging.info(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")

                # Log epoch loss
                logging.info(f"Epoch {epoch + 1}/{epochs} completed")
                progress_value = ((epoch+1) / epochs)
                # ✅ Write to shared file
                update_training_progress(progress_value)
        except Exception as e:
            logging.error(f"An error occurred during training: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during training: {str(e)}")
            

def run_validation(validation_loader, loss_fn):
        model.eval()
        total_val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch_input_ids, batch_labels, batch_labels_tot, seq_lengths in validation_loader:
                batch_input_ids = batch_input_ids.to(device)
                batch_labels = batch_labels.to(device)
                # Adjust the forward call as needed if your model requires three inputs.
                outputs, _, _ = model(batch_input_ids, batch_labels.reshape(-1), batch_labels_tot.reshape(-1))
                # Flatten outputs and targets for loss calculation
                logits = outputs.reshape(-1, outputs.size(-1))
                targets = batch_labels.reshape(-1)
                loss = loss_fn(logits, targets)
                total_val_loss += loss.item()
                num_batches += 1
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")
        model.train()
        return avg_val_loss


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: Model file path not provided.")
        sys.exit(1)

    model_path = sys.argv[1]  # Model state file
    config_path = sys.argv[2]  # Model config file

    # ✅ Load model parameters from JSON
    with open(config_path, "r") as f:
        model_config = json.load(f)
    seq_len = model_config["seq_length"]
    num_heads=model_config["num_heads"]
    seq_length=model_config["seq_length"]
    num_layers = model_config["num_layers"]
    genetic_algo_var = model_config["genetic_algo_var"]
    use_chunked_dataset = model_config["use_chunked_dataset"]
    tokenized_data_path = model_config["tokenized_data_path"]
    tokenizer_path = model_config["tokenizer_path"]
    batch_size = model_config["batch_size"]
    epochs = model_config["epochs"]
    learning_rate = model_config["learning_rate"]
    num_saved_rules = model_config["num_saved_rules"]
    architecture = model_config["architecture"]
    num_parameters = model_config["num_parameters"]
    hidden_size = model_config["hidden_size"]
    log_file_path = model_config["log_file_path"]

    if log_file_path:
        print(f"Log file will be saved to: {log_file_path}")
    else:
        log_file_path = 'training_debug.log'  # Default log file
        print(f"No log file selected. Using default: {log_file_path}")

            # Setup logging
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s')
    
    # ✅ Free up GPU memory before loading model
    torch.cuda.empty_cache()
    gc.collect()
    # ✅ Dynamically initialize the model
    if architecture == "Reasoning Model":
        model = Transformer_Model(
            vocab_size=model_config["vocab_size"],
            embedding_dim = model_config["embed_size"],
            hidden_dim = model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            num_heads = model_config["num_heads"],
            seq_length=model_config["seq_length"],
            ).cuda()
    elif architecture == "Rule Transformer":
        model = Rule_Transformer_Model(
            vocab_size=model_config["vocab_size"],
            embedding_dim= model_config["embed_size"],
            num_layers=model_config["num_layers"],
            num_heads = model_config["num_heads"],
            max_rules = model_config["num_saved_rules"],
            seq_length=model_config["seq_length"],
        ).cuda()

    # ✅ Load the model's saved state
    rank = dist.get_rank()  # Get process rank
    device = torch.device(f"cuda:{rank}")  # Assign each rank a GPU
    model.load_state_dict(torch.load("saved_model.pth", map_location=f"cuda:{rank}"))  # Ensure model is loaded correctly

    # Wrap with DDP after loading state
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.train()  # Set model to training mode
    # ✅ Call training function!
    training_loop(model, epochs, batch_size, genetic_algo_var, use_chunked_dataset, tokenized_data_path, tokenizer_path, seq_len, learning_rate, architecture)
    cleanup()  # Cleanup after training
    print("Training completed successfully.")
    sys.exit(0)