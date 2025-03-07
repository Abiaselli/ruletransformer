import torch
import torch.nn as nn
import torch.fft
import logging
import math
import argparse
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import sys
from transformers import PreTrainedTokenizerFast
import re
import torch.utils.checkpoint as checkpoint
import random

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
########################################
# 1. Build a Byte-Level Tokenizer/Vocab
########################################
seq_len = 500

tokenizer = PreTrainedTokenizerFast(tokenizer_file=r"C:\Users\abias\.cursor-tutor\vccdoe\mhlamodel\mhlatest\tokenizer_from_dataset_bytelevel3.json")
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<EOS>")
tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<BOS>")
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<PAD>")

########################################
# 2. Data Extraction
########################################

def extract_query_target_pairs( data):
    query_target_pairs = []
    with open(data, 'r', encoding='utf-8') as f:

        data = json.load(f)

        for conversation in data:
            if conversation.get("conversations"):
                messages = conversation.get("conversations", [])
                for i in range(len(messages) - 1):
                            if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                                query = messages[i].get("value", "")
                                print(query)
                                target = messages[i + 1].get("value", "")
                                tot_target = extract_thinking(messages[i + 1])
                                query_target_pairs.append((query.strip(), target.strip(), tot_target))
                            elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                                query = messages[i].get("value", "")
                                target = messages[i + 1].get("value", "")
                                tot_target = extract_thinking(messages[i + 1])
                                query_target_pairs.append((query.strip(), target.strip(), tot_target))
        return query_target_pairs

def extract_thinking(assistant_response):
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


def extract_data(data):
    query_target_pairs = []

    query_target_pairs.extend(extract_query_target_pairs(data)) 
    for i in range(min(5, len(query_target_pairs))):
        query, target, tot_target = query_target_pairs[i]
    tot_targets_list = []
    targets_list = []
    input_ids_list = []
    
    for query, target, tot_target in query_target_pairs:
        input_ids, labels, labels_tot = _generate_training_pairs(query, target, tot_target)

        if input_ids and labels and labels_tot:        
            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            tot_targets_list.append(torch.tensor(labels_tot, dtype=torch.long))
            targets_list.append(torch.tensor(labels, dtype=torch.long))
    return input_ids_list, tot_targets_list, targets_list

def _generate_training_pairs(query, target, tot_target):
        # Debugging logs

        # Ensure inputs are valid strings before tokenization
        query_ids = tokenizer.encode(str(query) if query else "", truncation=True, max_length=seq_len)
        target_ids = tokenizer.encode(str(target) if target else "", truncation=True, max_length=seq_len)
        tot_target_ids = tokenizer.encode(str(tot_target) if tot_target else "", truncation=True, max_length=seq_len)


        input_ids = [tokenizer.bos_token_id] +  query_ids + [tokenizer.eos_token_id]
        labels = [tokenizer.bos_token_id] + target_ids + [tokenizer.eos_token_id]
        labels_tot = [tokenizer.bos_token_id] + tot_target_ids + [tokenizer.eos_token_id]

        return input_ids, labels, labels_tot    

########################################
# 3. Dataset and Collate Function
########################################

class ChatDataset(Dataset):
    def __init__(self, json_data):
        self.input_ids, self.tot_targets, self.targets = extract_data(json_data)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.tot_targets[idx], self.targets[idx]

def collate_fn(batch, fixed_length):
    """
    For each sample, combine tot_targets and targets (in that order) to form the full target.
    Then pad or truncate both the input_ids and combined targets to fixed_length.
    """
    padded_inputs = []
    padded_targets = []
    
    for input_ids, tot, tgt in batch:
        combined_target = torch.cat((tot, tgt), dim=0)
        # Truncate or pad the combined target.
        if combined_target.size(0) > fixed_length:
            combined_target = combined_target[:fixed_length]
        else:
            pad_len = fixed_length - combined_target.size(0)
            combined_target = torch.cat(
                (combined_target, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)),
                dim=0
            )

        padded_targets.append(combined_target)
        
        # For input_ids: also truncate or pad.
        if input_ids.size(0) > fixed_length:
            padded_input = input_ids[:fixed_length]
        else:
            pad_len = fixed_length - input_ids.size(0)
            padded_input = torch.cat(
                (input_ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)),
                dim=0
            )

        padded_inputs.append(padded_input)
    
    padded_inputs = torch.stack(padded_inputs)
    padded_targets = torch.stack(padded_targets)
    return padded_inputs, padded_targets

########################################
# 4. Transformer Model Definition
########################################


##############################################
# Positional Encoding (Standard Sin/Cos Version)
##############################################
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
########################################
#Base Model
########################################


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
    def __init__(self, d_model, num_heads, d_latent, lambda_decay=0.01, memory_size=5):
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
    def __init__(self, vocab_size, embedding_dim, num_frequencies=50, device=device):
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
    def __init__(self, vocab_size, embedding_dim, num_rules=100, device="cpu"):
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
    def __init__(self, vocab_size, embedding_dim, num_frequencies=50, max_rules=100, device=device):
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

            print(f"🆕 Added a new rule! Total rules: {self.current_rule_count}")

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

            print(f"🔄 Replaced rule at index {worst_rule_idx}, now tracking {self.current_rule_count} rules")


class Transformer_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, seq_length, lambda_decay=0.01, memory_size=5, dropout=0.1, compression_factor=2, num_frequencies=50, max_rules=100):
        super(Transformer_Model, self).__init__()
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

########################################
# 5. Training Loop
########################################


def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch_idx, (src, target) in enumerate(dataloader):
        src = src.to(device)
        target = target.to(device)

        decoder_input = target[:, :-1]  # Remove last token from target to match output shape
        target_labels = target[:, 1:]  # Shift target labels by one position

        optimizer.zero_grad()
        
        # 🔹 Get predictions & rule-modified embeddings
        output = model(src, decoder_input)

        # 🔹 Ensure `output` and `target_labels` have the same sequence length
        seq_len = min(output.shape[1], target_labels.shape[1])  # Get the shorter sequence length
        output = output[:, :seq_len, :]  # Truncate logits if too long
        target_labels = target_labels[:, :seq_len]  # Truncate targets if too long

        # 🔹 Flatten for cross_entropy()
        loss = criterion(output.reshape(-1, output.shape[-1]), target_labels.reshape(-1))
        loss.backward()

        # 🔹 Track how rules affected loss
        prev_loss = loss.item()
        optimizer.step()

        # 🔹 After updating, re-run forward to see new loss
        with torch.no_grad():
            output_new = model(src, decoder_input)
            new_loss = criterion(output_new[:, :seq_len, :].reshape(-1, output_new.shape[-1]), 
                                 target_labels.reshape(-1)).item()
            loss_diff = prev_loss - new_loss  # Negative means rule improved loss

            # 🔹 Update rule effectiveness
            model.embedding.update_rule_scores(src, loss_diff)

        # 🔹 Occasionally add a new rule
        if batch_idx % 50 == 0:  # Every 50 batches, check for new rules
            model.embedding.add_new_rule()

        total_loss += loss.item()
    
    return total_loss / len(dataloader)


########################################
#6. inference
########################################


# Inference function for autoregressive decoding.
def inference(model, input_text, max_seq_length, device, max_generated=30):
                    model.eval()
                    with torch.no_grad():
                        # Tokenize the prompt and move to the correct device.
                        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
                        # Pad input_ids to the maximum sequence length
                        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                        generated_text = input_ids
                        generated = []
                        logging.debug(f"Padded input_ids Shape: {input_ids.shape}")

                        # Choose a start token for the dummy target.
                        # Here we use tokenizer.eos_token_id if available; otherwise, fallback to tokenizer.pad_token_id.
                        bos_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
                        eos_token = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
                        eos_token  = torch.tensor([[eos_token]], device=device)

                        tgt_ids = torch.tensor([[bos_token]], device=device)
                        tgt_ids = torch.cat([tgt_ids, input_ids], dim=1)
                        logging.info(f"tgt_ids: {tgt_ids}")

                        # Keep track of the original input length
                        input_length = input_ids.size(1)

                        for _ in range(seq_len - input_ids.size(1)):
                            # Generate the target mask for the current target sequence length.
                            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.size(1)).to(device)
                            # Forward pass through the model
                            outputs = model(input_ids, tgt_ids)
                            logging.debug(f"output shape: {outputs.shape}")

                            # Get logits for the last token and apply argmax to get the next token ID
                            next_token_logits = outputs[:, -1, :]  # Get the logits for the last position
                            repetition_penalty = 1.2  # Adjust for stronger penalty
                            # Apply repetition penalty while excluding special tokens like PAD (0)
                            for token in set(generated_text[0].tolist()):
                                if token not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
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
                            input_ids = input_ids[input_ids != tokenizer.pad_token_id].unsqueeze(0)
                            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                            logging.debug(f"input_ids: {input_ids}")
                            generated.append(tokenizer.decode(next_token_id[0].tolist()))
                            logging.debug(f"generated_text: {generated_text}")
                            #print(tgt_ids)
                            # Stop generation if eos_token is generated
                            if next_token_id.item() == eos_token or tgt_ids.size(1) >= max_seq_length:
                                break

                    return generated


########################################
# 7. Main Function
########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=r"C:\Users\abias\.cursor-tutor\vccdoe\reasoningmodel\test\skyt1sample.json", help='Path to JSON data')
    parser.add_argument('--epochs', type=int, default=105, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--max_seq_length', type=int, default=200, help='Fixed maximum sequence length')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # ***** NEW: Load tokenizer from file instead of building from the data *****

    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    dataset = ChatDataset(args.data)
    # Use a lambda to pass the fixed length to collate_fn.
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch, args.max_seq_length))
    
    embed_size = 500
    num_heads = 10
    num_layers = 16
    seq_length = args.max_seq_length
    # Initialize the integrated model with desired module toggles.
    model = Transformer_Model(vocab_size, embed_size, num_layers, num_heads, seq_length=args.max_seq_length).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    for epoch in range(1, args.epochs + 1):
        #avg_loss = train_model(model, dataloader, optimizer, criterion, device)
        avg_loss = train_model(model, dataloader, optimizer, criterion, device)

        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Set the model to evaluation mode and perform inference.
    prompt = "What is the critical temperature of a superconducting thin film made of lead with a thickness of 100 nm?"
    generated_text = inference(model, prompt, seq_length, device)
    print("Generated text:")
    print(generated_text)

if __name__ == '__main__':
    main()
