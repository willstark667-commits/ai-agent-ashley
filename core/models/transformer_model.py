"""
Transformer Language Model

Architecture:
  - Input: Sequence of tokens → Embeddings + Positional encoding
  - Multi-head attention: Each head learns different attention patterns
  - Feed-forward networks: Point-wise dense transformations
  - Layer normalization: Stabilize training
  - Output: Next token probability distribution

Key Innovation: Self-Attention
  - Query (Q), Key (K), Value (V) projections
  - Attention = softmax(Q * K^T / sqrt(d_k)) * V
  - Allows each token to attend to all other tokens in parallel

Strengths:
  ✓ Parallel processing (fast training)
  ✓ Captures long-range dependencies
  ✓ State-of-the-art performance (GPT, BERT, etc.)
  ✓ Scalable to billions of parameters

Weaknesses:
  ✗ High memory consumption (quadratic attention)
  ✗ Requires massive amounts of data and compute
  ✗ Position encoding is somewhat arbitrary
  ✗ Can be slow for very long sequences
"""

import numpy as np
from typing import List, Tuple, Dict
import math


class MultiHeadAttention:
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        """
        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Weight matrices for Q, K, V, and output projection
        self.W_q = np.random.randn(d_model, d_model) * math.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * math.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * math.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * math.sqrt(2.0 / d_model)
    
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, 
                                    V: np.ndarray, mask: np.ndarray = None) -> Tuple:
        """
        Scaled dot-product attention
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
        """
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(self.d_k)  # (batch, seq, seq)
        
        # Apply mask (for causal attention)
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        # Softmax
        attn_weights = self._softmax(scores, axis=-1)  # (batch, seq, seq)
        
        # Apply to values
        output = np.matmul(attn_weights, V)  # (batch, seq, d_k)
        
        return output, attn_weights
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
               mask: np.ndarray = None) -> Tuple:
        """
        Multi-head attention forward pass
        Input shapes: (seq_len, d_model)
        """
        batch_size, seq_len, _ = Q.shape
        
        # Project to Q, K, V
        Q = np.dot(Q, self.W_q)  # (seq_len, d_model)
        K = np.dot(K, self.W_k)
        V = np.dot(V, self.W_v)
        
        # Reshape for multi-head: (num_heads, seq_len, d_k)
        Q = Q.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        
        # Attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        
        # Final linear projection
        output = np.dot(attn_output, self.W_o)
        
        return output, attn_weights
    
    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class TransformerLanguageModel:
    """Transformer-based Language Model"""
    
    def __init__(self, vocab_size: int = 10000, d_model: int = 512,
                 num_heads: int = 8, num_layers: int = 6, 
                 d_ff: int = 2048, max_seq_len: int = 512):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Dimension of feed-forward networks
            max_seq_len: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.embeddings = np.random.randn(vocab_size, d_model) * math.sqrt(d_model)
        
        # Positional encoding (fixed or learned)
        self.positional_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer layers
        self.attention_layers = [MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
        
        # Feed-forward networks
        self.ff_weights_1 = [np.random.randn(d_ff, d_model) * 0.01 for _ in range(num_layers)]
        self.ff_weights_2 = [np.random.randn(d_model, d_ff) * 0.01 for _ in range(num_layers)]
        
        # Output layer
        self.W_out = np.random.randn(vocab_size, d_model) * 0.01
        self.b_out = np.zeros((vocab_size,))
        
        self.training_samples = 0
        self.loss_history = []
    
    @staticmethod
    def _create_positional_encoding(max_seq_len: int, d_model: int) -> np.ndarray:
        """
        Create sinusoidal positional encoding
        """
        pe = np.zeros((max_seq_len, d_model))
        position = np.arange(max_seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = np.cos(position * div_term[:d_model//2])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, tokens: List[int]) -> np.ndarray:
        """
        Forward pass through transformer
        Returns: (vocab_size,) logits
        """
        seq_len = len(tokens)
        
        # Embed tokens
        x = np.array([self.embeddings[t] for t in tokens])  # (seq_len, d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len]
        
        # Pass through transformer layers
        for layer_idx in range(self.num_layers):
            # Self-attention
            attn_out, _ = self.attention_layers[layer_idx].forward(x, x, x)
            x = x + attn_out  # Residual connection + layer norm (simplified)
            
            # Feed-forward
            ff_out = np.dot(x, self.ff_weights_1[layer_idx].T)  # (seq_len, d_ff)
            ff_out = np.maximum(ff_out, 0)  # ReLU
            ff_out = np.dot(ff_out, self.ff_weights_2[layer_idx].T)  # (seq_len, d_model)
            x = x + ff_out  # Residual connection + layer norm (simplified)
        
        # Use last token's representation for next token prediction
        logits = np.dot(x[-1], self.W_out) + self.b_out
        return logits
    
    def train_step(self, input_tokens: List[int], target_token: int, learning_rate: float = 0.0001):
        """
        Training step
        """
        logits = self.forward(input_tokens)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Cross-entropy loss
        loss = -np.log(probs[target_token] + 1e-10)
        self.loss_history.append(loss)
        
        self.training_samples += 1
        return loss
    
    def predict(self, tokens: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Predict next token
        """
        logits = self.forward(tokens)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        top_indices = np.argsort(probs)[-top_k:][::-1]
        return [(int(idx), float(probs[idx])) for idx in top_indices]
    
    def get_stats(self) -> Dict:
        """Get model statistics"""
        avg_loss = np.mean(self.loss_history[-100:]) if self.loss_history else 0
        
        # Calculate total parameters
        total_params = self.vocab_size * self.d_model  # embeddings
        total_params += self.d_model  # positional encoding
        
        # Transformer layers
        for _ in range(self.num_layers):
            # Multi-head attention
            total_params += 4 * self.d_model * self.d_model
            # Feed-forward
            total_params += self.d_model * self.d_ff + self.d_ff * self.d_model
        
        # Output layer
        total_params += self.vocab_size * self.d_model
        
        return {
            "model_type": "Transformer",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "total_parameters": total_params,
            "training_samples": self.training_samples,
            "average_loss": avg_loss,
            "loss_history_length": len(self.loss_history)
        }
