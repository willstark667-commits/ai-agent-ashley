"""
Convolutional Neural Network Language Model

Architecture:
  - Input: Sequence of tokens → Embeddings
  - Conv1D filters: Extract local n-gram patterns
  - Pooling: Aggregate features
  - Dense layers: Classification/generation
  - Output: Next token probability distribution

Strengths:
  ✓ Fast parallel processing
  ✓ Good for text classification
  ✓ Low memory footprint
  ✓ Captures local patterns well

Weaknesses:
  ✗ Limited long-range dependencies (receptive field)
  ✗ Position-aware features need explicit handling
  ✗ Stacking too many layers requires dilated convolutions
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import math


class CNNLanguageModel:
    """CNN-based Language Model for text classification and short-range prediction"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128,
                 num_filters: int = 100, filter_sizes: List[int] = None,
                 dropout: float = 0.5):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            num_filters: Number of filters per size
            filter_sizes: Kernel sizes for convolutions (e.g., [3, 4, 5] = 3,4,5-grams)
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes or [3, 4, 5]
        self.dropout = dropout
        
        # Embedding matrix: vocab_size x embedding_dim
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Conv filters: {filter_size: (num_filters, embedding_dim)}
        self.conv_filters = {}
        for size in self.filter_sizes:
            self.conv_filters[size] = np.random.randn(num_filters, size, embedding_dim) * 0.01
        
        # Dense layers after pooling
        total_filters = num_filters * len(self.filter_sizes)
        self.dense_w = np.random.randn(total_filters, vocab_size) * 0.01
        self.dense_b = np.zeros((vocab_size,))
        
        self.training_samples = 0
        self.loss_history = []
    
    def embed_tokens(self, tokens: List[int]) -> np.ndarray:
        """
        Embed sequence of tokens
        Returns: (seq_len, embedding_dim)
        """
        return np.array([self.embeddings[t] for t in tokens])
    
    def apply_convolution(self, embeddings: np.ndarray, filter_size: int) -> np.ndarray:
        """
        Apply 1D convolution with filters of given size
        
        Args:
            embeddings: (seq_len, embedding_dim)
            filter_size: Kernel size
        
        Returns:
            (seq_len - filter_size + 1, num_filters) after ReLU
        """
        seq_len, emb_dim = embeddings.shape
        filters = self.conv_filters[filter_size]  # (num_filters, filter_size, emb_dim)
        
        output = []
        for i in range(seq_len - filter_size + 1):
            window = embeddings[i:i+filter_size]  # (filter_size, emb_dim)
            
            # Convolve with each filter
            conv_out = []
            for f in range(self.num_filters):
                # Dot product: (filter_size, emb_dim) • (filter_size, emb_dim)
                val = np.sum(filters[f] * window) + 0.0  # bias
                conv_out.append(max(0, val))  # ReLU
            
            output.append(conv_out)
        
        return np.array(output)
    
    def max_pool(self, conv_output: np.ndarray) -> np.ndarray:
        """
        Max pooling over time dimension
        (seq_len, num_filters) → (num_filters,)
        """
        if len(conv_output) == 0:
            return np.zeros(self.num_filters)
        return np.max(conv_output, axis=0)
    
    def forward(self, tokens: List[int]) -> np.ndarray:
        """
        Forward pass: tokens → convolutions → pooling → dense → logits
        Returns: (vocab_size,) logits
        """
        # Embed
        embeddings = self.embed_tokens(tokens)
        
        # Multi-size convolution + pooling
        pooled_features = []
        for filter_size in self.filter_sizes:
            conv_out = self.apply_convolution(embeddings, filter_size)
            pooled = self.max_pool(conv_out)
            pooled_features.append(pooled)
        
        # Concatenate pooled outputs
        combined = np.concatenate(pooled_features)  # (total_filters,)
        
        # Dense layer
        logits = np.dot(combined, self.dense_w) + self.dense_b  # (vocab_size,)
        
        return logits
    
    def train_step(self, input_tokens: List[int], target_token: int, learning_rate: float = 0.001):
        """
        Training step: forward pass, compute loss, backward pass
        """
        # Forward
        logits = self.forward(input_tokens)
        
        # Softmax probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)
        
        # Cross-entropy loss
        loss = -np.log(probs[target_token] + 1e-10)
        self.loss_history.append(loss)
        
        # Simple SGD update on dense layer (full backprop would be complex)
        error = probs.copy()
        error[target_token] -= 1
        self.dense_w -= learning_rate * np.outer(np.concatenate([np.zeros(self.num_filters * i) for i in range(1)]), error)
        self.dense_b -= learning_rate * error
        
        self.training_samples += 1
        return loss
    
    def predict(self, tokens: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Predict next token given sequence
        Returns: [(token_id, probability), ...]
        """
        logits = self.forward(tokens)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        top_indices = np.argsort(probs)[-top_k:][::-1]
        return [(int(idx), float(probs[idx])) for idx in top_indices]
    
    def get_stats(self) -> Dict:
        """Get model statistics"""
        avg_loss = np.mean(self.loss_history[-100:]) if self.loss_history else 0
        return {
            "model_type": "CNN",
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_filters": self.num_filters,
            "filter_sizes": self.filter_sizes,
            "total_parameters": (
                self.vocab_size * self.embedding_dim +  # embeddings
                sum(self.num_filters * size * self.embedding_dim for size in self.filter_sizes) +  # conv
                self.num_filters * len(self.filter_sizes) * self.vocab_size  # dense
            ),
            "training_samples": self.training_samples,
            "average_loss": avg_loss,
            "loss_history_length": len(self.loss_history)
        }
