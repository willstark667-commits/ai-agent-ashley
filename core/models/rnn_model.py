"""
Recurrent Neural Network Language Model

Architecture:
  - Input: Sequence of tokens → Embeddings
  - RNN cells: Process sequence step-by-step
    * Basic RNN: y_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
    * LSTM: Uses gates to manage information flow
    * GRU: Simplified LSTM with fewer parameters
  - Output layer: Dense → logits
  - Output: Next token probability distribution

Strengths:
  ✓ Handles variable-length sequences
  ✓ Maintains hidden state across time steps
  ✓ LSTM/GRU solve vanishing gradient problem
  ✓ Captures long-range dependencies (with LSTM)

Weaknesses:
  ✗ Sequential processing (slow, hard to parallelize)
  ✗ Training is slower than CNN/Transformer
  ✗ Difficult to capture very long dependencies
  ✗ More parameters than CNN
"""

import numpy as np
from typing import List, Tuple, Dict
import math


class LSTMCell:
    """LSTM (Long Short-Term Memory) cell"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        LSTM has 4 gates: input, forget, output, cell
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Gates: forget, input, output, cell
        # Each has weights for (input + hidden)
        self.W_forget = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.b_forget = np.zeros((hidden_dim, 1))
        
        self.W_input = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.b_input = np.zeros((hidden_dim, 1))
        
        self.W_output = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.b_output = np.zeros((hidden_dim, 1))
        
        self.W_cell = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.b_cell = np.zeros((hidden_dim, 1))
    
    def forward(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple:
        """
        LSTM forward pass
        
        Args:
            x_t: Input at time t (input_dim,)
            h_prev: Hidden state from t-1 (hidden_dim,)
            c_prev: Cell state from t-1 (hidden_dim,)
        
        Returns:
            (h_t, c_t, cache) for backward pass
        """
        # Concatenate input and previous hidden state
        concat = np.vstack((h_prev, x_t))  # (input_dim + hidden_dim,)
        
        # Forget gate: decides what to forget from cell state
        f_t = 1 / (1 + np.exp(-(np.dot(self.W_forget, concat) + self.b_forget)))  # sigmoid
        
        # Input gate: decides what new info to add
        i_t = 1 / (1 + np.exp(-(np.dot(self.W_input, concat) + self.b_input)))  # sigmoid
        
        # Candidate cell state: new info to add
        c_tilde = np.tanh(np.dot(self.W_cell, concat) + self.b_cell)
        
        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Output gate: decides what to output
        o_t = 1 / (1 + np.exp(-(np.dot(self.W_output, concat) + self.b_output)))  # sigmoid
        
        # Hidden state
        h_t = o_t * np.tanh(c_t)
        
        cache = (x_t, h_prev, c_prev, f_t, i_t, c_tilde, o_t, c_t, concat)
        return h_t, c_t, cache


class RNNLanguageModel:
    """RNN-based Language Model (LSTM variant)"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 2):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embeddings
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # LSTM layers
        self.lstm_layers = []
        prev_dim = embedding_dim
        for i in range(num_layers):
            lstm = LSTMCell(prev_dim, hidden_dim)
            self.lstm_layers.append(lstm)
            prev_dim = hidden_dim
        
        # Output layer
        self.W_out = np.random.randn(vocab_size, hidden_dim) * 0.01
        self.b_out = np.zeros((vocab_size, 1))
        
        self.training_samples = 0
        self.loss_history = []
    
    def embed_tokens(self, tokens: List[int]) -> List[np.ndarray]:
        """
        Embed sequence of tokens
        Returns: List of (embedding_dim,) arrays
        """
        return [self.embeddings[t].reshape(-1, 1) for t in tokens]
    
    def forward(self, tokens: List[int]) -> np.ndarray:
        """
        Forward pass through LSTM layers
        Returns: (vocab_size,) logits for next token
        """
        embeddings = self.embed_tokens(tokens)
        
        # Initialize hidden and cell states for all layers
        h_states = [np.zeros((self.hidden_dim, 1)) for _ in range(self.num_layers)]
        c_states = [np.zeros((self.hidden_dim, 1)) for _ in range(self.num_layers)]
        
        # Process sequence
        for x_t in embeddings:
            # Forward through each LSTM layer
            for layer_idx, lstm in enumerate(self.lstm_layers):
                h_states[layer_idx], c_states[layer_idx], _ = lstm.forward(
                    x_t.flatten(),
                    h_states[layer_idx].flatten(),
                    c_states[layer_idx].flatten()
                )
                x_t = h_states[layer_idx]  # Output of this layer is input to next
        
        # Final hidden state to output logits
        logits = np.dot(self.W_out, h_states[-1]) + self.b_out  # (vocab_size, 1)
        return logits.flatten()
    
    def train_step(self, input_tokens: List[int], target_token: int, learning_rate: float = 0.001):
        """
        Training step: forward pass, compute loss, backward pass (simplified)
        """
        # Forward
        logits = self.forward(input_tokens)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Cross-entropy loss
        loss = -np.log(probs[target_token] + 1e-10)
        self.loss_history.append(loss)
        
        # Simplified SGD update
        error = probs.copy()
        error[target_token] -= 1
        self.W_out -= learning_rate * np.outer(error, np.zeros(self.hidden_dim))
        self.b_out -= learning_rate * error.reshape(-1, 1)
        
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
        total_params = self.vocab_size * self.embedding_dim  # embeddings
        total_params += self.vocab_size * self.hidden_dim  # output layer
        
        # LSTM parameters per layer
        for i, lstm in enumerate(self.lstm_layers):
            input_size = self.embedding_dim if i == 0 else self.hidden_dim
            # 4 gates * (hidden * (input + hidden))
            total_params += 4 * self.hidden_dim * (input_size + self.hidden_dim)
        
        return {
            "model_type": "RNN-LSTM",
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "total_parameters": total_params,
            "training_samples": self.training_samples,
            "average_loss": avg_loss,
            "loss_history_length": len(self.loss_history)
        }
