"""
Evolving Neural Network Language Model

Architecture:
  - Starts simple, grows over time
  - Adds layers/parameters based on learning progress
  - Uses genetic algorithms or architecture search
  - Adapts to data characteristics

Evolution Process:
  1. Start with small network
  2. Evaluate performance
  3. Mutate architecture (add layer, more parameters, etc.)
  4. Keep improvements, discard failures
  5. Repeat until convergence

Strengths:
  ✓ Automatically finds optimal architecture
  ✓ Adapts to problem complexity
  ✓ Can find novel architectures
  ✓ Efficient: grows only as needed

Weaknesses:
  ✗ Slow architecture search
  ✗ Difficult to reproduce
  ✗ Unstable training
  ✗ Hard to understand evolved architectures
"""

import numpy as np
from typing import List, Tuple, Dict
import random


class ENN(EvolvingNeuralNetwork):
    """
    Evolving Neural Network that grows and adapts its architecture
    """
    
    def __init__(self, vocab_size: int = 10000, initial_hidden_dim: int = 64):
        """
        Args:
            vocab_size: Size of vocabulary
            initial_hidden_dim: Starting hidden dimension
        """
        self.vocab_size = vocab_size
        self.hidden_dims = [initial_hidden_dim]  # Layer dimensions
        self.current_depth = 1
        self.max_depth = 8
        
        # Initialize simple network
        self.embeddings = np.random.randn(vocab_size, initial_hidden_dim) * 0.01
        self.layers = self._create_layers()
        
        self.generation = 0
        self.fitness_history = []
        self.architecture_history = []
        self.training_samples = 0
    
    def _create_layers(self) -> List:
        """
        Create network layers based on hidden_dims
        """
        layers = []
        for i in range(len(self.hidden_dims) - 1):
            W = np.random.randn(self.hidden_dims[i+1], self.hidden_dims[i]) * 0.01
            b = np.zeros((self.hidden_dims[i+1], 1))
            layers.append({'W': W, 'b': b})
        
        # Output layer
        W_out = np.random.randn(self.vocab_size, self.hidden_dims[-1]) * 0.01
        b_out = np.zeros((self.vocab_size, 1))
        layers.append({'W': W_out, 'b': b_out})
        
        return layers
    
    def forward(self, tokens: List[int]) -> np.ndarray:
        """
        Forward pass through network
        """
        # Embed
        x = np.mean([self.embeddings[t] for t in tokens], axis=0)  # Average pooling
        x = x.reshape(-1, 1)
        
        # Forward through layers
        for i, layer in enumerate(self.layers[:-1]):
            x = np.dot(layer['W'], x) + layer['b']
            x = np.maximum(x, 0)  # ReLU
        
        # Output
        logits = np.dot(self.layers[-1]['W'], x) + self.layers[-1]['b']
        return logits.flatten()
    
    def evaluate_fitness(self, test_tokens: List[List[int]], 
                        test_targets: List[int]) -> float:
        """
        Evaluate network fitness (accuracy on test set)
        """
        correct = 0
        for tokens, target in zip(test_tokens, test_targets):
            logits = self.forward(tokens)
            pred = np.argmax(logits)
            if pred == target:
                correct += 1
        
        fitness = correct / len(test_tokens) if test_tokens else 0
        return fitness
    
    def mutate_architecture(self):
        """
        Mutate network architecture: add layer, increase parameters, etc.
        """
        mutation_type = random.choice(['add_layer', 'increase_dim', 'remove_layer'])
        
        old_arch = self.hidden_dims.copy()
        
        if mutation_type == 'add_layer' and len(self.hidden_dims) < self.max_depth:
            # Add new layer
            new_dim = random.randint(self.hidden_dims[-1] // 2, self.hidden_dims[-1] * 2)
            self.hidden_dims.append(new_dim)
        
        elif mutation_type == 'increase_dim':
            # Increase dimensions of random layer
            idx = random.randint(0, len(self.hidden_dims) - 1)
            increase = random.randint(-self.hidden_dims[idx] // 2, self.hidden_dims[idx])
            self.hidden_dims[idx] = max(1, self.hidden_dims[idx] + increase)
        
        elif mutation_type == 'remove_layer' and len(self.hidden_dims) > 2:
            # Remove a layer
            idx = random.randint(1, len(self.hidden_dims) - 2)
            del self.hidden_dims[idx]
        
        # Recreate layers
        self.layers = self._create_layers()
        
        self.architecture_history.append({
            'generation': self.generation,
            'mutation': mutation_type,
            'old_arch': old_arch,
            'new_arch': self.hidden_dims.copy()
        })
    
    def evolve(self, test_tokens: List[List[int]], test_targets: List[int],
              generations: int = 10):
        """
        Evolve architecture over multiple generations
        """
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate current fitness
            fitness = self.evaluate_fitness(test_tokens, test_targets)
            self.fitness_history.append(fitness)
            
            # Mutate architecture
            self.mutate_architecture()
            
            # Evaluate new architecture
            new_fitness = self.evaluate_fitness(test_tokens, test_targets)
            
            # Keep mutation if improved, else revert (elitism)
            if new_fitness <= fitness:
                self.layers = self._create_layers()  # Revert
    
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
        # Calculate total parameters
        total_params = self.vocab_size * self.hidden_dims[0]  # embeddings
        for layer in self.layers:
            total_params += layer['W'].size + layer['b'].size
        
        return {
            "model_type": "ENN-EvolvingNeuralNetwork",
            "vocab_size": self.vocab_size,
            "architecture": self.hidden_dims,
            "current_depth": len(self.hidden_dims),
            "total_parameters": total_params,
            "generation": self.generation,
            "best_fitness": max(self.fitness_history) if self.fitness_history else 0,
            "fitness_history": self.fitness_history[-10:],  # Last 10 generations
            "mutations_applied": len(self.architecture_history)
        }
