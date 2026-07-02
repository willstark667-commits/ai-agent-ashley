"""
Hybrid Language Model (Multi-Model Ensemble)

Architecture:
  - Runs multiple model types in parallel
  - Routes input to appropriate models based on characteristics
  - Combines predictions using weighted ensemble
  - Adaptive routing: learns which model to trust for different tasks

Model Mix:
  - N-gram: Fast, low-resource, good for short-range prediction
  - CNN: Good for classification, local patterns
  - RNN: Good for sequences with dependencies
  - Transformer: Best overall, but expensive

Routing Strategy:
  1. Compute confidence scores from each model
  2. Weight predictions by confidence
  3. Learn weights over time
  4. Fallback to simpler model if complex model fails

Strengths:
  ✓ Combines strengths of multiple architectures
  ✓ Adaptive: learns which model works best
  ✓ Robust: fallback if one model fails
  ✓ Can balance quality vs. speed

Weaknesses:
  ✗ Requires maintaining multiple models
  ✗ Inference slower than single model
  ✗ Complex to train and tune
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class HybridLanguageModel:
    """Ensemble of multiple language models with adaptive routing"""
    
    def __init__(self, vocab_size: int = 10000, model_configs: Optional[Dict] = None):
        """
        Args:
            vocab_size: Size of vocabulary
            model_configs: Configuration for each model type
        """
        self.vocab_size = vocab_size
        self.models = {}
        self.model_weights = {}  # Learned weights for ensemble
        self.routing_scores = {}  # Performance scores per model
        self.config = model_configs or {}
        
        self._initialize_models(vocab_size)
    
    def _initialize_models(self, vocab_size: int):
        """
        Initialize all model types
        In production, lazy-load these
        """
        # Import model classes (lazy loading in real system)
        try:
            from .ngram_model import NGramModel
            self.models['ngram'] = NGramModel(n=3, min_freq=2)
            self.model_weights['ngram'] = 0.2
            self.routing_scores['ngram'] = {'prediction': 0.8, 'generation': 0.5}
        except:
            pass
        
        try:
            from .cnn_model import CNNLanguageModel
            self.models['cnn'] = CNNLanguageModel(
                vocab_size=vocab_size,
                embedding_dim=128,
                num_filters=100
            )
            self.model_weights['cnn'] = 0.25
            self.routing_scores['cnn'] = {'classification': 0.9, 'prediction': 0.7}
        except:
            pass
        
        try:
            from .rnn_model import RNNLanguageModel
            self.models['rnn'] = RNNLanguageModel(
                vocab_size=vocab_size,
                embedding_dim=128,
                hidden_dim=256
            )
            self.model_weights['rnn'] = 0.25
            self.routing_scores['rnn'] = {'generation': 0.85, 'sequence': 0.9}
        except:
            pass
        
        try:
            from .transformer_model import TransformerLanguageModel
            self.models['transformer'] = TransformerLanguageModel(
                vocab_size=vocab_size,
                d_model=256,
                num_heads=8,
                num_layers=3
            )
            self.model_weights['transformer'] = 0.3
            self.routing_scores['transformer'] = {'generation': 0.95, 'prediction': 0.9}
        except:
            pass
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for key in self.model_weights:
                self.model_weights[key] /= total_weight
    
    def predict_ensemble(self, tokens: List[int], top_k: int = 5,
                        task: str = "generation") -> List[Tuple[int, float]]:
        """
        Get predictions from all available models and combine them
        
        Args:
            tokens: Input token sequence
            top_k: Return top-k predictions
            task: Task type (generation, classification, prediction)
        
        Returns:
            Ensemble predictions with combined probabilities
        """
        ensemble_probs = np.zeros(self.vocab_size)
        total_weight = 0
        
        for model_name, model in self.models.items():
            if model is None:
                continue
            
            try:
                # Get model's predictions
                predictions = model.predict(tokens, top_k=min(top_k, self.vocab_size))
                
                # Get routing score for this task
                task_score = self.routing_scores[model_name].get(task, 0.5)
                
                # Weight combining model weight and task performance
                weight = self.model_weights.get(model_name, 0.1) * task_score
                total_weight += weight
                
                # Add to ensemble
                for token_id, prob in predictions:
                    ensemble_probs[token_id] += prob * weight
            
            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                continue
        
        # Normalize
        if total_weight > 0:
            ensemble_probs /= total_weight
        
        # Return top-k
        top_indices = np.argsort(ensemble_probs)[-top_k:][::-1]
        return [(int(idx), float(ensemble_probs[idx])) for idx in top_indices]
    
    def adaptive_route(self, tokens: List[int], task: str = "generation") -> Tuple[str, List]:
        """
        Route to single best model for speed
        
        Returns:
            (model_name, predictions)
        """
        best_model = None
        best_score = -1
        
        for model_name in self.models.keys():
            score = self.routing_scores[model_name].get(task, 0.5)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        if best_model and self.models[best_model] is not None:
            predictions = self.models[best_model].predict(tokens, top_k=5)
            return best_model, predictions
        
        # Fallback to ensemble if single model fails
        return "ensemble", self.predict_ensemble(tokens, task=task)
    
    def update_routing_weights(self, model_name: str, task: str, score: float):
        """
        Update routing scores based on performance feedback
        """
        if model_name in self.routing_scores:
            if task not in self.routing_scores[model_name]:
                self.routing_scores[model_name][task] = 0.5
            
            # Exponential moving average
            alpha = 0.1
            self.routing_scores[model_name][task] = (
                alpha * score + (1 - alpha) * self.routing_scores[model_name][task]
            )
    
    def get_stats(self) -> Dict:
        """Get ensemble statistics"""
        return {
            "model_type": "Hybrid-Ensemble",
            "num_models": len(self.models),
            "available_models": list(self.models.keys()),
            "model_weights": self.model_weights,
            "routing_scores": self.routing_scores,
            "individual_stats": {
                name: model.get_stats() 
                for name, model in self.models.items() 
                if model is not None
            }
        }
