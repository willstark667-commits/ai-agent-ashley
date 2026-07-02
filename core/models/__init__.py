"""
Language Models Package

Models:
  ✓ N-gram: Statistical language model
  ✓ CNN: Convolutional neural network
  ✓ RNN: Recurrent neural network (LSTM)
  ✓ Transformer: Attention-based model
  ✓ Hybrid: Multi-model ensemble
  ✓ ENN: Evolving neural network

Usage:
  from ai_studio.core.models import NGramModel, CNNLanguageModel, RNNLanguageModel
  from ai_studio.core.models import TransformerLanguageModel, HybridLanguageModel, ENN
  
  # Choose model based on task
  if task == 'fast_prediction':
      model = NGramModel(n=3)
  elif task == 'classification':
      model = CNNLanguageModel()
  elif task == 'generation':
      model = TransformerLanguageModel()
  elif task == 'best_overall':
      model = HybridLanguageModel()
  elif task == 'adaptive':
      model = ENN()
"""

from .ngram_model import NGramModel
from .cnn_model import CNNLanguageModel
from .rnn_model import RNNLanguageModel
from .transformer_model import TransformerLanguageModel
from .hybrid_model import HybridLanguageModel
from .enn_model import ENN

__all__ = [
    "NGramModel",
    "CNNLanguageModel",
    "RNNLanguageModel",
    "TransformerLanguageModel",
    "HybridLanguageModel",
    "ENN"
]
