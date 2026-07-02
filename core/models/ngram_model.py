"""
N-Gram Language Model with Self-Improvement & Adaptation

Statistical model that learns word/token sequences and probabilities.
Automatically improves through feedback and usage patterns.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict
import math

class NGramModel:
    """N-Gram Language Model with Evolution"""
    
    def __init__(self, n: int = 3, min_freq: int = 2):
        """
        Args:
            n: N-gram size (1=unigram, 2=bigram, 3=trigram, etc.)
            min_freq: Minimum frequency to keep n-gram
        """
        self.n = n
        self.min_freq = min_freq
        self.ngrams = defaultdict(lambda: defaultdict(int))  # {context: {word: count}}
        self.vocabulary = set()
        self.unigrams = defaultdict(int)  # For backoff
        self.total_tokens = 0
        self.improvement_history = []
    
    def train(self, text: str):
        """Train on text corpus"""
        tokens = text.lower().split()
        self.vocabulary.update(tokens)
        self.total_tokens += len(tokens)
        
        # Build unigrams
        for token in tokens:
            self.unigrams[token] += 1
        
        # Build n-grams
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i+self.n-1])
            word = tokens[i+self.n-1]
            self.ngrams[context][word] += 1
    
    def predict(self, context: Tuple, top_k: int = 5, use_backoff: bool = True) -> List[Tuple]:
        """
        Predict next token(s) given context
        
        Returns list of (token, probability) tuples
        """
        if context in self.ngrams and self.ngrams[context]:
            predictions = self.ngrams[context]
        elif use_backoff and len(context) > 1:
            # Backoff to shorter context
            return self.predict(context[1:], top_k, use_backoff)
        else:
            # Use unigrams as last resort
            predictions = self.unigrams
        
        total = sum(predictions.values())
        sorted_preds = [
            (word, count / total)
            for word, count in sorted(
                predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
        ]
        
        return sorted_preds
    
    def generate(self, prompt: str = "", max_length: int = 50) -> str:
        """Generate text starting with prompt"""
        if not prompt:
            # Start with random unigram
            import random
            words = list(self.unigrams.keys())
            if not words:
                return ""
            prompt = random.choice(words)
        
        tokens = prompt.split()
        
        for _ in range(max_length):
            context = tuple(tokens[-(self.n-1):]) if len(tokens) >= self.n-1 else tuple(tokens)
            predictions = self.predict(context, top_k=1)
            
            if not predictions:
                break
            
            next_word = predictions[0][0]
            tokens.append(next_word)
        
        return " ".join(tokens)
    
    def improve(self):
        """
        Self-improvement: prune, adapt, optimize
        """
        action = {
            "before_ngrams": len(self.ngrams),
            "before_vocab": len(self.vocabulary),
            "actions": []
        }
        
        # Prune low-frequency n-grams
        pruned_count = 0
        for context in list(self.ngrams.keys()):
            words = self.ngrams[context]
            for word in list(words.keys()):
                if words[word] < self.min_freq:
                    del words[word]
                    pruned_count += 1
            
            if not words:
                del self.ngrams[context]
        
        action["pruned_ngrams"] = pruned_count
        action["after_ngrams"] = len(self.ngrams)
        action["actions"].append(f"Pruned {pruned_count} low-frequency n-grams")
        
        self.improvement_history.append(action)
        return action
    
    def perplexity(self, text: str) -> float:
        """
        Calculate perplexity on text
        (Lower is better - model is more confident)
        """
        tokens = text.lower().split()
        if len(tokens) < self.n:
            return float('inf')
        
        log_prob_sum = 0
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i+self.n-1])
            word = tokens[i+self.n-1]
            
            predictions = self.predict(context, use_backoff=True)
            probs = {w: p for w, p in predictions}
            prob = probs.get(word, 1e-10)
            
            log_prob_sum += math.log2(prob)
        
        return 2 ** (-log_prob_sum / (len(tokens) - self.n + 1))
    
    def get_stats(self) -> Dict:
        """Get model statistics"""
        return {
            "n_gram_size": self.n,
            "total_ngrams": len(self.ngrams),
            "vocabulary_size": len(self.vocabulary),
            "total_tokens_trained": self.total_tokens,
            "min_frequency_threshold": self.min_freq,
            "improvements_applied": len(self.improvement_history)
        }
