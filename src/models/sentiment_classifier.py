"""
Simple Sentiment Classifier using Naive Bayes
A basic machine learning model for sentiment analysis
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pickle
import logging

logger = logging.getLogger(__name__)


class SentimentClassifier:
    """
    Simple Naive Bayes sentiment classifier for text.

    This is a basic implementation that can classify text as positive, negative, or neutral.
    """

    def __init__(self):
        self.word_probs = {}
        self.class_probs = {}
        self.vocabulary = set()
        self.classes = ["positive", "negative", "neutral"]
        self.is_trained = False

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by cleaning and tokenizing."""
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and special characters
        text = re.sub(r"[^\w\s]", "", text)

        # Split into words
        words = text.split()

        # Remove very short words
        words = [word for word in words if len(word) > 1]

        return words

    def train(self, texts: List[str], labels: List[str]) -> None:
        """
        Train the sentiment classifier.

        Args:
            texts: List of text samples
            labels: List of corresponding labels ('positive', 'negative', 'neutral')
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")

        # Initialize counts
        class_counts = defaultdict(int)
        word_counts = defaultdict(lambda: defaultdict(int))
        total_words = defaultdict(int)

        # Count occurrences
        for text, label in zip(texts, labels):
            if label not in self.classes:
                continue

            class_counts[label] += 1
            words = self._preprocess_text(text)

            for word in words:
                word_counts[label][word] += 1
                total_words[label] += 1
                self.vocabulary.add(word)

        # Calculate probabilities
        total_samples = sum(class_counts.values())

        # Class probabilities
        for class_name in self.classes:
            self.class_probs[class_name] = class_counts[class_name] / total_samples

        # Word probabilities (with Laplace smoothing)
        vocab_size = len(self.vocabulary)
        self.word_probs = {}

        for class_name in self.classes:
            self.word_probs[class_name] = {}
            for word in self.vocabulary:
                # Laplace smoothing: (count + 1) / (total_words + vocab_size)
                count = word_counts[class_name][word]
                self.word_probs[class_name][word] = (count + 1) / (
                    total_words[class_name] + vocab_size
                )

        self.is_trained = True
        logger.info(f"Trained sentiment classifier on {len(texts)} samples")

    def predict(self, text: str) -> Dict[str, float]:
        """
        Predict sentiment probabilities for text.

        Args:
            text: Text to classify

        Returns:
            Dictionary with probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        words = self._preprocess_text(text)
        if not words:
            # Return neutral for empty text
            return {
                class_name: (1.0 if class_name == "neutral" else 0.0)
                for class_name in self.classes
            }

        # Calculate log probabilities for each class
        log_probs = {}

        for class_name in self.classes:
            # Start with class probability
            log_prob = np.log(self.class_probs[class_name])

            # Add word probabilities
            for word in words:
                if word in self.word_probs[class_name]:
                    log_prob += np.log(self.word_probs[class_name][word])
                else:
                    # Unknown word - use uniform probability
                    log_prob += np.log(1.0 / (len(self.vocabulary) + 1))

            log_probs[class_name] = log_prob

        # Convert to probabilities
        max_log_prob = max(log_probs.values())
        probs = {}
        total_prob = 0

        for class_name, log_prob in log_probs.items():
            # Subtract max for numerical stability
            prob = np.exp(log_prob - max_log_prob)
            probs[class_name] = prob
            total_prob += prob

        # Normalize
        for class_name in probs:
            probs[class_name] /= total_prob

        return probs

    def predict_class(self, text: str) -> str:
        """
        Predict the most likely sentiment class.

        Args:
            text: Text to classify

        Returns:
            Predicted class ('positive', 'negative', or 'neutral')
        """
        probs = self.predict(text)
        return max(probs, key=probs.get)

    def save_model(self, filepath: str) -> None:
        """Save the trained model to a file."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            "word_probs": self.word_probs,
            "class_probs": self.class_probs,
            "vocabulary": list(self.vocabulary),
            "classes": self.classes,
            "is_trained": self.is_trained,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model from a file."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.word_probs = model_data["word_probs"]
        self.class_probs = model_data["class_probs"]
        self.vocabulary = set(model_data["vocabulary"])
        self.classes = model_data["classes"]
        self.is_trained = model_data["is_trained"]

        logger.info(f"Model loaded from {filepath}")

    @classmethod
    def create_sample_model(cls) -> "SentimentClassifier":
        """Create a pre-trained model with sample data for demonstration."""
        classifier = cls()

        # Sample training data
        texts = [
            "I love this product, it's amazing!",
            "This is fantastic, highly recommend",
            "Great quality and excellent service",
            "Wonderful experience, will buy again",
            "This is terrible, complete waste",
            "Awful product, do not buy",
            "Horrible quality and bad service",
            "Worst purchase I've ever made",
            "This is okay, nothing special",
            "Average product, meets expectations",
            "Decent quality but could be better",
            "It's fine, works as expected",
        ]

        labels = [
            "positive",
            "positive",
            "positive",
            "positive",
            "negative",
            "negative",
            "negative",
            "negative",
            "neutral",
            "neutral",
            "neutral",
            "neutral",
        ]

        classifier.train(texts, labels)
        return classifier
