"""
Advanced Natural Language Processing Module for Aetheron AI Platform
Includes text processing, embeddings, sentiment analysis, and language models
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass
import json
import os
import re
from datetime import datetime
from collections import Counter, defaultdict
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NLPConfig:
    """Configuration for NLP tasks"""
    task_type: str = "classification"  # classification, sentiment, embedding, generation
    vocab_size: int = 10000
    embedding_dim: int = 128
    max_sequence_length: int = 256
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    language: str = "en"

class TextPreprocessor:
    """Advanced text preprocessing utilities"""
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True,
                 remove_stopwords: bool = True, language: str = "en"):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.language = language
        
        # Common English stopwords
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'you', 'your', 'this', 'but',
            'or', 'not', 'have', 'had', 'can', 'could', 'should', 'would'
        }
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = self.clean_text(text)
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        tokens = text.split()
        
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        return tokens
    
    def extract_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """Extract n-grams from tokens"""
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract basic text features"""
        tokens = self.tokenize(text)
        
        features = {
            'char_count': len(text),
            'word_count': len(tokens),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_word_length': np.mean([len(word) for word in tokens]) if tokens else 0,
            'unique_words': len(set(tokens)),
            'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
            'punctuation_count': sum(1 for char in text if char in string.punctuation),
            'uppercase_count': sum(1 for char in text if char.isupper()),
            'digit_count': sum(1 for char in text if char.isdigit())
        }
        
        return features

class Vocabulary:
    """Vocabulary manager for text data"""
    
    def __init__(self, max_vocab_size: int = 10000):
        self.max_vocab_size = max_vocab_size
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.word_counts = Counter()
        self.current_id = 4
    
    def build_vocabulary(self, texts: List[str], preprocessor: TextPreprocessor):
        """Build vocabulary from texts"""
        logger.info("Building vocabulary...")
        
        # Count word frequencies
        for text in texts:
            tokens = preprocessor.tokenize(text)
            self.word_counts.update(tokens)
        
        # Add most frequent words to vocabulary
        most_common = self.word_counts.most_common(self.max_vocab_size - 4)
        
        for word, count in most_common:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.current_id
                self.id_to_word[self.current_id] = word
                self.current_id += 1
        
        logger.info(f"Built vocabulary with {len(self.word_to_id)} words")
    
    def text_to_sequence(self, text: str, preprocessor: TextPreprocessor, 
                        max_length: int = None) -> List[int]:
        """Convert text to sequence of token IDs"""
        tokens = preprocessor.tokenize(text)
        sequence = [self.word_to_id.get(token, 1) for token in tokens]  # 1 is <UNK>
        
        if max_length:
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
            else:
                sequence = sequence + [0] * (max_length - len(sequence))  # 0 is <PAD>
        
        return sequence
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """Convert sequence of token IDs back to text"""
        words = [self.id_to_word.get(token_id, '<UNK>') for token_id in sequence]
        # Remove padding and special tokens
        words = [word for word in words if word not in ['<PAD>', '<START>', '<END>']]
        return ' '.join(words)

class WordEmbeddings:
    """Simple word embedding implementation"""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings randomly
        self.embeddings = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
    
    def get_embedding(self, word_id: int) -> np.ndarray:
        """Get embedding for word ID"""
        if 0 <= word_id < self.vocab_size:
            return self.embeddings[word_id]
        else:
            return self.embeddings[1]  # UNK token
    
    def get_sequence_embeddings(self, sequence: List[int]) -> np.ndarray:
        """Get embeddings for sequence of word IDs"""
        embeddings = np.array([self.get_embedding(word_id) for word_id in sequence])
        return embeddings
    
    def update_embedding(self, word_id: int, gradient: np.ndarray, learning_rate: float):
        """Update embedding using gradient"""
        if 0 <= word_id < self.vocab_size:
            self.embeddings[word_id] -= learning_rate * gradient

class SimpleRNN:
    """Simple Recurrent Neural Network"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.Wxh = np.random.randn(input_dim, hidden_dim) * 0.1
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.Why = np.random.randn(hidden_dim, output_dim) * 0.1
        self.bh = np.zeros((1, hidden_dim))
        self.by = np.zeros((1, output_dim))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass through RNN"""
        seq_length = X.shape[0]
        hidden_states = []
        
        h = np.zeros((1, self.hidden_dim))
        
        for t in range(seq_length):
            x_t = X[t].reshape(1, -1)
            h = self.tanh(np.dot(x_t, self.Wxh) + np.dot(h, self.Whh) + self.bh)
            hidden_states.append(h.copy())
        
        # Output from last hidden state
        output = np.dot(h, self.Why) + self.by
        output = self.softmax(output)
        
        return output, hidden_states
    
    def predict(self, X: np.ndarray) -> int:
        """Make prediction"""
        output, _ = self.forward(X)
        return np.argmax(output[0])

class SentimentAnalyzer:
    """Sentiment analysis using simple neural network"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 32):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Components
        self.embeddings = WordEmbeddings(vocab_size, embedding_dim)
        self.rnn = SimpleRNN(embedding_dim, hidden_dim, 3)  # 3 classes: neg, neu, pos
        
        # Training history
        self.training_history = {'loss': [], 'accuracy': []}
    
    def predict_sentiment(self, sequence: List[int]) -> Dict[str, Any]:
        """Predict sentiment for sequence"""
        # Get embeddings
        embeddings = self.embeddings.get_sequence_embeddings(sequence)
        
        # Forward through RNN
        output, _ = self.rnn.forward(embeddings)
        
        # Get prediction
        probabilities = output[0]
        predicted_class = np.argmax(probabilities)
        
        sentiment_labels = ['negative', 'neutral', 'positive']
        
        return {
            'sentiment': sentiment_labels[predicted_class],
            'confidence': probabilities[predicted_class],
            'probabilities': {
                'negative': probabilities[0],
                'neutral': probabilities[1],
                'positive': probabilities[2]
            }
        }
    
    def train_step(self, sequence: List[int], label: int) -> float:
        """Single training step"""
        # Get embeddings
        embeddings = self.embeddings.get_sequence_embeddings(sequence)
        
        # Forward pass
        output, hidden_states = self.rnn.forward(embeddings)
        
        # Calculate loss (cross-entropy)
        predicted_probs = output[0]
        loss = -np.log(predicted_probs[label] + 1e-8)
        
        # Simple weight updates (approximation)
        self.rnn.Why -= self.rnn.learning_rate * np.outer(hidden_states[-1], predicted_probs)
        self.rnn.by -= self.rnn.learning_rate * predicted_probs
        
        return loss

class TextClassifier:
    """General text classification"""
    
    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int = 64):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Components
        self.embeddings = WordEmbeddings(vocab_size, embedding_dim)
        
        # Simple classification layer (bag of words + linear)
        self.classifier_weights = np.random.randn(embedding_dim, num_classes) * 0.1
        self.classifier_bias = np.zeros(num_classes)
        
        # Training history
        self.training_history = {'loss': [], 'accuracy': []}
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def predict(self, sequence: List[int]) -> Dict[str, Any]:
        """Predict class for sequence"""
        # Get embeddings and average them (bag of words)
        embeddings = self.embeddings.get_sequence_embeddings(sequence)
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Classification
        logits = np.dot(avg_embedding, self.classifier_weights) + self.classifier_bias
        probabilities = self.softmax(logits)
        predicted_class = np.argmax(probabilities)
        
        return {
            'predicted_class': predicted_class,
            'confidence': probabilities[predicted_class],
            'probabilities': probabilities.tolist()
        }
    
    def train_step(self, sequence: List[int], label: int) -> float:
        """Single training step"""
        # Get embeddings
        embeddings = self.embeddings.get_sequence_embeddings(sequence)
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Forward pass
        logits = np.dot(avg_embedding, self.classifier_weights) + self.classifier_bias
        probabilities = self.softmax(logits)
        
        # Calculate loss
        loss = -np.log(probabilities[label] + 1e-8)
        
        # Update weights (simplified gradient descent)
        grad_output = probabilities.copy()
        grad_output[label] -= 1
        
        learning_rate = 0.001
        self.classifier_weights -= learning_rate * np.outer(avg_embedding, grad_output)
        self.classifier_bias -= learning_rate * grad_output
        
        return loss

class TextSummarizer:
    """Extractive text summarization"""
    
    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences
    
    def extractive_summarize(self, text: str, preprocessor: TextPreprocessor) -> str:
        """Create extractive summary"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= self.max_sentences:
            return text
        
        # Score sentences based on word frequency
        all_words = []
        for sentence in sentences:
            words = preprocessor.tokenize(sentence)
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        
        # Score each sentence
        sentence_scores = []
        for sentence in sentences:
            words = preprocessor.tokenize(sentence)
            score = sum(word_freq[word] for word in words)
            sentence_scores.append((score, sentence))
        
        # Select top sentences
        sentence_scores.sort(reverse=True)
        top_sentences = [sent for _, sent in sentence_scores[:self.max_sentences]]
        
        return '. '.join(top_sentences) + '.'

class NLPExperimentTracker:
    """Track NLP experiments"""
    
    def __init__(self, experiment_dir: str = "experiments/nlp"):
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        
        self.experiments = []
    
    def create_experiment(self, name: str, config: NLPConfig, 
                         description: str = "") -> str:
        """Create new NLP experiment"""
        experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_path = os.path.join(self.experiment_dir, experiment_id)
        os.makedirs(experiment_path, exist_ok=True)
        
        # Save config
        config_path = os.path.join(experiment_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        # Save metadata
        metadata = {
            'id': experiment_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'task_type': config.task_type,
            'status': 'created'
        }
        
        metadata_path = os.path.join(experiment_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.experiments.append(metadata)
        logger.info(f"Created NLP experiment: {experiment_id}")
        
        return experiment_id
    
    def save_results(self, experiment_id: str, results: Dict[str, Any]):
        """Save experiment results"""
        experiment_path = os.path.join(self.experiment_dir, experiment_id)
        results_path = os.path.join(experiment_path, "results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                serializable_results[key] = [v.tolist() for v in value]
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results for experiment: {experiment_id}")

def create_sample_texts() -> Tuple[List[str], List[int]]:
    """Create sample texts for testing"""
    texts = [
        "This is a great product! I love it very much.",
        "The service was terrible. I hate this company.",
        "The weather is nice today. It's sunny and warm.",
        "I'm feeling sad about the recent news.",
        "This movie is amazing! Best film I've ever seen.",
        "The food was okay, nothing special.",
        "I'm so excited about the upcoming vacation!",
        "The meeting was boring and too long.",
        "Beautiful sunset over the ocean tonight.",
        "Traffic is horrible during rush hour."
    ]
    
    # Sentiment labels: 0=negative, 1=neutral, 2=positive
    labels = [2, 0, 1, 0, 2, 1, 2, 0, 2, 0]
    
    return texts, labels

def create_nlp_system(task_type: str = "sentiment") -> Dict[str, Any]:
    """Create and configure NLP system"""
    
    # Create configuration
    config = NLPConfig(task_type=task_type, vocab_size=1000, max_sequence_length=50)
    
    # Create components
    preprocessor = TextPreprocessor()
    vocabulary = Vocabulary(config.vocab_size)
    sentiment_analyzer = SentimentAnalyzer(config.vocab_size, config.embedding_dim)
    text_classifier = TextClassifier(config.vocab_size, 3)  # 3 classes for sentiment
    summarizer = TextSummarizer()
    tracker = NLPExperimentTracker()
    
    return {
        'config': config,
        'preprocessor': preprocessor,
        'vocabulary': vocabulary,
        'sentiment_analyzer': sentiment_analyzer,
        'text_classifier': text_classifier,
        'summarizer': summarizer,
        'tracker': tracker
    }

# Example usage and testing
if __name__ == "__main__":
    print("Testing NLP System...")
    
    # Create NLP system
    nlp_system = create_nlp_system("sentiment")
    
    # Get sample data
    sample_texts, sample_labels = create_sample_texts()
    
    # Create experiment
    experiment_id = nlp_system['tracker'].create_experiment(
        "sentiment_analysis_test",
        nlp_system['config'],
        "Testing sentiment analysis on sample texts"
    )
    
    # Build vocabulary
    nlp_system['vocabulary'].build_vocabulary(sample_texts, nlp_system['preprocessor'])
    
    # Test text preprocessing
    print("Testing text preprocessing...")
    for i, text in enumerate(sample_texts[:3]):
        tokens = nlp_system['preprocessor'].tokenize(text)
        features = nlp_system['preprocessor'].extract_features(text)
        sequence = nlp_system['vocabulary'].text_to_sequence(
            text, nlp_system['preprocessor'], nlp_system['config'].max_sequence_length
        )
        
        print(f"Text {i+1}: {text}")
        print(f"Tokens: {tokens}")
        print(f"Sequence length: {len(sequence)}")
        print(f"Features: {features}")
        print()
    
    # Test sentiment analysis
    print("Testing sentiment analysis...")
    sentiment_analyzer = nlp_system['sentiment_analyzer']
    
    # Simple training loop
    training_losses = []
    for epoch in range(10):
        epoch_loss = 0
        for text, label in zip(sample_texts, sample_labels):
            sequence = nlp_system['vocabulary'].text_to_sequence(
                text, nlp_system['preprocessor'], 20
            )
            loss = sentiment_analyzer.train_step(sequence, label)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(sample_texts)
        training_losses.append(avg_loss)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    # Test predictions
    print("\nTesting sentiment predictions...")
    predictions = []
    for i, text in enumerate(sample_texts[:5]):
        sequence = nlp_system['vocabulary'].text_to_sequence(
            text, nlp_system['preprocessor'], 20
        )
        result = sentiment_analyzer.predict_sentiment(sequence)
        predictions.append(result)
        
        print(f"Text: {text}")
        print(f"Predicted sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        print(f"True label: {['negative', 'neutral', 'positive'][sample_labels[i]]}")
        print()
    
    # Test text classification
    print("Testing text classification...")
    classifier = nlp_system['text_classifier']
    
    # Train classifier
    for epoch in range(5):
        epoch_loss = 0
        for text, label in zip(sample_texts, sample_labels):
            sequence = nlp_system['vocabulary'].text_to_sequence(
                text, nlp_system['preprocessor'], 20
            )
            loss = classifier.train_step(sequence, label)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(sample_texts)
        if epoch % 2 == 0:
            print(f"Classifier Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    # Test summarization
    print("Testing text summarization...")
    long_text = " ".join(sample_texts)
    summary = nlp_system['summarizer'].extractive_summarize(
        long_text, nlp_system['preprocessor']
    )
    
    print(f"Original text length: {len(long_text)} characters")
    print(f"Summary length: {len(summary)} characters")
    print(f"Summary: {summary}")
    
    # Save results
    results = {
        'training_losses': training_losses,
        'predictions': [p['sentiment'] for p in predictions],
        'vocabulary_size': len(nlp_system['vocabulary'].word_to_id),
        'summary_compression': len(summary) / len(long_text),
        'sample_features': nlp_system['preprocessor'].extract_features(sample_texts[0])
    }
    
    nlp_system['tracker'].save_results(experiment_id, results)
    
    print("\n✅ NLP module tests completed successfully!")
