"""
Text processing utilities for JarvisAI.

This module provides text analysis functions including keyword extraction,
sentiment analysis, and text summarization.
"""

import re
from collections import Counter
from typing import List, Dict, Any


def extract_keywords(text: str, num_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis."""
    # Clean and tokenize text
    words = re.findall(r"\b\w+\b", text.lower())
    # Remove common stop words
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
    }
    filtered_words = [
        word for word in words if len(word) > 2 and word not in stop_words
    ]

    # Count frequency
    word_counts = Counter(filtered_words)

    # Return top keywords
    return [word for word, _ in word_counts.most_common(num_keywords)]


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Simple sentiment analysis based on word lists."""
    positive_words = {
        "good",
        "great",
        "excellent",
        "amazing",
        "wonderful",
        "fantastic",
        "love",
        "like",
        "best",
        "awesome",
        "perfect",
        "happy",
        "joy",
        "pleasure",
        "success",
        "win",
        "victory",
        "achievement",
    }
    negative_words = {
        "bad",
        "terrible",
        "awful",
        "horrible",
        "hate",
        "dislike",
        "worst",
        "ugly",
        "sad",
        "angry",
        "frustrated",
        "failure",
        "lose",
        "defeat",
        "problem",
        "issue",
        "error",
        "bug",
    }

    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)

    total_sentiment_words = positive_count + negative_count
    if total_sentiment_words == 0:
        sentiment = "neutral"
        confidence = 0.5
    else:
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = positive_count / total_sentiment_words
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = negative_count / total_sentiment_words
        else:
            sentiment = "neutral"
            confidence = 0.5

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 3),
        "positive_words": positive_count,
        "negative_words": negative_count,
    }


def generate_summary(text: str, max_length: int = 150) -> str:
    """Generate a simple extractive summary."""
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text[:max_length] + "..." if len(text) > max_length else text

    # Simple scoring based on sentence length and position
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        # Prefer sentences in the middle and beginning
        position_score = 1.0 - abs(i - len(sentences) // 2) / (len(sentences) // 2)
        length_score = min(
            len(sentence.split()) / 20, 1.0
        )  # Prefer sentences around 20 words
        score = (position_score + length_score) / 2
        scored_sentences.append((score, sentence))

    # Sort by score and take top sentences
    scored_sentences.sort(reverse=True)
    summary_sentences = [sentence for _, sentence in scored_sentences[:3]]  # Take top 3

    summary = ". ".join(summary_sentences) + "."
    return summary[:max_length] + "..." if len(summary) > max_length else summary
