import re
import json
from typing import Dict, Any, List, Tuple
import numpy as np
from datetime import datetime


class DataProcessor:
    def __init__(self):
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])

    async def clean_text(self, text: str) -> str:
        """Clean and normalize text data"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)

        # Normalize case
        text = text.lower()

        return text.strip()

    async def extract_statistics(self, text: str) -> Dict[str, Any]:
        """Extract basic statistics from text"""
        if not text:
            return {}

        stats = {
            "word_count": len(text.split()),
            "char_count": len(text),
            "sentence_count": len(re.split(r'[.!?]+', text)),
            "avg_word_length": np.mean([len(word) for word in text.split()]) if text.split() else 0,
            "unique_words": len(set(text.split())),
            "numeric_count": len(re.findall(r'\b\d+\b', text))
        }

        # Calculate readability score (simplified)
        if stats["sentence_count"] > 0 and stats["word_count"] > 0:
            stats["readability"] = 206.835 - 1.015 * (stats["word_count"] / stats["sentence_count"]) - 84.6 * (
                        len(re.findall(r'\b\w{3,}\b', text)) / stats["word_count"])
        else:
            stats["readability"] = 0

        # Sentiment analysis (simplified)
        positive_words = set(['good', 'great', 'excellent', 'positive', 'success', 'effective', 'improved'])
        negative_words = set(['bad', 'poor', 'negative', 'failure', 'ineffective', 'worse'])

        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words > 0:
            stats["avg_sentiment"] = (positive_count - negative_count) / total_sentiment_words
        else:
            stats["avg_sentiment"] = 0

        return stats

    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {
            "organizations": [],
            "locations": [],
            "dates": [],
            "numbers": [],
            "acronyms": []
        }

        # Extract organizations (words in all caps or title case)
        org_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        entities["organizations"] = re.findall(org_pattern, text)

        # Extract dates
        date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b'
        entities["dates"] = re.findall(date_pattern, text)

        # Extract numbers
        number_pattern = r'\b(\d+\.?\d*)\b'
        entities["numbers"] = re.findall(number_pattern, text)

        # Extract acronyms (words in all caps with 2-5 letters)
        acronym_pattern = r'\b([A-Z]{2,5})\b'
        entities["acronyms"] = re.findall(acronym_pattern, text)

        return entities

    async def find_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Find patterns in text data"""
        patterns = []

        # Look for comparative patterns
        comparative_patterns = [
            (r'(\w+)\s+vs\.?\s+(\w+)', 'comparison'),
            (r'(\w+)\s+compared to\s+(\w+)', 'comparison'),
            (r'(\w+)\s+better than\s+(\w+)', 'superiority'),
            (r'(\w+)\s+worse than\s+(\w+)', 'inferiority')
        ]

        for pattern, pattern_type in comparative_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                patterns.append({
                    "type": pattern_type,
                    "pattern": match,
                    "context": self._get_context(text, match[0] if isinstance(match, tuple) else match)
                })

        # Look for numerical patterns
        num_pattern = r'(\d+\.?\d*)\s*(?:%|percent|times|\x)'
        num_matches = re.findall(num_pattern, text)
        if len(num_matches) >= 2:
            patterns.append({
                "type": "numerical_trend",
                "pattern": num_matches,
                "context": "Multiple numerical values found"
            })

        return patterns[:5]  # Return top 5 patterns

    def _get_context(self, text: str, term: str, window: int = 50) -> str:
        """Get context around a term"""
        if term not in text:
            return ""

        start = max(0, text.find(term) - window)
        end = min(len(text), text.find(term) + len(term) + window)

        context = text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."

        return context

    async def validate_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality metrics"""
        quality_metrics = {
            "completeness": 0.0,
            "consistency": 0.0,
            "timeliness": 0.0,
            "validity": 0.0,
            "overall_score": 0.0
        }

        # Check completeness
        total_fields = len(data)
        non_empty_fields = sum(1 for value in data.values() if value)
        quality_metrics["completeness"] = non_empty_fields / total_fields if total_fields > 0 else 0

        # Check timeliness (simple check for recent dates)
        date_fields = [v for v in data.values() if isinstance(v, str) and any(year in v for year in ['2024', '2023'])]
        if date_fields:
            quality_metrics["timeliness"] = len(date_fields) / total_fields

        # Overall score (weighted average)
        weights = {"completeness": 0.4, "consistency": 0.2, "timeliness": 0.2, "validity": 0.2}
        quality_metrics["overall_score"] = sum(
            quality_metrics[metric] * weights[metric]
            for metric in weights.keys()
        )

        return quality_metrics