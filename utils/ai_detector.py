import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from utils.text_analyzer import TextAnalyzer

class AIDetector:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        self.ai_indicators = [
            r'\bhowever\b.*\bimportant\b.*\bnote\b',
            r'\bit is crucial to\b',
            r'\bin conclusion\b',
            r'\bas an ai\b',
            r'\bas a language model\b',
            r'\bmoreover\b',
            r'\bfurthermore\b',
            r'\badditionally\b',
            r'\bin summary\b',
            r'\bto summarize\b'
        ]
        
    def detect_ai_patterns(self, text):
        """Detect common AI writing patterns"""
        text_lower = text.lower()
        
        # Pattern matching
        pattern_matches = 0
        for pattern in self.ai_indicators:
            if re.search(pattern, text_lower, re.IGNORECASE):
                pattern_matches += 1
        
        # Structural analysis
        analysis = self.analyzer.analyze_text(text)
        
        # AI-like characteristics
        ai_score = 0
        
        # High lexical diversity (AI tends to use diverse vocabulary)
        if analysis['lexical_diversity'] > 0.7:
            ai_score += 20
        
        # Perfect grammar (AI rarely makes grammar mistakes)
        if analysis['grammar_errors'] == 0:
            ai_score += 15
        
        # Consistent sentence length
        if analysis['avg_sentence_length'] > 15 and analysis['avg_sentence_length'] < 25:
            ai_score += 10
        
        # Pattern matches
        ai_score += pattern_matches * 5
        
        # Readability score (AI often produces very readable text)
        if analysis['flesch_reading_ease'] > 60:
            ai_score += 10
        
        # Cap the score at 100
        ai_score = min(ai_score, 100)
        
        human_score = 100 - ai_score
        
        return {
            'ai_score': ai_score,
            'human_score': human_score,
            'pattern_matches': pattern_matches,
            'indicators_found': self.get_specific_indicators(text_lower),
            'analysis': analysis
        }
    
    def get_specific_indicators(self, text_lower):
        """Get specific AI indicators found in text"""
        indicators = []
        indicator_patterns = {
            'Overly formal transitions': [r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b'],
            'Academic phrasing': [r'\bit is important to note\b', r'\bit is crucial to\b'],
            'Summary language': [r'\bin conclusion\b', r'\bin summary\b', r'\bto summarize\b'],
            'AI self-reference': [r'\bas an ai\b', r'\bas a language model\b']
        }
        
        for category, patterns in indicator_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    indicators.append(category)
                    break
        
        return indicators
    
    def advanced_detection(self, text):
        """More advanced AI detection using multiple features"""
        base_detection = self.detect_ai_patterns(text)
        
        # Additional features
        sentences = text.split('.')
        sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]
        
        if sentence_lengths:
            length_variance = np.var(sentence_lengths)
            # Low variance in sentence length can indicate AI
            if length_variance < 10:
                base_detection['ai_score'] += 5
        
        # Check for repetitive structures
        words = text.lower().split()
        word_freq = Counter(words)
        most_common_ratio = word_freq.most_common(1)[0][1] / len(words) if words else 0
        if most_common_ratio > 0.05:  # If a word appears too frequently
            base_detection['ai_score'] += 3
        
        base_detection['ai_score'] = min(base_detection['ai_score'], 100)
        base_detection['human_score'] = 100 - base_detection['ai_score']
        
        return base_detection