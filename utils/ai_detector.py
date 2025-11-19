import re
from collections import Counter
import numpy as np
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
            r'\bto summarize\b',
            r'\bit is worth noting\b',
            r'\bit should be noted\b',
            r'\bin this context\b',
            r'\bfrom this perspective\b',
            r'\bupon analysis\b'
        ]
        
    def detect_ai_patterns(self, text):
        """Advanced AI detection with multiple techniques"""
        if not text or len(text.strip()) < 10:
            return self._get_default_scores()
            
        text_lower = text.lower()
        
        # Multiple detection methods
        pattern_score = self._pattern_analysis(text_lower)
        structural_score = self._structural_analysis(text)
        vocabulary_score = self._vocabulary_analysis(text)
        readability_score = self._readability_analysis(text)
        
        # Weighted combination
        total_ai_score = (
            pattern_score * 0.4 +
            structural_score * 0.3 +
            vocabulary_score * 0.2 +
            readability_score * 0.1
        )
        
        ai_score = min(total_ai_score, 100)
        human_score = 100 - ai_score
        
        return {
            'ai_score': ai_score,
            'human_score': human_score,
            'pattern_matches': int(pattern_score / 5),
            'indicators_found': self.get_specific_indicators(text_lower),
            'analysis': self.analyzer.analyze_text(text),
            'confidence': self._calculate_confidence(ai_score)
        }
    
    def _pattern_analysis(self, text_lower):
        """Analyze patterns characteristic of AI writing"""
        score = 0
        
        # Pattern matching
        for pattern in self.ai_indicators:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += 8
        
        # Check for perfect grammar structure
        if re.search(r'\.\s+[A-Z]', text_lower) and not re.search(r'\.\s+[a-z]', text_lower):
            score += 10
        
        # Check for repetitive sentence starters
        sentences = re.split(r'[.!?]+', text_lower)
        starters = [sent.strip().split()[0] if sent.strip().split() else '' for sent in sentences if sent.strip()]
        starter_counts = Counter(starters)
        if starters and starter_counts.most_common(1)[0][1] / len(starters) > 0.3:
            score += 15
        
        return min(score, 40)
    
    def _structural_analysis(self, text):
        """Analyze structural patterns"""
        score = 0
        analysis = self.analyzer.analyze_text(text)
        
        # Perfect grammar (AI rarely makes mistakes)
        if analysis['grammar_errors'] == 0:
            score += 20
        
        # Consistent sentence length (AI tends to be uniform)
        if 5 <= analysis['avg_sentence_length'] <= 25:
            score += 10
        
        # High lexical diversity (AI uses varied vocabulary)
        if analysis['lexical_diversity'] > 0.7:
            score += 10
        
        return min(score, 30)
    
    def _vocabulary_analysis(self, text):
        """Analyze vocabulary usage"""
        score = 0
        words = text.lower().split()
        
        if len(words) < 10:
            return score
        
        word_freq = Counter(words)
        
        # Check for overly formal vocabulary
        formal_words = ['moreover', 'furthermore', 'however', 'therefore', 'consequently', 
                       'additionally', 'notwithstanding', 'accordingly']
        formal_count = sum(1 for word in words if word in formal_words)
        if formal_count / len(words) > 0.05:
            score += 15
        
        # Check for lack of contractions (AI tends to avoid them)
        contractions = ["don't", "can't", "won't", "it's", "that's", "you're"]
        contraction_count = sum(1 for word in words if word in contractions)
        if contraction_count / len(words) < 0.01 and len(words) > 50:
            score += 10
        
        return min(score, 20)
    
    def _readability_analysis(self, text):
        """Analyze readability patterns"""
        score = 0
        analysis = self.analyzer.analyze_text(text)
        
        # AI often produces very readable text
        if analysis['flesch_reading_ease'] > 60:
            score += 5
        
        # Consistent grade level
        if 8 <= analysis['flesch_kincaid_grade'] <= 12:
            score += 5
        
        return min(score, 10)
    
    def _calculate_confidence(self, ai_score):
        """Calculate detection confidence"""
        if ai_score > 80:
            return "High"
        elif ai_score > 60:
            return "Medium"
        elif ai_score > 40:
            return "Low"
        else:
            return "Very Low"
    
    def _get_default_scores(self):
        """Return default scores for invalid text"""
        return {
            'ai_score': 0,
            'human_score': 100,
            'pattern_matches': 0,
            'indicators_found': [],
            'analysis': {},
            'confidence': 'Very Low'
        }
    
    def get_specific_indicators(self, text_lower):
        """Get specific AI indicators found in text"""
        indicators = []
        indicator_patterns = {
            'Overly formal transitions': [r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b'],
            'Academic phrasing': [r'\bit is important to note\b', r'\bit is crucial to\b', r'\bit is worth noting\b'],
            'Summary language': [r'\bin conclusion\b', r'\bin summary\b', r'\bto summarize\b'],
            'AI self-reference': [r'\bas an ai\b', r'\bas a language model\b'],
            'Formal structure': [r'\bit should be noted\b', r'\bupon analysis\b', r'\bin this context\b']
        }
        
        for category, patterns in indicator_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    indicators.append(category)
                    break
        
        return indicators
    
    def advanced_detection(self, text):
        """More advanced AI detection using ensemble methods"""
        base_result = self.detect_ai_patterns(text)
        
        # Additional checks for advanced detection
        if len(text.split()) > 100:
            # Check for paragraph structure consistency
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            if len(paragraphs) > 1:
                para_lengths = [len(p.split()) for p in paragraphs]
                length_variance = np.var(para_lengths)
                if length_variance < 50:  # Very consistent paragraph lengths
                    base_result['ai_score'] = min(base_result['ai_score'] + 5, 100)
        
        base_result['human_score'] = 100 - base_result['ai_score']
        
        return base_result