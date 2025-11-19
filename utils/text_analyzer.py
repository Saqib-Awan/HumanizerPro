import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textstat import flesch_reading_ease, flesch_kincaid_grade
import language_tool_python
from collections import Counter
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextAnalyzer:
    def __init__(self):
        self.tool = language_tool_python.LanguageTool('en-US')
        self.stop_words = set(stopwords.words('english'))
    
    def analyze_text(self, text):
        """Comprehensive text analysis"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words_clean = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        # Basic metrics
        char_count = len(text)
        word_count = len(words_clean)
        sentence_count = len(sentences)
        paragraph_count = len(text.split('\n\n'))
        
        # Readability scores
        flesch_ease = flesch_reading_ease(text)
        flesch_grade = flesch_kincaid_grade(text)
        
        # Vocabulary diversity
        unique_words = len(set(words_clean))
        lexical_diversity = unique_words / len(words_clean) if words_clean else 0
        
        # Sentence structure
        avg_sentence_length = word_count / sentence_count if sentence_count else 0
        avg_word_length = sum(len(word) for word in words_clean) / len(words_clean) if words_clean else 0
        
        # Grammar and spelling
        matches = self.tool.check(text)
        grammar_errors = len(matches)
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'flesch_reading_ease': flesch_ease,
            'flesch_kincaid_grade': flesch_grade,
            'lexical_diversity': lexical_diversity,
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'grammar_errors': grammar_errors,
            'unique_words': unique_words
        }
    
    def get_grammar_suggestions(self, text):
        """Get grammar and style suggestions"""
        matches = self.tool.check(text)
        suggestions = []
        
        for match in matches[:10]:  # Limit to top 10 suggestions
            suggestions.append({
                'context': match.context,
                'message': match.message,
                'replacements': match.replacements[:3],
                'offset': match.offset,
                'error_length': match.errorLength
            })
        
        return suggestions