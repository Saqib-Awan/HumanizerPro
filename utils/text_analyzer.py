import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textstat import flesch_reading_ease, flesch_kincaid_grade
from collections import Counter
import string

# Download required NLTK data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

# Download data on initialization
download_nltk_data()

class TextAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.common_errors = self._load_common_errors()
    
    def _load_common_errors(self):
        """Load common grammar errors without Java dependency"""
        return {
            'their there they\'re': ['their', 'there', 'they\'re'],
            'your you\'re': ['your', 'you\'re'],
            'its it\'s': ['its', 'it\'s'],
            'affect effect': ['affect', 'effect'],
            'then than': ['then', 'than'],
            'loose lose': ['loose', 'lose']
        }
    
    def analyze_text(self, text):
        """Comprehensive text analysis"""
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
            words_clean = [word for word in words if word.isalnum() and word not in self.stop_words]
            
            # Basic metrics
            char_count = len(text)
            word_count = len(words_clean)
            sentence_count = len(sentences)
            paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
            
            # Readability scores
            try:
                flesch_ease = flesch_reading_ease(text)
                flesch_grade = flesch_kincaid_grade(text)
            except:
                flesch_ease = 0
                flesch_grade = 0
            
            # Vocabulary diversity
            unique_words = len(set(words_clean))
            lexical_diversity = unique_words / len(words_clean) if words_clean else 0
            
            # Sentence structure
            avg_sentence_length = word_count / sentence_count if sentence_count else 0
            avg_word_length = sum(len(word) for word in words_clean) / len(words_clean) if words_clean else 0
            
            # Grammar and spelling (simplified without Java)
            grammar_errors = self._check_common_errors(text)
            
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
        except Exception as e:
            # Return basic metrics if analysis fails
            return {
                'char_count': len(text),
                'word_count': len(text.split()),
                'sentence_count': len(text.split('.')),
                'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'lexical_diversity': 0,
                'avg_sentence_length': 0,
                'avg_word_length': 0,
                'grammar_errors': 0,
                'unique_words': 0
            }
    
    def _check_common_errors(self, text):
        """Check for common grammar errors without Java"""
        error_count = 0
        
        # Check for common homophone errors
        text_lower = text.lower()
        
        # Their/there/they're confusion
        if re.search(r'\btheir\b.*\bthey\'re\b', text_lower) or \
           re.search(r'\bthere\b.*\btheir\b', text_lower):
            error_count += 1
        
        # Your/you're confusion
        if re.search(r'\byour\b.*\byou\'re\b', text_lower):
            error_count += 1
        
        # Its/it's confusion
        if re.search(r'\bits\b.*\bit\'s\b', text_lower):
            error_count += 1
        
        # Double words
        double_words = re.findall(r'\b(\w+)\s+\1\b', text_lower)
        error_count += len(double_words)
        
        return error_count
    
    def get_grammar_suggestions(self, text):
        """Get grammar and style suggestions without Java dependency"""
        suggestions = []
        
        # Check for long sentences
        sentences = sent_tokenize(text)
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) > 30:
                suggestions.append({
                    'context': sentence[:50] + '...' if len(sentence) > 50 else sentence,
                    'message': 'Sentence is quite long. Consider breaking it into shorter sentences.',
                    'replacements': ['Try splitting this long sentence'],
                    'offset': text.find(sentence),
                    'error_length': len(sentence)
                })
        
        # Check for passive voice (basic detection)
        passive_patterns = [
            r'\bis\s+\w+ed\b',
            r'\bare\s+\w+ed\b',
            r'\bwas\s+\w+ed\b',
            r'\bwere\s+\w+ed\b',
            r'\bbe\s+\w+ed\b',
            r'\bbeing\s+\w+ed\b',
            r'\bbeen\s+\w+ed\b'
        ]
        
        for pattern in passive_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                suggestions.append({
                    'context': text[max(0, match.start()-20):match.end()+20],
                    'message': 'Possible passive voice detected',
                    'replacements': ['Consider using active voice'],
                    'offset': match.start(),
                    'error_length': match.end() - match.start()
                })
        
        # Check for complex words (more than 3 syllables)
        complex_words = []
        words = word_tokenize(text)
        for word in words:
            if len(word) > 12 and word.isalpha():  # Very long words
                complex_words.append(word)
        
        for word in set(complex_words[:3]):  # Limit to top 3
            suggestions.append({
                'context': f'...{word}...',
                'message': 'Complex word detected',
                'replacements': ['Consider using a simpler alternative'],
                'offset': text.find(word),
                'error_length': len(word)
            })
        
        return suggestions[:10]  # Limit to 10 suggestions