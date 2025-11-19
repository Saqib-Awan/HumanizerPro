import re
import random
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from utils.text_analyzer import TextAnalyzer

class Humanizer:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        
        # Advanced human-like variations
        self.transition_alternatives = {
            'however': ['but', 'though', 'on the other hand', 'yet', 'then again', 'that said'],
            'moreover': ['also', 'besides', 'what\'s more', 'furthermore', 'on top of that'],
            'furthermore': ['additionally', 'moreover', 'also', 'plus'],
            'additionally': ['plus', 'as well', 'on top of that', 'not to mention'],
            'therefore': ['so', 'thus', 'as a result', 'because of this'],
            'consequently': ['as a result', 'therefore', 'so', 'that\'s why'],
            'in conclusion': ['to sum up', 'overall', 'all in all', 'basically', 'long story short'],
            'significantly': ['importantly', 'notably', 'crucially', 'seriously'],
            'nevertheless': ['anyway', 'still', 'even so', 'regardless'],
            'subsequently': ['later', 'afterwards', 'then', 'next']
        }
        
        self.formal_alternatives = {
            r'\bit is important to note\b': ['keep in mind', 'remember that', 'note that', 'don\'t forget'],
            r'\bit is crucial to\b': ['it\'s vital to', 'we need to', 'we must', 'we have to'],
            r'\butilize\b': ['use', 'work with', 'employ'],
            r'\bfacilitate\b': ['help', 'make easier', 'assist with'],
            r'\bimplement\b': ['put in place', 'set up', 'start using'],
            r'\boptimal\b': ['best', 'ideal', 'perfect', 'right'],
            r'\bparameters\b': ['settings', 'limits', 'boundaries', 'rules'],
            r'\bleverage\b': ['use', 'make use of', 'take advantage of'],
            r'\bcommence\b': ['start', 'begin', 'kick off'],
            r'\bterminate\b': ['end', 'stop', 'finish'],
            r'\bapproximately\b': ['about', 'around', 'roughly', 'more or less'],
            r'\bsubsequently\b': ['later', 'afterwards', 'then'],
            r'\bconsequently\b': ['so', 'as a result', 'because of this'],
            r'\bnevertheless\b': ['anyway', 'still', 'even so'],
            r'\bnotwithstanding\b': ['despite', 'even with', 'regardless of'],
            r'\baccordingly\b': ['so', 'therefore', 'thus'],
            r'\bfurthermore\b': ['also', 'besides', 'what\'s more'],
            r'\bhence\b': ['so', 'therefore', 'thus'],
            r'\bthus\b': ['so', 'therefore', 'as a result']
        }

        # Human writing patterns
        self.conversational_starters = [
            'Well, ', 'You know, ', 'Actually, ', 'So, ', 'Look, ', 'Honestly, ',
            'I mean, ', 'Basically, ', 'The thing is, ', 'To be honest, ',
            'Frankly, ', 'Seriously, ', 'No kidding, ', 'Believe it or not, '
        ]
        
        self.filler_phrases = [
            'kind of', 'sort of', 'you know', 'I think', 'I believe', 'I feel like',
            'in a way', 'more or less', 'to some extent', 'pretty much',
            'basically', 'essentially', 'virtually', 'practically'
        ]
        
        self.human_imperfections = [
            'um', 'ah', 'er', 'like', 'right', 'okay', 'so', 'well',
            'anyway', 'anyhow', 'moving on', 'back to'
        ]

    def humanize_text(self, text, intensity='medium'):
        """Advanced humanization with multiple techniques"""
        if not text or len(text.strip()) < 10:
            return text
            
        humanized = text
        
        # Apply all humanization techniques
        humanized = self._remove_ai_patterns(humanized)
        humanized = self._vary_sentence_structure(humanized)
        humanized = self._add_conversational_elements(humanized)
        humanized = self._introduce_imperfections(humanized)
        humanized = self._vary_vocabulary(humanized)
        humanized = self._adjust_formality(humanized)
        
        # Intensity-based transformations
        if intensity == 'medium':
            humanized = self._apply_medium_transformations(humanized)
        elif intensity == 'high':
            humanized = self._apply_advanced_transformations(humanized)
        
        humanized = self._final_polish(humanized)
        
        return humanized
    
    def _remove_ai_patterns(self, text):
        """Remove common AI writing patterns"""
        # Replace formal transitions
        for pattern, alternatives in self.transition_alternatives.items():
            if re.search(r'\b' + pattern + r'\b', text, re.IGNORECASE):
                replacement = random.choice(alternatives)
                text = re.sub(r'\b' + pattern + r'\b', replacement, text, flags=re.IGNORECASE)
        
        # Replace formal phrases
        for pattern, alternatives in self.formal_alternatives.items():
            if re.search(pattern, text, re.IGNORECASE):
                replacement = random.choice(alternatives)
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _vary_sentence_structure(self, text):
        """Vary sentence length and structure"""
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return text
            
        varied_sentences = []
        
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            
            # Occasionally combine short sentences
            if (i > 0 and len(words) < 8 and random.random() < 0.3 and 
                len(varied_sentences) > 0):
                last_sentence = varied_sentences.pop()
                combined = last_sentence + ' ' + sentence.lower()
                varied_sentences.append(combined)
            # Occasionally split long sentences
            elif len(words) > 25 and random.random() < 0.6:
                split_point = self._find_natural_break(words)
                if split_point:
                    part1 = ' '.join(words[:split_point])
                    part2 = ' '.join(words[split_point:])
                    varied_sentences.append(part1 + '.')
                    varied_sentences.append(part2.capitalize())
                else:
                    varied_sentences.append(sentence)
            else:
                varied_sentences.append(sentence)
        
        return ' '.join(varied_sentences)
    
    def _find_natural_break(self, words):
        """Find natural break points in sentences"""
        break_points = []
        for i, word in enumerate(words):
            if word.endswith(',') or word.endswith(';') or word in ['and', 'but', 'or']:
                if 10 <= i <= len(words) - 10:  # Ensure reasonable split
                    break_points.append(i + 1)
        
        return random.choice(break_points) if break_points else None
    
    def _add_conversational_elements(self, text):
        """Add conversational phrases and contractions"""
        sentences = sent_tokenize(text)
        conversational_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Add conversational starters occasionally
            if i > 0 and random.random() < 0.4:
                starter = random.choice(self.conversational_starters)
                if not any(sentence.startswith(s.strip()) for s in self.conversational_starters):
                    sentence = starter + sentence[0].lower() + sentence[1:]
            
            conversational_sentences.append(sentence)
        
        text = ' '.join(conversational_sentences)
        
        # Add contractions
        contractions = {
            'it is': 'it\'s', 'do not': 'don\'t', 'does not': 'doesn\'t',
            'cannot': 'can\'t', 'will not': 'won\'t', 'have not': 'haven\'t',
            'has not': 'hasn\'t', 'had not': 'hadn\'t', 'would not': 'wouldn\'t',
            'should not': 'shouldn\'t', 'could not': 'couldn\'t', 'that is': 'that\'s',
            'what is': 'what\'s', 'who is': 'who\'s', 'where is': 'where\'s',
            'when is': 'when\'s', 'why is': 'why\'s', 'how is': 'how\'s',
            'there is': 'there\'s', 'here is': 'here\'s', 'they are': 'they\'re',
            'we are': 'we\'re', 'you are': 'you\'re', 'I am': 'I\'m'
        }
        
        for formal, contraction in contractions.items():
            if random.random() < 0.7:  # Higher probability for contractions
                text = re.sub(r'\b' + formal + r'\b', contraction, text, flags=re.IGNORECASE)
        
        return text
    
    def _introduce_imperfections(self, text):
        """Introduce human-like imperfections"""
        words = text.split()
        if len(words) < 20:
            return text
            
        # Add occasional filler words
        if random.random() < 0.3:
            insert_point = random.randint(5, len(words) - 5)
            filler = random.choice(self.filler_phrases)
            words.insert(insert_point, filler)
        
        # Add conversational imperfections occasionally
        if random.random() < 0.2:
            insert_point = random.randint(10, len(words) - 10)
            imperfection = random.choice(self.human_imperfections)
            words.insert(insert_point, imperfection)
        
        # Occasionally repeat words for emphasis (human-like)
        if random.random() < 0.15 and len(words) > 15:
            repeat_point = random.randint(5, len(words) - 5)
            word_to_repeat = words[repeat_point]
            if len(word_to_repeat) > 3 and word_to_repeat.isalpha():
                words.insert(repeat_point + 1, word_to_repeat)
        
        return ' '.join(words)
    
    def _vary_vocabulary(self, text):
        """Vary vocabulary to avoid repetition"""
        words = word_tokenize(text)
        if len(words) < 10:
            return text
            
        # Simple synonym replacement for common words
        synonyms = {
            'good': ['great', 'nice', 'excellent', 'awesome', 'fantastic'],
            'bad': ['terrible', 'awful', 'horrible', 'poor', 'lousy'],
            'big': ['large', 'huge', 'enormous', 'massive', 'giant'],
            'small': ['tiny', 'little', 'miniature', 'compact', 'petite'],
            'important': ['crucial', 'vital', 'essential', 'key', 'critical'],
            'interesting': ['fascinating', 'intriguing', 'compelling', 'engaging'],
            'different': ['various', 'diverse', 'assorted', 'varied'],
            'many': ['numerous', 'multiple', 'countless', 'several']
        }
        
        new_words = []
        for word in words:
            if word.lower() in synonyms and random.random() < 0.3:
                new_words.append(random.choice(synonyms[word.lower()]))
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def _adjust_formality(self, text):
        """Adjust formality level to be more casual"""
        # Replace passive voice with active voice when possible
        passive_patterns = [
            (r'is\s+(\w+ed)\s+by', r'\\1s'),
            (r'are\s+(\w+ed)\s+by', r'\\1'),
            (r'was\s+(\w+ed)\s+by', r'\\1ed'),
            (r'were\s+(\w+ed)\s+by', r'\\1ed')
        ]
        
        for pattern, replacement in passive_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Add more casual phrasing
        casual_replacements = {
            r'in order to': 'to',
            r'with regard to': 'about',
            r'with respect to': 'about',
            r'due to the fact that': 'because',
            r'in the event that': 'if',
            r'at this point in time': 'now',
            r'prior to': 'before',
            r'subsequent to': 'after'
        }
        
        for formal, casual in casual_replacements.items():
            text = re.sub(formal, casual, text, flags=re.IGNORECASE)
        
        return text
    
    def _apply_medium_transformations(self, text):
        """Medium intensity transformations"""
        # Add more sentence variety
        sentences = sent_tokenize(text)
        if len(sentences) > 2:
            # Occasionally use rhetorical questions
            if random.random() < 0.3:
                mid_point = len(sentences) // 2
                question = "Right? " if random.random() < 0.5 else "You know what I mean? "
                sentences.insert(mid_point, question)
        
        return ' '.join(sentences)
    
    def _apply_advanced_transformations(self, text):
        """Advanced intensity transformations"""
        sentences = sent_tokenize(text)
        advanced_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            
            # More aggressive sentence splitting
            if len(words) > 20 and random.random() < 0.8:
                splits = self._find_multiple_breaks(words)
                if splits:
                    current_start = 0
                    for split_point in splits:
                        part = ' '.join(words[current_start:split_point])
                        advanced_sentences.append(part + '.')
                        current_start = split_point
                    if current_start < len(words):
                        part = ' '.join(words[current_start:])
                        advanced_sentences.append(part.capitalize())
                else:
                    advanced_sentences.append(sentence)
            else:
                advanced_sentences.append(sentence)
        
        text = ' '.join(advanced_sentences)
        
        # Add more conversational elements
        words = text.split()
        if len(words) > 30:
            # Add multiple conversational elements
            for _ in range(random.randint(1, 3)):
                if len(words) > 10:
                    insert_point = random.randint(5, len(words) - 5)
                    element = random.choice(self.filler_phrases + self.human_imperfections)
                    words.insert(insert_point, element)
        
        return ' '.join(words)
    
    def _find_multiple_breaks(self, words):
        """Find multiple natural break points"""
        breaks = []
        for i, word in enumerate(words):
            if (word.endswith(',') or word.endswith(';') or 
                word in ['and', 'but', 'or', 'however', 'although']):
                if 8 <= i <= len(words) - 8:
                    breaks.append(i + 1)
        
        # Return up to 2 break points for longer sentences
        return breaks[:2] if len(breaks) > 2 else breaks
    
    def _final_polish(self, text):
        """Final polishing while maintaining human feel"""
        # Ensure proper capitalization
        sentences = sent_tokenize(text)
        polished_sentences = []
        
        for sentence in sentences:
            if sentence and sentence[0].isalpha():
                # Occasionally don't capitalize to maintain conversational feel
                if random.random() < 0.1 and len(polished_sentences) > 0:
                    sentence = sentence[0].lower() + sentence[1:]
                else:
                    sentence = sentence[0].upper() + sentence[1:]
            polished_sentences.append(sentence)
        
        text = ' '.join(polished_sentences)
        
        # Remove any double spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def get_humanization_report(self, original_text, humanized_text):
        """Generate a comparison report"""
        original_analysis = self.analyzer.analyze_text(original_text)
        humanized_analysis = self.analyzer.analyze_text(humanized_text)
        
        improvements = {
            'readability_change': humanized_analysis['flesch_reading_ease'] - original_analysis['flesch_reading_ease'],
            'lexical_diversity_change': humanized_analysis['lexical_diversity'] - original_analysis['lexical_diversity'],
            'sentence_variety': abs(humanized_analysis['avg_sentence_length'] - original_analysis['avg_sentence_length']),
            'word_count_change': humanized_analysis['word_count'] - original_analysis['word_count'],
            'grammar_improvement': original_analysis['grammar_errors'] - humanized_analysis['grammar_errors']
        }
        
        return {
            'original_analysis': original_analysis,
            'humanized_analysis': humanized_analysis,
            'improvements': improvements
        }