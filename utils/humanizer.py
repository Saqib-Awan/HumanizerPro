import re
import random
from nltk.tokenize import sent_tokenize
from utils.text_analyzer import TextAnalyzer

class Humanizer:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        
        # Human-like variations
        self.transition_alternatives = {
            'however': ['but', 'though', 'on the other hand', 'yet'],
            'moreover': ['also', 'besides', 'what\'s more', 'furthermore'],
            'furthermore': ['additionally', 'moreover', 'also'],
            'additionally': ['plus', 'as well', 'on top of that'],
            'therefore': ['so', 'thus', 'as a result'],
            'consequently': ['as a result', 'therefore', 'so'],
            'in conclusion': ['to sum up', 'overall', 'all in all'],
            'significantly': ['importantly', 'notably', 'crucially']
        }
        
        self.formal_alternatives = {
            r'\bit is important to note\b': ['keep in mind', 'remember that', 'note that'],
            r'\bit is crucial to\b': ['it\'s vital to', 'we need to', 'we must'],
            r'\butilize\b': ['use'],
            r'\bfacilitate\b': ['help', 'make easier'],
            r'\bimplement\b': ['put in place', 'set up'],
            r'\boptimal\b': ['best', 'ideal'],
            r'\bparameters\b': ['settings', 'limits'],
            r'\bleverage\b': ['use', 'make use of'],
            r'\bcommence\b': ['start', 'begin'],
            r'\bterminate\b': ['end', 'stop'],
            r'\bapproximately\b': ['about', 'around'],
            r'\bsubsequently\b': ['later', 'afterwards'],
            r'\bconsequently\b': ['so', 'as a result']
        }

    def humanize_text(self, text, intensity='medium'):
        """Main humanization function"""
        if not text or len(text.strip()) < 10:
            return text
            
        humanized = text
        
        # Apply transformations based on intensity
        if intensity == 'low':
            humanized = self._apply_basic_humanization(humanized)
        elif intensity == 'medium':
            humanized = self._apply_medium_humanization(humanized)
        else:  # high
            humanized = self._apply_advanced_humanization(humanized)
        
        # Final polish
        humanized = self._polish_text(humanized)
        
        return humanized
    
    def _apply_basic_humanization(self, text):
        """Basic level humanization"""
        # Replace AI-specific phrases
        for pattern, alternatives in self.formal_alternatives.items():
            if re.search(pattern, text, re.IGNORECASE):
                replacement = random.choice(alternatives)
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _apply_medium_humanization(self, text):
        """Medium level humanization"""
        text = self._apply_basic_humanization(text)
        
        # Vary sentence structure
        sentences = sent_tokenize(text)
        humanized_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Occasionally start with conversational phrases
            if i > 0 and random.random() < 0.3:
                starters = ['Well, ', 'You know, ', 'Actually, ', 'So, ']
                if not any(sentence.startswith(s.strip()) for s in starters):
                    sentence = random.choice(starters) + sentence[0].lower() + sentence[1:]
            
            humanized_sentences.append(sentence)
        
        text = ' '.join(humanized_sentences)
        
        # Add occasional contractions
        contractions = {
            'it is': 'it\'s',
            'do not': 'don\'t',
            'cannot': 'can\'t',
            'will not': 'won\'t',
            'have not': 'haven\'t',
            'has not': 'hasn\'t',
            'had not': 'hadn\'t',
            'would not': 'wouldn\'t',
            'should not': 'shouldn\'t',
            'could not': 'couldn\'t',
            'that is': 'that\'s',
            'what is': 'what\'s'
        }
        
        for formal, contraction in contractions.items():
            if random.random() < 0.4:
                text = re.sub(r'\b' + formal + r'\b', contraction, text, flags=re.IGNORECASE)
        
        return text
    
    def _apply_advanced_humanization(self, text):
        """Advanced level humanization"""
        text = self._apply_medium_humanization(text)
        
        # More aggressive transformations
        sentences = sent_tokenize(text)
        
        if len(sentences) > 2:
            # Occasionally break long sentences
            modified_sentences = []
            for sentence in sentences:
                words = sentence.split()
                if len(words) > 25 and random.random() < 0.6:
                    # Split long sentence at natural break point
                    split_points = [i for i, word in enumerate(words) 
                                  if word.endswith(',') or word.endswith(';')]
                    if split_points:
                        split_point = random.choice(split_points) + 1
                    else:
                        split_point = random.randint(15, 20)
                    
                    part1 = ' '.join(words[:split_point])
                    part2 = ' '.join(words[split_point:])
                    modified_sentences.extend([part1 + '.', part2.capitalize()])
                else:
                    modified_sentences.append(sentence)
            
            text = ' '.join(modified_sentences)
        
        # Add more conversational elements
        conversational_inserts = [
            ' I mean,', ' Basically,', ' Honestly,', ' Actually,',
            ' You see,', ' The thing is,', ' Look,', ' Well,'
        ]
        
        words = text.split()
        if len(words) > 50 and random.random() < 0.3:
            insert_point = random.randint(20, len(words) - 10)
            insert_phrase = random.choice(conversational_inserts)
            words.insert(insert_point, insert_phrase)
            text = ' '.join(words)
        
        # Add occasional filler words for natural flow
        filler_words = ['like', 'you know', 'I think', 'sort of', 'kind of']
        if len(words) > 30 and random.random() < 0.2:
            insert_point = random.randint(10, len(words) - 5)
            filler = random.choice(filler_words)
            words.insert(insert_point, filler)
            text = ' '.join(words)
        
        return text
    
    def _polish_text(self, text):
        """Final polishing of the text"""
        # Ensure proper capitalization after sentence splits
        sentences = sent_tokenize(text)
        corrected_sentences = []
        
        for sentence in sentences:
            if sentence and sentence[0].isalpha():
                sentence = sentence[0].upper() + sentence[1:]
            corrected_sentences.append(sentence)
        
        return ' '.join(corrected_sentences)
    
    def get_humanization_report(self, original_text, humanized_text):
        """Generate a comparison report"""
        original_analysis = self.analyzer.analyze_text(original_text)
        humanized_analysis = self.analyzer.analyze_text(humanized_text)
        
        improvements = {
            'readability_change': humanized_analysis['flesch_reading_ease'] - original_analysis['flesch_reading_ease'],
            'lexical_diversity_change': humanized_analysis['lexical_diversity'] - original_analysis['lexical_diversity'],
            'sentence_variety': abs(humanized_analysis['avg_sentence_length'] - original_analysis['avg_sentence_length']),
            'word_count_change': humanized_analysis['word_count'] - original_analysis['word_count']
        }
        
        return {
            'original_analysis': original_analysis,
            'humanized_analysis': humanized_analysis,
            'improvements': improvements
        }