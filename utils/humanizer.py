import re
import random
from nltk.tokenize import sent_tokenize
import language_tool_python
from utils.text_analyzer import TextAnalyzer

class Humanizer:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        self.tool = language_tool_python.LanguageTool('en-US')
        
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
            r'\bleverage\b': ['use', 'make use of']
        }

    def humanize_text(self, text, intensity='medium'):
        """Main humanization function"""
        humanized = text
        
        # Apply transformations based on intensity
        if intensity == 'low':
            humanized = self._apply_basic_humanization(humanized)
        elif intensity == 'medium':
            humanized = self._apply_medium_humanization(humanized)
        else:  # high
            humanized = self._apply_advanced_humanization(humanized)
        
        # Final grammar check and polish
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
                if not sentence.startswith(tuple(s.startswith(' ') for s in starters)):
                    sentence = random.choice(starters) + sentence.lower()
            
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
            'should not': 'shouldn\'t'
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
                if len(sentence.split()) > 25 and random.random() < 0.6:
                    # Split long sentence
                    words = sentence.split()
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
            ' You see,', ' The thing is,', ' Look,'
        ]
        
        words = text.split()
        if len(words) > 50:
            insert_point = random.randint(20, len(words) - 10)
            if random.random() < 0.3:
                words.insert(insert_point, random.choice(conversational_inserts))
                text = ' '.join(words)
        
        return text
    
    def _polish_text(self, text):
        """Final polishing of the text"""
        # Fix any grammar issues introduced
        matches = self.tool.check(text)
        corrected_text = language_tool_python.utils.correct(text, matches)
        
        # Ensure proper capitalization
        sentences = sent_tokenize(corrected_text)
        corrected_sentences = [sentence[0].upper() + sentence[1:] if sentence else sentence 
                             for sentence in sentences]
        
        return ' '.join(corrected_sentences)
    
    def get_humanization_report(self, original_text, humanized_text):
        """Generate a comparison report"""
        original_analysis = self.analyzer.analyze_text(original_text)
        humanized_analysis = self.analyzer.analyze_text(humanized_text)
        
        improvements = {
            'readability_change': humanized_analysis['flesch_reading_ease'] - original_analysis['flesch_reading_ease'],
            'lexical_diversity_change': humanized_analysis['lexical_diversity'] - original_analysis['lexical_diversity'],
            'sentence_variety': abs(humanized_analysis['avg_sentence_length'] - original_analysis['avg_sentence_length']),
            'grammar_improvement': original_analysis['grammar_errors'] - humanized_analysis['grammar_errors']
        }
        
        return {
            'original_analysis': original_analysis,
            'humanized_analysis': humanized_analysis,
            'improvements': improvements
        }