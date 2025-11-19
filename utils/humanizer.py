import re
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from utils.text_analyzer import TextAnalyzer

class UltimateHumanizer:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        self.ai_patterns_db = self._build_ai_patterns_database()
        self.human_signatures = self._build_human_signatures()
        
    def _build_ai_patterns_database(self):
        return {
            'formal_transitions': {
                'however': ['but', 'though', 'then again'],
                'moreover': ['also', 'besides', 'plus'],
                'furthermore': ['additionally', 'also'],
                'therefore': ['so', 'thus'],
                'consequently': ['so', 'as a result'],
            },
            'academic_phrases': {
                'it is important to note': ['keep in mind', 'remember that'],
                'it is crucial to': ['we need to', 'it\'s vital to'],
                'in conclusion': ['to wrap up', 'overall'],
            },
            'perfection_markers': {
                'optimal': ['best', 'ideal'],
                'utilize': ['use', 'work with'],
                'facilitate': ['help', 'assist'],
            }
        }
    
    def _build_human_signatures(self):
        return {
            'conversational_starters': ['Well,', 'You know,', 'Actually,'],
            'personal_elements': ['I think', 'I believe', 'In my experience'],
            'contractions': {
                'it is': 'it\'s', 'do not': 'don\'t', 'does not': 'doesn\'t',
                'cannot': 'can\'t', 'will not': 'won\'t', 'that is': 'that\'s',
            }
        }

    def ultimate_humanize(self, text, intensity='extreme'):
        if not text or len(text.strip()) < 10:
            return text
        
        original_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        humanized_paragraphs = []
        
        for paragraph in original_paragraphs:
            humanized_paragraph = self._smart_humanize_paragraph(paragraph)
            humanized_paragraphs.append(humanized_paragraph)
        
        return '\n\n'.join(humanized_paragraphs)
    
    def _smart_humanize_paragraph(self, paragraph):
        sentences = sent_tokenize(paragraph)
        if not sentences:
            return paragraph
        
        transformed_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Apply smart transformations that preserve meaning
            transformed = self._smart_sentence_transform(sentence, i)
            transformed_sentences.append(transformed)
        
        return ' '.join(transformed_sentences)
    
    def _smart_sentence_transform(self, sentence, sentence_index):
        if len(sentence.split()) < 4:
            return sentence
        
        original = sentence
        
        # 1. Remove AI patterns (preserves meaning)
        sentence = self._remove_ai_patterns(sentence)
        
        # 2. Change voice (active/passive) - maintains meaning
        if random.random() < 0.4:
            sentence = self._change_voice(sentence)
        
        # 3. Add subtle human elements
        sentence = self._add_subtle_human_elements(sentence, sentence_index)
        
        # 4. Ensure meaning preservation
        if self._is_meaning_preserved(original, sentence):
            return sentence
        else:
            return original
    
    def _remove_ai_patterns(self, sentence):
        # Replace formal words with casual alternatives
        for category, patterns in self.ai_patterns_db.items():
            for formal, alternatives in patterns.items():
                if re.search(r'\b' + re.escape(formal) + r'\b', sentence, re.IGNORECASE):
                    replacement = random.choice(alternatives)
                    sentence = re.sub(r'\b' + re.escape(formal) + r'\b', replacement, sentence, re.IGNORECASE)
        return sentence
    
    def _change_voice(self, sentence):
        """Change between active and passive voice while preserving meaning"""
        words = sentence.lower().split()
        
        # Active to Passive
        if random.random() < 0.5:
            # Pattern: "Subject verb object" -> "Object is verbed by subject"
            if len(words) >= 3:
                # Simple transformation for common patterns
                active_patterns = [
                    (r'(\w+)s? (\w+)s? (\w+)', r'\3 is \2ed by \1'),
                    (r'(\w+) (\w+)s? (\w+)', r'\3 is \2ed by \1'),
                ]
                
                for pattern, replacement in active_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        try:
                            new_sentence = re.sub(pattern, replacement, sentence, re.IGNORECASE)
                            if self._is_meaning_preserved(sentence, new_sentence):
                                return new_sentence.capitalize()
                        except:
                            pass
        
        # Passive to Active  
        else:
            # Pattern: "Object is verbed by subject" -> "Subject verbs object"
            passive_patterns = [
                (r'(\w+) is (\w+)ed by (\w+)', r'\3 \2s \1'),
                (r'(\w+) are (\w+)ed by (\w+)', r'\3 \2 \1'),
            ]
            
            for pattern, replacement in passive_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    try:
                        new_sentence = re.sub(pattern, replacement, sentence, re.IGNORECASE)
                        if self._is_meaning_preserved(sentence, new_sentence):
                            return new_sentence.capitalize()
                    except:
                        pass
        
        return sentence
    
    def _add_subtle_human_elements(self, sentence, sentence_index):
        words = sentence.split()
        
        # Add conversational starter (only if it makes sense)
        if sentence_index == 0 and random.random() < 0.3:
            starters = self.human_signatures['conversational_starters']
            if not any(sentence.startswith(s.strip()) for s in starters):
                starter = random.choice(starters)
                sentence = starter + ' ' + sentence[0].lower() + sentence[1:]
        
        # Add personal elements sparingly
        if random.random() < 0.2 and len(words) > 6:
            personal = random.choice(self.human_signatures['personal_elements'])
            insert_point = random.randint(1, min(2, len(words) - 3))
            words.insert(insert_point, personal)
            sentence = ' '.join(words)
        
        # Apply contractions moderately
        for formal, contraction in self.human_signatures['contractions'].items():
            if random.random() < 0.4:
                sentence = re.sub(r'\b' + formal + r'\b', contraction, sentence, re.IGNORECASE)
        
        return sentence
    
    def _is_meaning_preserved(self, original, transformed):
        """Check if the transformed sentence preserves the original meaning"""
        original_words = set(original.lower().split())
        transformed_words = set(transformed.lower().split())
        
        # If too many words are different, meaning might be lost
        common_words = original_words.intersection(transformed_words)
        similarity = len(common_words) / max(len(original_words), len(transformed_words))
        
        return similarity > 0.6  # At least 60% word overlap
    
    def humanize_text(self, text, intensity='extreme'):
        return self.ultimate_humanize(text, intensity)
    
    def get_humanization_report(self, original_text, humanized_text):
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