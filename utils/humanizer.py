import re
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from utils.text_analyzer import TextAnalyzer

class UltimateHumanizer:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        self.ai_patterns_db = self._build_ai_patterns_database()
        self.human_signatures = self._build_human_signatures()
        self.quality_preservers = self._build_quality_preservers()
        
    def _build_ai_patterns_database(self):
        return {
            'formal_transitions': {
                'however': ['but', 'though', 'then again', 'that said'],
                'moreover': ['also', 'besides', 'what\'s more', 'plus'],
                'furthermore': ['additionally', 'also', 'plus'],
                'therefore': ['so', 'thus', 'that\'s why'],
                'consequently': ['so', 'as a result', 'because of this'],
                'thus': ['so', 'therefore', 'as a result'],
                'hence': ['so', 'therefore', 'that\'s why'],
            },
            'academic_phrases': {
                'it is important to note': ['keep in mind', 'remember that', 'note that'],
                'it is crucial to': ['we need to', 'we must', 'it\'s vital to'],
                'it is worth noting': ['it\'s worth remembering', 'don\'t forget'],
                'in conclusion': ['to wrap up', 'overall', 'basically'],
                'in summary': ['to sum up', 'basically', 'long story short'],
                'to summarize': ['in short', 'put simply', 'basically'],
            },
            'perfection_markers': {
                'optimal': ['best', 'ideal', 'great'],
                'utilize': ['use', 'work with', 'employ'],
                'facilitate': ['help', 'make easier', 'assist'],
                'implement': ['set up', 'put in place', 'start'],
                'leverage': ['use', 'make use of', 'take advantage of'],
                'commence': ['start', 'begin', 'get going'],
            }
        }
    
    def _build_human_signatures(self):
        return {
            'conversational_starters': [
                'Well,', 'You know,', 'Actually,', 'So,', 'Look,', 
                'Honestly,', 'I mean,', 'Basically,', 'To be honest,'
            ],
            'personal_elements': [
                'I think', 'I believe', 'I feel', 'In my experience', 
                'From what I\'ve seen', 'Personally,', 'The way I see it'
            ],
            'filler_phrases': [
                'kind of', 'sort of', 'you know', 'I think', 
                'in a way', 'more or less', 'pretty much'
            ],
            'contractions': {
                'it is': 'it\'s', 'do not': 'don\'t', 'does not': 'doesn\'t',
                'cannot': 'can\'t', 'will not': 'won\'t', 'have not': 'haven\'t',
                'has not': 'hasn\'t', 'that is': 'that\'s', 'what is': 'what\'s',
                'they are': 'they\'re', 'we are': 'we\'re', 'you are': 'you\'re',
                'I am': 'I\'m', 'he is': 'he\'s', 'she is': 'she\'s'
            },
            'casual_alternatives': {
                'utilize': 'use', 'facilitate': 'help', 'implement': 'set up',
                'optimal': 'best', 'parameters': 'settings', 'leverage': 'use',
                'approximately': 'about', 'subsequently': 'later', 'therefore': 'so',
                'however': 'but', 'moreover': 'also', 'furthermore': 'plus',
            }
        }

    def _build_quality_preservers(self):
        return {
            'sentence_enhancers': [
                "What's interesting is that {sentence}",
                "I've found that {sentence}",
                "From my experience, {sentence}",
                "The thing about {sentence} is that it makes sense",
                "You'll notice that {sentence}",
            ],
            'voice_changers': [
                # Active to passive
                (r'(\w+) (\w+)s? (\w+)', r'The \3 is \2ed by the \1'),
                (r'We (\w+) (\w+)', r'The \2 is \1ed by us'),
                
                # Passive to active  
                (r'(\w+) is (\w+)ed by (\w+)', r'The \3 \2s the \1'),
                (r'(\w+) are (\w+)ed by (\w+)', r'The \3 \2 the \1'),
            ]
        }

    def ultimate_humanize(self, text, intensity='extreme'):
        if not text or len(text.strip()) < 10:
            return text
        
        original_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        humanized_paragraphs = []
        
        for paragraph in original_paragraphs:
            humanized_paragraph = self._balanced_humanize_paragraph(paragraph)
            humanized_paragraphs.append(humanized_paragraph)
        
        return '\n\n'.join(humanized_paragraphs)
    
    def _balanced_humanize_paragraph(self, paragraph):
        sentences = sent_tokenize(paragraph)
        if not sentences:
            return paragraph
        
        transformed_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Apply balanced transformations
            transformed = self._balanced_sentence_transform(sentence, i)
            transformed_sentences.append(transformed)
        
        return ' '.join(transformed_sentences)
    
    def _balanced_sentence_transform(self, sentence, sentence_index):
        if len(sentence.split()) < 4:
            return sentence
        
        original = sentence
        
        # PHASE 1: Remove AI patterns (high probability for AI evasion)
        sentence = self._aggressive_ai_removal(sentence)
        
        # PHASE 2: Add human elements (medium probability for balance)
        sentence = self._balanced_human_injection(sentence, sentence_index)
        
        # PHASE 3: Enhance quality (maintains good writing)
        sentence = self._quality_enhancement(sentence)
        
        # PHASE 4: Change voice (adds variation for AI evasion)
        if random.random() < 0.3:
            sentence = self._change_voice_smart(sentence)
        
        # Final quality check
        if self._is_quality_preserved(original, sentence):
            return sentence
        else:
            # Fallback: minimal transformation that preserves quality
            return self._minimal_safe_transform(original)
    
    def _aggressive_ai_removal(self, sentence):
        """Aggressive AI pattern removal for maximum evasion"""
        # High probability replacements for AI evasion
        for category, patterns in self.ai_patterns_db.items():
            for formal, alternatives in patterns.items():
                if re.search(r'\b' + re.escape(formal) + r'\b', sentence, re.IGNORECASE):
                    replacement = random.choice(alternatives)
                    sentence = re.sub(r'\b' + re.escape(formal) + r'\b', replacement, sentence, re.IGNORECASE)
        
        # Aggressive contraction application (90% probability for AI evasion)
        for formal, contraction in self.human_signatures['contractions'].items():
            if random.random() < 0.9:
                sentence = re.sub(r'\b' + formal + r'\b', contraction, sentence, re.IGNORECASE)
        
        return sentence
    
    def _balanced_human_injection(self, sentence, sentence_index):
        """Balanced human element injection"""
        words = sentence.split()
        
        # Conversational starters (40% probability - balanced)
        if sentence_index == 0 and random.random() < 0.4:
            starters = self.human_signatures['conversational_starters']
            if not any(sentence.startswith(s.strip()) for s in starters):
                starter = random.choice(starters)
                sentence = starter + ' ' + sentence[0].lower() + sentence[1:]
        
        # Personal elements (30% probability - subtle but effective)
        if random.random() < 0.3 and len(words) > 6:
            personal = random.choice(self.human_signatures['personal_elements'])
            insert_point = random.randint(1, min(3, len(words) - 3))
            words.insert(insert_point, personal)
            sentence = ' '.join(words)
        
        # Filler phrases (20% probability - natural human touch)
        if random.random() < 0.2 and len(words) > 8:
            filler = random.choice(self.human_signatures['filler_phrases'])
            insert_point = random.randint(2, len(words) - 3)
            words.insert(insert_point, filler)
            sentence = ' '.join(words)
        
        return sentence
    
    def _quality_enhancement(self, sentence):
        """Enhance sentence quality while maintaining meaning"""
        words = sentence.split()
        
        # Occasionally use sentence enhancers (25% probability)
        if random.random() < 0.25 and len(words) > 5:
            enhancer = random.choice(self.quality_preservers['sentence_enhancers'])
            try:
                enhanced = enhancer.format(sentence=sentence[0].lower() + sentence[1:])
                if len(enhanced.split()) <= len(words) + 8:  # Don't make too long
                    return enhanced
            except:
                pass
        
        return sentence
    
    def _change_voice_smart(self, sentence):
        """Smart voice changing that preserves meaning"""
        original = sentence
        
        # Try active to passive
        for pattern, replacement in self.quality_preservers['voice_changers']:
            if re.search(pattern, sentence, re.IGNORECASE):
                try:
                    new_sentence = re.sub(pattern, replacement, sentence, re.IGNORECASE)
                    # Check if the transformation makes sense
                    if (self._is_meaning_preserved(original, new_sentence) and 
                        len(new_sentence.split()) >= len(original.split()) - 2):  # Don't shorten too much
                        return new_sentence.capitalize()
                except:
                    continue
        
        return sentence
    
    def _is_quality_preserved(self, original, transformed):
        """Check if quality and meaning are preserved"""
        if len(transformed.split()) < len(original.split()) * 0.7:  # Not too short
            return False
        
        if len(transformed.split()) > len(original.split()) * 1.5:  # Not too long
            return False
        
        return self._is_meaning_preserved(original, transformed)
    
    def _is_meaning_preserved(self, original, transformed):
        """Check if meaning is preserved"""
        original_words = set(original.lower().split())
        transformed_words = set(transformed.lower().split())
        
        common_words = original_words.intersection(transformed_words)
        similarity = len(common_words) / max(len(original_words), len(transformed_words))
        
        return similarity > 0.5  # Reasonable similarity threshold
    
    def _minimal_safe_transform(self, sentence):
        """Apply minimal safe transformations as fallback"""
        # Only apply contractions and basic AI pattern removal
        for formal, contraction in self.human_signatures['contractions'].items():
            if random.random() < 0.7:
                sentence = re.sub(r'\b' + formal + r'\b', contraction, sentence, re.IGNORECASE)
        
        return sentence
    
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