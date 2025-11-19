import re
import random
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import language_tool_python
from utils.text_analyzer import TextAnalyzer

class UltraHumanizer:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        
        # Advanced transformation databases
        self.ai_patterns_db = self._build_ai_patterns_database()
        self.human_patterns_db = self._build_human_patterns_database()
        self.synonym_chains = self._build_synonym_chains()
        self.sentence_blueprints = self._build_sentence_blueprints()
        
    def _build_ai_patterns_database(self):
        """Comprehensive database of AI writing patterns"""
        return {
            'formal_transitions': [
                r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
                r'\bconsequently\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
                r'\bnevertheless\b', r'\bnonetheless\b', r'\bnotwithstanding\b'
            ],
            'academic_phrases': [
                r'\bit is important to note\b', r'\bit is crucial to\b', r'\bit is worth noting\b',
                r'\bit should be emphasized\b', r'\bit is evident that\b', r'\bfrom this perspective\b',
                r'\bin this context\b', r'\bupon careful analysis\b', r'\bit becomes apparent\b'
            ],
            'structural_patterns': [
                r'\bin conclusion\b', r'\bin summary\b', r'\bto summarize\b', r'\boverall\b',
                r'\bas a result\b', r'\bin essence\b', r'\bfundamentally\b'
            ],
            'perfection_indicators': [
                r'\boptimal\b', r'\bmaximize\b', r'\bminimize\b', r'\befficient\b',
                r'\beffectively\b', r'\bsignificantly\b', r'\bconsiderably\b'
            ]
        }
    
    def _build_human_patterns_database(self):
        """Database of human writing characteristics"""
        return {
            'conversational_starters': [
                'Well,', 'You know,', 'Actually,', 'So,', 'Look,', 'Honestly,',
                'I mean,', 'Basically,', 'The thing is,', 'To be honest,',
                'Frankly,', 'Seriously,', 'No kidding,', 'Believe it or not,',
                'Truth be told,', 'If you ask me,', 'In my opinion,'
            ],
            'filler_phrases': [
                'kind of', 'sort of', 'you know', 'I think', 'I believe', 
                'I feel like', 'in a way', 'more or less', 'to some extent',
                'pretty much', 'basically', 'essentially', 'like I said',
                'as I mentioned', 'going back to', 'anyway'
            ],
            'imperfections': [
                'um', 'ah', 'er', 'like', 'right', 'okay', 'well', 'anyway',
                'anyhow', 'so yeah', 'you see', 'I guess', 'supposedly'
            ],
            'contractions': {
                'it is': 'it\'s', 'do not': 'don\'t', 'does not': 'doesn\'t',
                'cannot': 'can\'t', 'will not': 'won\'t', 'have not': 'haven\'t',
                'has not': 'hasn\'t', 'had not': 'hadn\'t', 'would not': 'wouldn\'t',
                'should not': 'shouldn\'t', 'could not': 'couldn\'t', 'that is': 'that\'s',
                'what is': 'what\'s', 'who is': 'who\'s', 'where is': 'where\'s',
                'there is': 'there\'s', 'here is': 'here\'s', 'they are': 'they\'re',
                'we are': 'we\'re', 'you are': 'you\'re', 'I am': 'I\'m',
                'he is': 'he\'s', 'she is': 'she\'s', 'it would': 'it\'d',
                'that would': 'that\'d', 'I would': 'I\'d', 'you would': 'you\'d'
            },
            'casual_alternatives': {
                'utilize': 'use', 'facilitate': 'help', 'implement': 'set up',
                'optimal': 'best', 'parameters': 'settings', 'leverage': 'use',
                'commence': 'start', 'terminate': 'end', 'approximately': 'about',
                'subsequently': 'later', 'consequently': 'so', 'therefore': 'so',
                'however': 'but', 'moreover': 'also', 'furthermore': 'plus',
                'additionally': 'also', 'thus': 'so', 'hence': 'so'
            }
        }
    
    def _build_synonym_chains(self):
        """Advanced synonym chains for vocabulary variation"""
        return {
            'important': ['crucial', 'vital', 'key', 'essential', 'critical', 'major', 'significant'],
            'good': ['great', 'excellent', 'awesome', 'fantastic', 'wonderful', 'terrific', 'amazing'],
            'bad': ['terrible', 'awful', 'horrible', 'poor', 'lousy', 'dreadful', 'unfortunate'],
            'big': ['large', 'huge', 'enormous', 'massive', 'giant', 'substantial', 'considerable'],
            'small': ['tiny', 'little', 'miniature', 'compact', 'petite', 'modest', 'limited'],
            'show': ['demonstrate', 'illustrate', 'reveal', 'display', 'exhibit', 'present'],
            'help': ['assist', 'aid', 'support', 'facilitate', 'guide', 'advise'],
            'change': ['alter', 'modify', 'adjust', 'transform', 'adapt', 'revise'],
            'make': ['create', 'produce', 'generate', 'develop', 'construct', 'build'],
            'use': ['utilize', 'employ', 'apply', 'operate', 'work with', 'handle']
        }
    
    def _build_sentence_blueprints(self):
        """Human-like sentence structure templates"""
        return [
            # Simple declarative
            "{subject} {verb} {object}",
            # With adverb
            "{subject} {adverb} {verb} {object}",
            # With preposition
            "{subject} {verb} {object} {preposition} {context}",
            # Question form
            "Why does {subject} {verb} {object}?",
            # Exclamation
            "It's amazing how {subject} {verb} {object}!",
            # Conversational
            "You know, {subject} really {verb} {object}",
            # Personal
            "I think {subject} {verb} {object}",
            # Comparative
            "When {subject} {verb} {object}, it's like {comparison}"
        ]

    def ultra_humanize(self, text, intensity='extreme'):
        """Ultra-advanced humanization that beats all AI detectors"""
        if not text or len(text.strip()) < 10:
            return text
        
        # Multi-stage transformation pipeline
        stages = [
            self._stage1_deconstruct_ai_patterns,
            self._stage2_rewrite_sentences,
            self._stage3_vocabulary_reshuffle,
            self._stage4_add_human_elements,
            self._stage5_structural_randomization,
            self._stage6_conversational_weaving,
            self._stage7_imperfection_injection,
            self._stage8_final_human_touch
        ]
        
        humanized = text
        for stage in stages:
            humanized = stage(humanized, intensity)
        
        return humanized
    
    def _stage1_deconstruct_ai_patterns(self, text, intensity):
        """Remove all AI fingerprints"""
        # Remove formal transitions
        for pattern in self.ai_patterns_db['formal_transitions']:
            alternatives = ['but', 'and', 'so', 'also', 'plus', 'then', 'now']
            replacement = random.choice(alternatives)
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Replace academic phrases
        for pattern in self.ai_patterns_db['academic_phrases']:
            casual_versions = [
                'keep in mind that', 'remember that', 'note that', 
                'don\'t forget', 'it\'s worth remembering'
            ]
            replacement = random.choice(casual_versions)
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Eliminate structural patterns
        for pattern in self.ai_patterns_db['structural_patterns']:
            text = re.sub(pattern, '', text, flags=re.IGNECASE)
        
        return text
    
    def _stage2_rewrite_sentences(self, text, intensity):
        """Completely rewrite sentence structures"""
        sentences = sent_tokenize(text)
        rewritten_sentences = []
        
        for sentence in sentences:
            if random.random() < 0.8:  # High probability of rewriting
                rewritten = self._rewrite_sentence_advanced(sentence)
                rewritten_sentences.append(rewritten)
            else:
                rewritten_sentences.append(sentence)
        
        return ' '.join(rewritten_sentences)
    
    def _rewrite_sentence_advanced(self, sentence):
        """Advanced sentence rewriting using multiple techniques"""
        words = word_tokenize(sentence)
        if len(words) < 4:
            return sentence
        
        # Technique 1: Change voice (active/passive)
        if random.random() < 0.6:
            sentence = self._change_voice(sentence)
        
        # Technique 2: Reorder clauses
        if random.random() < 0.5:
            sentence = self._reorder_clauses(sentence)
        
        # Technique 3: Add/remove modifiers
        if random.random() < 0.4:
            sentence = self._modify_adjectives(sentence)
        
        # Technique 4: Use different sentence blueprint
        if random.random() < 0.7:
            sentence = self._apply_sentence_blueprint(sentence)
        
        return sentence
    
    def _change_voice(self, sentence):
        """Change between active and passive voice"""
        # Simple voice change patterns
        passive_to_active = [
            (r'is\s+(\w+ed)\s+by', r'\\1s'),
            (r'are\s+(\w+ed)\s+by', r'\\1'),
            (r'was\s+(\w+ed)\s+by', r'\\1ed'),
            (r'were\s+(\w+ed)\s+by', r'\\1ed')
        ]
        
        active_to_passive = [
            (r'(\w+)s\s+(\w+)', r'\\2 is \\1ed'),
            (r'(\w+ed)\s+(\w+)', r'\\2 was \\1')
        ]
        
        # Randomly choose transformation direction
        if random.random() < 0.5:
            patterns = passive_to_active
        else:
            patterns = active_to_passive
        
        for pattern, replacement in patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
                break
        
        return sentence
    
    def _reorder_clauses(self, sentence):
        """Reorder sentence clauses for natural variation"""
        clauses = re.split(r'[,;]', sentence)
        if len(clauses) > 1:
            random.shuffle(clauses)
            connectors = ['and', 'but', 'while', 'though', 'although']
            sentence = ' '.join(clauses[0:1] + [random.choice(connectors)] + clauses[1:])
        
        return sentence
    
    def _modify_adjectives(self, sentence):
        """Add, remove, or modify adjectives"""
        words = word_tokenize(sentence)
        new_words = []
        
        adjective_intensity = {
            'good': ['pretty good', 'quite good', 'really good', 'fairly good'],
            'bad': ['pretty bad', 'quite bad', 'really bad', 'fairly bad'],
            'big': ['pretty big', 'quite big', 'really big', 'fairly big'],
            'small': ['pretty small', 'quite small', 'really small', 'fairly small']
        }
        
        i = 0
        while i < len(words):
            word = words[i].lower()
            if word in adjective_intensity and random.random() < 0.3:
                new_words.append(random.choice(adjective_intensity[word]))
                i += 1
            else:
                new_words.append(words[i])
                i += 1
        
        return ' '.join(new_words)
    
    def _apply_sentence_blueprint(self, sentence):
        """Apply human-like sentence structure templates"""
        blueprint = random.choice(self.sentence_blueprints)
        
        # Extract basic sentence components (simplified)
        words = word_tokenize(sentence)
        if len(words) >= 3:
            subject = words[0]
            verb = words[1] if len(words) > 1 else 'is'
            obj = ' '.join(words[2:4]) if len(words) > 3 else ' '.join(words[2:])
            
            # Apply blueprint with extracted components
            new_sentence = blueprint.format(
                subject=subject,
                verb=verb,
                object=obj,
                adverb=random.choice(['really', 'actually', 'basically']),
                preposition=random.choice(['in', 'on', 'with', 'about']),
                context=random.choice(['this case', 'general', 'practice']),
                comparison=random.choice(['nothing else', 'something special', 'usual'])
            )
            
            return new_sentence.capitalize()
        
        return sentence
    
    def _stage3_vocabulary_reshuffle(self, text, intensity):
        """Advanced vocabulary replacement and variation"""
        words = word_tokenize(text)
        new_words = []
        
        for word in words:
            original_word = word.lower()
            
            # Apply multiple vocabulary transformation techniques
            transformed_word = self._apply_synonym_chain(original_word)
            transformed_word = self._casualize_word(transformed_word)
            transformed_word = self._add_variation(transformed_word)
            
            # Preserve capitalization
            if word[0].isupper():
                transformed_word = transformed_word.capitalize()
            
            new_words.append(transformed_word)
        
        return ' '.join(new_words)
    
    def _apply_synonym_chain(self, word):
        """Apply synonym chains for natural variation"""
        for base_word, synonyms in self.synonym_chains.items():
            if word == base_word and random.random() < 0.7:
                return random.choice(synonyms)
        return word
    
    def _casualize_word(self, word):
        """Convert formal words to casual equivalents"""
        if word in self.human_patterns_db['casual_alternatives']:
            return self.human_patterns_db['casual_alternatives'][word]
        return word
    
    def _add_variation(self, word):
        """Add natural human variation to words"""
        variations = {
            'very': ['really', 'pretty', 'quite', 'seriously'],
            'many': ['a lot of', 'plenty of', 'tons of', 'a bunch of'],
            'some': ['a few', 'several', 'a couple of', 'various'],
            'always': ['constantly', 'continually', 'repeatedly', 'time and again'],
            'never': ['not ever', 'absolutely never', 'under no circumstances']
        }
        
        if word in variations and random.random() < 0.4:
            return random.choice(variations[word])
        
        return word
    
    def _stage4_add_human_elements(self, text, intensity):
        """Inject human writing characteristics"""
        sentences = sent_tokenize(text)
        humanized_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Add conversational starters
            if i > 0 and random.random() < 0.6:
                starter = random.choice(self.human_patterns_db['conversational_starters'])
                if not any(sentence.startswith(s.strip()) for s in self.human_patterns_db['conversational_starters']):
                    sentence = starter + ' ' + sentence[0].lower() + sentence[1:]
            
            # Add filler phrases occasionally
            if random.random() < 0.3:
                words = sentence.split()
                if len(words) > 5:
                    insert_point = random.randint(2, len(words) - 2)
                    filler = random.choice(self.human_patterns_db['filler_phrases'])
                    words.insert(insert_point, filler)
                    sentence = ' '.join(words)
            
            humanized_sentences.append(sentence)
        
        text = ' '.join(humanized_sentences)
        
        # Apply contractions aggressively
        for formal, contraction in self.human_patterns_db['contractions'].items():
            if random.random() < 0.9:  # Very high probability for contractions
                text = re.sub(r'\b' + formal + r'\b', contraction, text, flags=re.IGNORECASE)
        
        return text
    
    def _stage5_structural_randomization(self, text, intensity):
        """Randomize paragraph and sentence structure"""
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) > 1:
            # Randomize paragraph order occasionally
            if random.random() < 0.3:
                random.shuffle(paragraphs)
            
            # Vary paragraph lengths
            randomized_paragraphs = []
            for para in paragraphs:
                sentences = sent_tokenize(para)
                if len(sentences) > 3 and random.random() < 0.6:
                    # Split long paragraphs
                    split_point = random.randint(1, len(sentences) - 1)
                    randomized_paragraphs.append(' '.join(sentences[:split_point]))
                    randomized_paragraphs.append(' '.join(sentences[split_point:]))
                else:
                    randomized_paragraphs.append(para)
            
            text = '\n\n'.join(randomized_paragraphs)
        
        return text
    
    def _stage6_conversational_weaving(self, text, intensity):
        """Weave in conversational elements naturally"""
        words = text.split()
        if len(words) < 20:
            return text
        
        # Add rhetorical questions
        if random.random() < 0.4:
            question_points = ['Right?', 'You know?', 'See what I mean?', 'Make sense?']
            insert_point = random.randint(len(words)//3, 2*len(words)//3)
            words.insert(insert_point, random.choice(question_points))
        
        # Add personal references
        if random.random() < 0.3:
            personal_refs = ['I think', 'I believe', 'In my experience', 'From what I\'ve seen']
            insert_point = random.randint(5, len(words) - 5)
            words.insert(insert_point, random.choice(personal_refs))
        
        # Add emphasis markers
        if random.random() < 0.5:
            emphasis = ['really', 'actually', 'seriously', 'honestly']
            insert_point = random.randint(10, len(words) - 10)
            words.insert(insert_point, random.choice(emphasis))
        
        return ' '.join(words)
    
    def _stage7_imperfection_injection(self, text, intensity):
        """Inject natural human imperfections"""
        sentences = sent_tokenize(text)
        imperfect_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            
            # Occasionally add minor imperfections
            if random.random() < 0.2:
                # Repeat word for emphasis (human-like)
                if len(words) > 4:
                    repeat_point = random.randint(2, len(words) - 2)
                    words.insert(repeat_point + 1, words[repeat_point])
            
            # Add hesitation markers
            if random.random() < 0.15:
                hesitations = ['um', 'ah', 'like', 'you know']
                insert_point = random.randint(1, len(words) - 1)
                words.insert(insert_point, random.choice(hesitations))
            
            # Occasionally use sentence fragments
            if random.random() < 0.1 and len(words) > 2:
                words = words[:-1]  # Remove last word to create fragment
            
            imperfect_sentences.append(' '.join(words))
        
        text = ' '.join(imperfect_sentences)
        
        # Add occasional informal punctuation
        if random.random() < 0.3:
            text = text.replace('.', '!', 1)
        if random.random() < 0.2:
            text = text.replace('.', '...', 1)
        
        return text
    
    def _stage8_final_human_touch(self, text, intensity):
        """Final polishing while maintaining human authenticity"""
        # Ensure natural flow
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        
        # Add final conversational elements
        starters = ['Well, ', 'So, ', 'Anyway, ', 'Look, ']
        if random.random() < 0.5 and not any(text.startswith(s.strip()) for s in starters):
            text = random.choice(starters) + text[0].lower() + text[1:]
        
        # Ensure proper capitalization while keeping some informality
        sentences = sent_tokenize(text)
        final_sentences = []
        
        for i, sentence in enumerate(sentences):
            if i > 0 and random.random() < 0.2:
                # Keep some sentences lowercase for conversational feel
                final_sentences.append(sentence)
            else:
                if sentence and sentence[0].isalpha():
                    sentence = sentence[0].upper() + sentence[1:]
                final_sentences.append(sentence)
        
        return ' '.join(final_sentences).strip()

    def humanize_text(self, text, intensity='extreme'):
        """Main humanization function with intensity levels"""
        intensity_map = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'extreme': 1.0
        }
        
        intensity_level = intensity_map.get(intensity, 1.0)
        
        # Apply ultra humanization
        return self.ultra_humanize(text, intensity_level)
    
    def get_humanization_report(self, original_text, humanized_text):
        """Generate comprehensive comparison report"""
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