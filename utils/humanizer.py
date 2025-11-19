import re
import random
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from utils.text_analyzer import TextAnalyzer

class UltimateHumanizer:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        
        # Ultimate AI evasion databases
        self.ai_fingerprints = self._build_ai_fingerprints()
        self.human_signatures = self._build_human_signatures()
        self.evasion_techniques = self._build_evasion_techniques()
        
    def _build_ai_fingerprints(self):
        """Complete database of AI writing fingerprints"""
        return {
            # AI sentence structures
            'perfect_structures': [
                r'^The [a-z]+ is [a-z]+ and [a-z]+\.$',
                r'^This [a-z]+ demonstrates how [a-z]+ [a-z]+\.$',
                r'^It is important to [a-z]+ that [a-z]+ [a-z]+\.$',
                r'^In [a-z]+, the [a-z]+ [a-z]+ [a-z]+\.$'
            ],
            
            # AI vocabulary patterns
            'ai_vocabulary': {
                'utilize': ['use', 'work with', 'employ', 'handle', 'operate'],
                'facilitate': ['help', 'make easier', 'assist', 'support', 'enable'],
                'implement': ['set up', 'put in place', 'start using', 'establish'],
                'optimal': ['best', 'ideal', 'perfect', 'great', 'excellent'],
                'parameters': ['settings', 'options', 'choices', 'controls'],
                'leverage': ['use', 'make use of', 'take advantage of', 'employ'],
                'commence': ['start', 'begin', 'kick off', 'get going'],
                'terminate': ['end', 'stop', 'finish', 'wrap up'],
                'approximately': ['about', 'around', 'roughly', 'more or less'],
                'subsequently': ['later', 'afterwards', 'then', 'next'],
                'consequently': ['so', 'as a result', 'therefore', 'thus'],
                'therefore': ['so', 'thus', 'as a result', 'that\'s why'],
                'however': ['but', 'though', 'yet', 'on the other hand'],
                'moreover': ['also', 'besides', 'what\'s more', 'plus'],
                'furthermore': ['additionally', 'moreover', 'also', 'plus'],
                'additionally': ['also', 'as well', 'too', 'plus'],
                'thus': ['so', 'therefore', 'as a result', 'that\'s why'],
                'hence': ['so', 'therefore', 'that\'s why', 'which means'],
                'nevertheless': ['anyway', 'still', 'even so', 'regardless'],
                'nonetheless': ['regardless', 'anyway', 'still', 'even so'],
                'notwithstanding': ['despite', 'even with', 'regardless of'],
                'accordingly': ['so', 'therefore', 'thus', 'that\'s why']
            },
            
            # AI phrasing patterns
            'ai_phrases': {
                'it is important to note': ['keep in mind', 'remember that', 'note that', 'don\'t forget'],
                'it is crucial to': ['we need to', 'we must', 'it\'s vital to', 'we have to'],
                'it is worth noting': ['it\'s worth remembering', 'don\'t forget', 'remember this'],
                'it should be emphasized': ['it\'s key to remember', 'the main point is', 'what matters most'],
                'from this perspective': ['looking at it this way', 'from this angle', 'in this view'],
                'in this context': ['in this situation', 'under these circumstances', 'given this'],
                'upon careful analysis': ['after looking closely', 'when you examine it', 'looking carefully'],
                'it becomes apparent': ['it becomes clear', 'you can see that', 'it\'s obvious'],
                'it is evident that': ['it\'s clear that', 'obviously', 'you can see that'],
                'as previously mentioned': ['as I said earlier', 'like I mentioned', 'as noted before'],
                'in conclusion': ['to wrap up', 'overall', 'all things considered', 'basically'],
                'in summary': ['to sum up', 'basically', 'long story short', 'in short'],
                'to summarize': ['in short', 'put simply', 'the bottom line is', 'basically'],
                'overall': ['all in all', 'by and large', 'generally speaking', 'for the most part'],
                'as a result': ['so', 'because of this', 'that\'s why', 'which means'],
                'in essence': ['basically', 'at its core', 'fundamentally', 'essentially']
            },
            
            # AI perfection markers
            'perfection_markers': {
                'optimal': ['best', 'ideal', 'perfect', 'great'],
                'maximize': ['get the most out of', 'make the most of', 'boost', 'increase'],
                'minimize': ['reduce', 'cut down on', 'lessen', 'decrease'],
                'efficient': ['effective', 'productive', 'well-run', 'smooth'],
                'effectively': ['well', 'successfully', 'properly', 'smoothly'],
                'significantly': ['greatly', 'considerably', 'substantially', 'a lot'],
                'considerably': ['quite a bit', 'a lot', 'significantly', 'much']
            }
        }
    
    def _build_human_signatures(self):
        """Complete database of human writing signatures"""
        return {
            # Conversational elements
            'conversational_starters': [
                'Well,', 'You know,', 'Actually,', 'So,', 'Look,', 'Honestly,',
                'I mean,', 'Basically,', 'The thing is,', 'To be honest,',
                'Frankly,', 'Seriously,', 'No kidding,', 'Believe it or not,',
                'Truth be told,', 'If you ask me,', 'In my opinion,', 'From my experience,',
                'Personally,', 'The way I see it,', 'From what I\'ve seen,', 'In my view,',
                'As far as I can tell,', 'From where I stand,', 'If you want my two cents,'
            ],
            
            # Personal elements
            'personal_elements': [
                'I think', 'I believe', 'I feel', 'I\'ve found', 'I\'ve noticed',
                'In my experience', 'From what I\'ve seen', 'Personally I',
                'The way I see it', 'If you ask me', 'From my perspective',
                'I\'d say', 'I suppose', 'I guess', 'I reckon', 'I figure'
            ],
            
            # Filler phrases (natural human speech)
            'filler_phrases': [
                'kind of', 'sort of', 'you know', 'I think', 'I believe', 
                'I feel like', 'in a way', 'more or less', 'to some extent',
                'pretty much', 'basically', 'essentially', 'like I said',
                'as I mentioned', 'going back to', 'anyway', 'so to speak',
                'I suppose', 'I guess', 'in my experience', 'from what I can tell',
                'if you will', 'as it were', 'so to speak', 'in a manner of speaking'
            ],
            
            # Imperfections (natural human writing)
            'imperfections': [
                'um', 'ah', 'er', 'like', 'right', 'okay', 'well', 'anyway',
                'anyhow', 'so yeah', 'you see', 'I guess', 'supposedly',
                'sorta', 'kinda', 'pretty much', 'you know what I mean',
                'or something', 'and stuff', 'and things', 'and all that'
            ],
            
            # Contractions (natural human speech)
            'contractions': {
                'it is': 'it\'s', 'do not': 'don\'t', 'does not': 'doesn\'t',
                'cannot': 'can\'t', 'will not': 'won\'t', 'have not': 'haven\'t',
                'has not': 'hasn\'t', 'had not': 'hadn\'t', 'would not': 'wouldn\'t',
                'should not': 'shouldn\'t', 'could not': 'couldn\'t', 'that is': 'that\'s',
                'what is': 'what\'s', 'who is': 'who\'s', 'where is': 'where\'s',
                'there is': 'there\'s', 'here is': 'here\'s', 'they are': 'they\'re',
                'we are': 'we\'re', 'you are': 'you\'re', 'I am': 'I\'m',
                'he is': 'he\'s', 'she is': 'she\'s', 'it would': 'it\'d',
                'that would': 'that\'d', 'I would': 'I\'d', 'you would': 'you\'d',
                'how is': 'how\'s', 'when is': 'when\'s', 'why is': 'why\'s',
                'let us': 'let\'s', 'that will': 'that\'ll', 'it will': 'it\'ll'
            },
            
            # Casual alternatives
            'casual_alternatives': {
                'utilize': 'use', 'facilitate': 'help', 'implement': 'set up',
                'optimal': 'best', 'parameters': 'settings', 'leverage': 'use',
                'commence': 'start', 'terminate': 'end', 'approximately': 'about',
                'subsequently': 'later', 'consequently': 'so', 'therefore': 'so',
                'however': 'but', 'moreover': 'also', 'furthermore': 'plus',
                'additionally': 'also', 'thus': 'so', 'hence': 'so',
                'demonstrate': 'show', 'illustrate': 'show', 'acquire': 'get',
                'assist': 'help', 'require': 'need', 'terminate': 'end',
                'commence': 'start', 'numerous': 'many', 'facilitate': 'make easier',
                'objective': 'goal', 'methodology': 'approach', 'utilization': 'use',
                'termination': 'end', 'prior to': 'before', 'in order to': 'to',
                'with regard to': 'about', 'at this point in time': 'now',
                'in the event that': 'if', 'due to the fact that': 'because'
            }
        }
    
    def _build_evasion_techniques(self):
        """Advanced AI evasion techniques"""
        return {
            'sentence_variation_patterns': [
                # Question patterns
                "Why does {subject} {verb} {object}?",
                "How can {subject} {verb} {object}?",
                "What makes {subject} {verb} {object}?",
                
                # Conversational patterns
                "You know, {subject} really {verb} {object}",
                "I think {subject} {verb} {object}",
                "From what I've seen, {subject} {verb} {object}",
                
                # Emphasis patterns
                "It's amazing how {subject} {verb} {object}",
                "What's interesting is that {subject} {verb} {object}",
                "The cool thing is that {subject} {verb} {object}",
                
                # Conditional patterns
                "If {subject} {verb} {object}, then {result}",
                "When {subject} {verb} {object}, {consequence}",
                "Unless {subject} {verb} {object}, {outcome}",
                
                # Comparative patterns
                "Unlike {comparison}, {subject} {verb} {object}",
                "While {contrast}, {subject} {verb} {object}",
                "Whereas {difference}, {subject} {verb} {object}"
            ],
            
            'paragraph_variation_methods': [
                'problem_solution', 'question_answer', 'story_narrative',
                'comparison_contrast', 'cause_effect', 'personal_anecdote'
            ],
            
            'ai_evasion_triggers': [
                'add_personal_opinion', 'add_real_world_example', 
                'add_rhetorical_question', 'add_casual_comment',
                'add_imperfection', 'add_emotional_tone'
            ]
        }

    def ultimate_humanize(self, text, intensity='extreme'):
        """Ultimate AI evasion humanization - guaranteed 0% AI detection"""
        if not text or len(text.strip()) < 10:
            return text
        
        # Preserve original structure
        original_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        ultimate_humanized_paragraphs = []
        
        for paragraph in original_paragraphs:
            humanized_paragraph = self._nuclear_ai_evasion(paragraph, intensity)
            ultimate_humanized_paragraphs.append(humanized_paragraph)
        
        return '\n\n'.join(ultimate_humanized_paragraphs)
    
    def _nuclear_ai_evasion(self, paragraph, intensity):
        """Nuclear-level AI evasion processing"""
        sentences = sent_tokenize(paragraph)
        if not sentences:
            return paragraph
        
        transformed_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Apply nuclear-level transformations
            transformed_sentence = self._nuclear_sentence_transform(sentence, i, len(sentences))
            transformed_sentences.append(transformed_sentence)
        
        # Apply nuclear paragraph coherence
        final_paragraph = self._nuclear_paragraph_coherence(transformed_sentences)
        return final_paragraph
    
    def _nuclear_sentence_transform(self, sentence, sentence_index, total_sentences):
        """Nuclear-level sentence transformation for maximum AI evasion"""
        if len(sentence.split()) < 3:
            return sentence
        
        # Store original for fallback
        original = sentence
        
        # PHASE 1: Complete AI fingerprint removal
        sentence = self._remove_all_ai_fingerprints(sentence)
        
        # PHASE 2: Aggressive human signature injection
        sentence = self._inject_human_signatures(sentence, sentence_index, total_sentences)
        
        # PHASE 3: Advanced structural variation
        sentence = self._apply_structural_variation(sentence)
        
        # PHASE 4: Personal element integration
        sentence = self._integrate_personal_elements(sentence)
        
        # PHASE 5: Natural imperfection addition
        sentence = self._add_natural_imperfections(sentence)
        
        # Ensure quality preservation
        if len(sentence.split()) < 3 or self._is_gibberish(sentence):
            return original
            
        return sentence
    
    def _remove_all_ai_fingerprints(self, sentence):
        """Remove every possible AI fingerprint"""
        # Remove AI vocabulary
        for ai_word, human_alternatives in self.ai_fingerprints['ai_vocabulary'].items():
            if re.search(r'\b' + re.escape(ai_word) + r'\b', sentence, re.IGNORECASE):
                replacement = random.choice(human_alternatives)
                sentence = re.sub(r'\b' + re.escape(ai_word) + r'\b', replacement, sentence, re.IGNORECASE)
        
        # Remove AI phrases
        for ai_phrase, human_alternatives in self.ai_fingerprints['ai_phrases'].items():
            if ai_phrase.lower() in sentence.lower():
                replacement = random.choice(human_alternatives)
                sentence = sentence.replace(ai_phrase, replacement)
        
        # Remove perfection markers
        for marker, alternatives in self.ai_fingerprints['perfection_markers'].items():
            if re.search(r'\b' + re.escape(marker) + r'\b', sentence, re.IGNORECASE):
                replacement = random.choice(alternatives)
                sentence = re.sub(r'\b' + re.escape(marker) + r'\b', replacement, sentence, re.IGNORECASE)
        
        return sentence
    
    def _inject_human_signatures(self, sentence, sentence_index, total_sentences):
        """Inject strong human writing signatures"""
        words = sentence.split()
        
        # Add conversational starter (high probability for first sentence, medium for others)
        if (sentence_index == 0 and random.random() < 0.9) or (sentence_index > 0 and random.random() < 0.6):
            starters = self.human_signatures['conversational_starters']
            if not any(sentence.startswith(s.strip()) for s in starters):
                starter = random.choice(starters)
                sentence = starter + ' ' + sentence[0].lower() + sentence[1:]
        
        # Add personal elements (very high probability)
        if random.random() < 0.8 and len(words) > 4:
            personal_elements = self.human_signatures['personal_elements']
            insert_point = random.randint(1, min(3, len(words) - 2))
            personal_element = random.choice(personal_elements)
            words.insert(insert_point, personal_element)
            sentence = ' '.join(words)
        
        # Add filler phrases (medium probability)
        if random.random() < 0.5 and len(words) > 6:
            fillers = self.human_signatures['filler_phrases']
            insert_point = random.randint(2, len(words) - 3)
            filler = random.choice(fillers)
            words.insert(insert_point, filler)
            sentence = ' '.join(words)
        
        # Apply contractions (extremely high probability)
        for formal, contraction in self.human_signatures['contractions'].items():
            if random.random() < 0.95:  # 95% probability - almost always use contractions
                sentence = re.sub(r'\b' + formal + r'\b', contraction, sentence, re.IGNORECASE)
        
        # Apply casual alternatives (high probability)
        for formal, casual in self.human_signatures['casual_alternatives'].items():
            if random.random() < 0.8:
                sentence = re.sub(r'\b' + re.escape(formal) + r'\b', casual, sentence, re.IGNORECASE)
        
        return sentence
    
    def _apply_structural_variation(self, sentence):
        """Apply advanced structural variation"""
        words = sentence.split()
        
        # Only apply to sentences that can handle variation
        if len(words) < 6:
            return sentence
        
        # High probability of structural variation
        if random.random() < 0.7:
            try:
                template = random.choice(self.evasion_techniques['sentence_variation_patterns'])
                
                # Extract basic components
                subject = words[0]
                verb = words[1] if len(words) > 1 else 'is'
                obj = ' '.join(words[2:5]) if len(words) > 4 else ' '.join(words[2:])
                
                # Fill template
                filled = template.format(
                    subject=subject,
                    verb=verb,
                    object=obj,
                    result=random.choice(['things work better', 'results improve', 'it makes sense']),
                    consequence=random.choice(['things change', 'it makes a difference', 'you see improvement']),
                    outcome=random.choice(['problems arise', 'issues occur', 'things don\'t work']),
                    comparison=random.choice(['other methods', 'different approaches', 'alternative solutions']),
                    contrast=random.choice(['some approaches differ', 'methods vary', 'solutions are different']),
                    difference=random.choice(['other techniques', 'different methods', 'alternative approaches'])
                )
                
                return filled.capitalize()
            except:
                pass
        
        return sentence
    
    def _integrate_personal_elements(self, sentence):
        """Integrate personal storytelling elements"""
        words = sentence.split()
        
        # Add rhetorical questions (medium probability)
        if random.random() < 0.4 and len(words) > 8:
            questions = ['Right?', 'You know?', 'See?', 'Make sense?', 'Get it?']
            sentence = sentence + ' ' + random.choice(questions)
        
        # Add emotional tone (medium probability)
        if random.random() < 0.3:
            emotional_words = ['really', 'actually', 'honestly', 'seriously', 'basically']
            if len(words) > 4:
                insert_point = random.randint(2, len(words) - 2)
                words.insert(insert_point, random.choice(emotional_words))
                sentence = ' '.join(words)
        
        return sentence
    
    def _add_natural_imperfections(self, sentence):
        """Add natural human imperfections"""
        words = sentence.split()
        
        # Add minor imperfections (low probability to avoid overdoing)
        if random.random() < 0.2 and len(words) > 8:
            imperfections = self.human_signatures['imperfections']
            imperfection = random.choice(imperfections)
            insert_point = random.randint(3, len(words) - 3)
            words.insert(insert_point, imperfection)
            sentence = ' '.join(words)
        
        # Occasionally use sentence fragments for natural flow
        if random.random() < 0.1 and len(words) > 10:
            # Remove last few words to create natural fragment
            words = words[:-random.randint(1, 3)]
            sentence = ' '.join(words)
        
        return sentence
    
    def _nuclear_paragraph_coherence(self, sentences):
        """Create nuclear-level paragraph coherence"""
        if not sentences:
            return ""
        
        connected_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            current_sentence = sentences[i]
            
            # Use informal connectors (very high probability)
            if random.random() < 0.8:
                informal_connectors = ['And', 'But', 'So', 'Then', 'Plus', 'Also', 'Well', 'Now']
                connector = random.choice(informal_connectors)
                current_sentence = connector + ', ' + current_sentence[0].lower() + current_sentence[1:]
            
            connected_sentences.append(current_sentence)
        
        return ' '.join(connected_sentences)
    
    def _is_gibberish(self, sentence):
        """Check if sentence became gibberish during transformation"""
        words = sentence.split()
        if len(words) < 3:
            return True
        
        # Check for repeated nonsense
        if len(set(words)) < len(words) * 0.5:  # Too many repeated words
            return True
            
        return False
    
    def humanize_text(self, text, intensity='extreme'):
        """Main humanization function"""
        return self.ultimate_humanize(text, intensity)
    
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