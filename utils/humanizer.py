import re
import random
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from utils.text_analyzer import TextAnalyzer

class UltraHumanizer:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        
        # Enhanced AI detection evasion patterns
        self.ai_patterns_db = self._build_ai_patterns_database()
        self.human_patterns_db = self._build_human_patterns_database()
        self.synonym_chains = self._build_synonym_chains()
        self.sentence_expanders = self._build_sentence_expanders()
        self.ai_detection_evasion = self._build_ai_evasion_patterns()
        
    def _build_ai_patterns_database(self):
        """Comprehensive database of AI writing patterns"""
        return {
            'formal_transitions': {
                'however': ['but', 'though', 'then again', 'that said', 'on the flip side', 'then'],
                'moreover': ['also', 'besides', 'what\'s more', 'on top of that', 'plus'],
                'furthermore': ['plus', 'additionally', 'not to mention', 'and'],
                'additionally': ['also', 'as well', 'too', 'and'],
                'consequently': ['so', 'as a result', 'because of this', 'that\'s why'],
                'therefore': ['so', 'thus', 'that\'s why', 'which means'],
                'thus': ['so', 'therefore', 'as a result', 'that\'s why'],
                'hence': ['so', 'therefore', 'that\'s why', 'which is why'],
                'nevertheless': ['anyway', 'still', 'even so', 'regardless'],
                'nonetheless': ['regardless', 'anyway', 'still', 'even so']
            },
            'academic_phrases': {
                'it is important to note': ['keep in mind', 'remember that', 'note that', 'don\'t forget that'],
                'it is crucial to': ['we need to', 'we must', 'it\'s vital to', 'we have to'],
                'it is worth noting': ['it\'s worth remembering', 'don\'t forget', 'remember this'],
                'it should be emphasized': ['it\'s key to remember', 'the main point is', 'what matters most is'],
                'from this perspective': ['looking at it this way', 'from this angle', 'in this view'],
                'in this context': ['in this situation', 'under these circumstances', 'given this'],
                'upon careful analysis': ['after looking closely', 'when you examine it', 'looking at it carefully'],
                'it becomes apparent': ['it becomes clear', 'you can see that', 'it\'s obvious that'],
                'it is evident that': ['it\'s clear that', 'obviously', 'you can see that'],
                'as previously mentioned': ['as I said earlier', 'like I mentioned before', 'as noted before']
            },
            'structural_patterns': {
                'in conclusion': ['to wrap up', 'overall', 'all things considered', 'basically'],
                'in summary': ['to sum up', 'basically', 'long story short', 'in short'],
                'to summarize': ['in short', 'put simply', 'the bottom line is', 'basically'],
                'overall': ['all in all', 'by and large', 'generally speaking', 'for the most part'],
                'as a result': ['so', 'because of this', 'that\'s why', 'which means'],
                'in essence': ['basically', 'at its core', 'fundamentally', 'essentially']
            },
            'perfection_indicators': {
                'optimal': ['best', 'ideal', 'perfect', 'great', 'excellent'],
                'maximize': ['get the most out of', 'make the most of', 'boost', 'increase'],
                'minimize': ['reduce', 'cut down on', 'lessen', 'decrease', 'lower'],
                'efficient': ['effective', 'productive', 'well-run', 'smooth', 'streamlined'],
                'effectively': ['well', 'successfully', 'properly', 'smoothly', 'easily'],
                'significantly': ['greatly', 'considerably', 'substantially', 'a lot', 'much'],
                'considerably': ['quite a bit', 'a lot', 'significantly', 'much', 'greatly'],
                'utilize': ['use', 'work with', 'employ', 'apply', 'handle'],
                'facilitate': ['help', 'make easier', 'assist with', 'support', 'enable']
            }
        }
    
    def _build_human_patterns_database(self):
        """Enhanced human writing characteristics for AI evasion"""
        return {
            'conversational_starters': [
                'Well,', 'You know,', 'Actually,', 'So,', 'Look,', 'Honestly,',
                'I mean,', 'Basically,', 'The thing is,', 'To be honest,',
                'Frankly,', 'Seriously,', 'No kidding,', 'Believe it or not,',
                'Truth be told,', 'If you ask me,', 'In my opinion,', 'From my experience,',
                'Personally,', 'The way I see it,', 'From what I\'ve seen,', 'In my view,'
            ],
            'filler_phrases': [
                'kind of', 'sort of', 'you know', 'I think', 'I believe', 
                'I feel like', 'in a way', 'more or less', 'to some extent',
                'pretty much', 'basically', 'essentially', 'like I said',
                'as I mentioned', 'going back to', 'anyway', 'so to speak',
                'I suppose', 'I guess', 'in my experience', 'from what I can tell'
            ],
            'imperfections': [
                'um', 'ah', 'er', 'like', 'right', 'okay', 'well', 'anyway',
                'anyhow', 'so yeah', 'you see', 'I guess', 'supposedly',
                'sorta', 'kinda', 'pretty much', 'you know what I mean'
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
                'that would': 'that\'d', 'I would': 'I\'d', 'you would': 'you\'d',
                'how is': 'how\'s', 'when is': 'when\'s', 'why is': 'why\'s'
            },
            'casual_alternatives': {
                'utilize': 'use', 'facilitate': 'help', 'implement': 'set up',
                'optimal': 'best', 'parameters': 'settings', 'leverage': 'use',
                'commence': 'start', 'terminate': 'end', 'approximately': 'about',
                'subsequently': 'later', 'consequently': 'so', 'therefore': 'so',
                'however': 'but', 'moreover': 'also', 'furthermore': 'plus',
                'additionally': 'also', 'thus': 'so', 'hence': 'so',
                'demonstrate': 'show', 'illustrate': 'show', 'utilize': 'use',
                'acquire': 'get', 'assist': 'help', 'require': 'need',
                'terminate': 'end', 'commence': 'start', 'approximately': 'about',
                'numerous': 'many', 'facilitate': 'make easier', 'objective': 'goal',
                'methodology': 'approach', 'utilization': 'use', 'termination': 'end',
                'approximately': 'about', 'subsequently': 'then', 'prior to': 'before',
                'in order to': 'to', 'with regard to': 'about', 'at this point in time': 'now'
            }
        }

    def _build_ai_evasion_patterns(self):
        """Specific patterns to evade AI detection"""
        return {
            'sentence_length_variation': [8, 12, 15, 18, 22, 25, 30, 35],
            'start_sentence_with': ['I', 'You', 'We', 'It', 'This', 'That', 'There', 'Here'],
            'personal_pronouns': ['I', 'me', 'my', 'mine', 'you', 'your', 'yours', 'we', 'our', 'ours'],
            'informal_connectors': ['and', 'but', 'so', 'then', 'plus', 'also', 'well', 'now'],
            'rhetorical_questions': ['Right?', 'You know?', 'See?', 'Get it?', 'Make sense?'],
            'emphasis_words': ['really', 'actually', 'seriously', 'honestly', 'basically', 'literally']
        }
    
    def _build_synonym_chains(self):
        """Enhanced synonym chains for better AI evasion"""
        return {
            'important': ['crucial', 'vital', 'key', 'essential', 'critical', 'major', 'significant', 'paramount', 'big'],
            'good': ['great', 'excellent', 'awesome', 'fantastic', 'wonderful', 'terrific', 'amazing', 'superb', 'solid'],
            'bad': ['terrible', 'awful', 'horrible', 'poor', 'lousy', 'dreadful', 'unfortunate', 'subpar', 'weak'],
            'big': ['large', 'huge', 'enormous', 'massive', 'giant', 'substantial', 'considerable', 'sizable', 'major'],
            'small': ['tiny', 'little', 'miniature', 'compact', 'petite', 'modest', 'limited', 'minuscule', 'minor'],
            'show': ['demonstrate', 'illustrate', 'reveal', 'display', 'exhibit', 'present', 'indicate', 'prove'],
            'help': ['assist', 'aid', 'support', 'facilitate', 'guide', 'advise', 'mentor', 'back up'],
            'change': ['alter', 'modify', 'adjust', 'transform', 'adapt', 'revise', 'amend', 'shift'],
            'make': ['create', 'produce', 'generate', 'develop', 'construct', 'build', 'fashion', 'form'],
            'use': ['utilize', 'employ', 'apply', 'operate', 'work with', 'handle', 'leverage', 'wield'],
            'think': ['believe', 'feel', 'consider', 'suppose', 'reckon', 'figure', 'deem', 'judge'],
            'get': ['obtain', 'acquire', 'receive', 'secure', 'gain', 'procure', 'attain', 'score'],
            'give': ['provide', 'offer', 'supply', 'furnish', 'donate', 'contribute', 'bestow', 'grant'],
            'tell': ['inform', 'notify', 'advise', 'apprise', 'communicate', 'share', 'disclose', 'reveal'],
            'understand': ['comprehend', 'grasp', 'fathom', 'apprehend', 'discern', 'perceive', 'get'],
            'explain': ['clarify', 'elucidate', 'expound', 'interpret', 'describe', 'illustrate', 'break down']
        }

    def _build_sentence_expanders(self):
        """Enhanced sentence expansion for AI evasion"""
        return {
            'add_personal_opinion': [
                " I think this is important because",
                " From my experience,",
                " Personally, I believe that",
                " In my view,",
                " The way I see it,"
            ],
            'add_real_world_context': [
                " in real-world situations",
                " based on what I've seen",
                " from practical experience",
                " in everyday practice",
                " in actual use cases"
            ],
            'add_casual_emphasis': [
                " which is really quite",
                " and honestly it's",
                " basically making it",
                " seriously improving the",
                " actually helping to"
            ],
            'add_rhetorical_element': [
                " you know what I mean?",
                " which makes sense, right?",
                " and that's pretty clear,",
                " so you can see how",
                " which is kind of obvious,"
            ]
        }

    def ultra_humanize(self, text, intensity='extreme'):
        """Advanced humanization focused on AI detection evasion"""
        if not text or len(text.strip()) < 10:
            return text
        
        # Preserve original paragraph structure
        original_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        humanized_paragraphs = []
        
        for paragraph in original_paragraphs:
            humanized_paragraph = self._aggressive_ai_evasion_processing(paragraph, intensity)
            humanized_paragraphs.append(humanized_paragraph)
        
        return '\n\n'.join(humanized_paragraphs)
    
    def _aggressive_ai_evasion_processing(self, paragraph, intensity):
        """Aggressive processing focused on AI detection evasion"""
        sentences = sent_tokenize(paragraph)
        if not sentences:
            return paragraph
        
        humanized_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Apply aggressive AI evasion transformations
            humanized_sentence = self._aggressive_sentence_evasion(sentence, i)
            humanized_sentences.append(humanized_sentence)
        
        # Apply paragraph-level AI evasion
        coherent_paragraph = self._ai_evasion_paragraph_coherence(humanized_sentences)
        return coherent_paragraph
    
    def _aggressive_sentence_evasion(self, sentence, sentence_index):
        """Aggressively transform sentence to evade AI detection"""
        if len(sentence.split()) < 3:
            return sentence
        
        # Step 1: Remove ALL AI patterns aggressively
        sentence = self._aggressive_ai_pattern_removal(sentence)
        
        # Step 2: Add personal and conversational elements
        sentence = self._add_aggressive_human_elements(sentence, sentence_index)
        
        # Step 3: Vary sentence structure aggressively
        sentence = self._aggressive_sentence_variation(sentence)
        
        # Step 4: Add AI evasion specific patterns
        sentence = self._add_ai_evasion_patterns(sentence)
        
        return sentence
    
    def _aggressive_ai_pattern_removal(self, sentence):
        """Aggressively remove all AI patterns"""
        original_sentence = sentence
        
        # Aggressive replacement of ALL AI patterns
        for category, patterns in self.ai_patterns_db.items():
            for formal, alternatives in patterns.items():
                if re.search(r'\b' + re.escape(formal) + r'\b', sentence, re.IGNORECASE):
                    replacement = random.choice(alternatives)
                    sentence = re.sub(r'\b' + re.escape(formal) + r'\b', replacement, sentence, re.IGNORECASE)
        
        # Additional aggressive replacements
        aggressive_replacements = {
            r'\bvery\b': random.choice(['really', 'pretty', 'quite', 'seriously']),
            r'\bmany\b': random.choice(['a lot of', 'plenty of', 'tons of', 'loads of']),
            r'\bsome\b': random.choice(['a few', 'several', 'a couple of', 'various']),
            r'\balways\b': random.choice(['constantly', 'all the time', 'repeatedly', 'consistently']),
            r'\bnever\b': random.choice(['not ever', 'absolutely never', 'no way', 'under no circumstances'])
        }
        
        for pattern, replacement in aggressive_replacements.items():
            if random.random() < 0.6:
                sentence = re.sub(pattern, replacement, sentence, re.IGNORECASE)
        
        return sentence if len(sentence.split()) >= 3 else original_sentence
    
    def _add_aggressive_human_elements(self, sentence, sentence_index):
        """Add aggressive human writing elements"""
        words = sentence.split()
        
        # Add personal pronouns frequently
        if random.random() < 0.7 and len(words) > 4:
            personal_pronouns = self.ai_detection_evasion['personal_pronouns']
            insert_point = random.randint(1, len(words) - 2)
            words.insert(insert_point, random.choice(personal_pronouns))
        
        # Add conversational starters (especially at beginning)
        if sentence_index == 0 and random.random() < 0.8:
            starters = self.human_patterns_db['conversational_starters']
            sentence = random.choice(starters) + ' ' + sentence[0].lower() + sentence[1:]
        elif random.random() < 0.4:
            starters = self.human_patterns_db['conversational_starters']
            sentence = random.choice(starters) + ' ' + sentence[0].lower() + sentence[1:]
        
        # Add filler phrases aggressively
        if random.random() < 0.5 and len(words) > 6:
            filler = random.choice(self.human_patterns_db['filler_phrases'])
            insert_point = random.randint(2, len(words) - 3)
            words.insert(insert_point, filler)
        
        sentence = ' '.join(words)
        
        # Aggressive contraction application
        for formal, contraction in self.human_patterns_db['contractions'].items():
            if random.random() < 0.9:  # Very high probability
                sentence = re.sub(r'\b' + formal + r'\b', contraction, sentence, re.IGNORECASE)
        
        return sentence
    
    def _aggressive_sentence_variation(self, sentence):
        """Aggressively vary sentence structure"""
        words = sentence.split()
        
        # Add rhetorical questions occasionally
        if random.random() < 0.3 and len(words) > 8:
            questions = self.ai_detection_evasion['rhetorical_questions']
            sentence = sentence + ' ' + random.choice(questions)
        
        # Add emphasis words
        if random.random() < 0.4:
            emphasis = random.choice(self.ai_detection_evasion['emphasis_words'])
            if len(words) > 3:
                insert_point = random.randint(1, len(words) - 2)
                words.insert(insert_point, emphasis)
                sentence = ' '.join(words)
        
        return sentence
    
    def _add_ai_evasion_patterns(self, sentence):
        """Add specific patterns that help evade AI detection"""
        # Ensure sentence doesn't start with typical AI patterns
        ai_starter_patterns = ['The', 'This', 'It', 'There', 'One']
        words = sentence.split()
        
        if words and words[0] in ai_starter_patterns and random.random() < 0.6:
            human_starters = ['Well,', 'So,', 'You know,', 'Actually,', 'Look,']
            sentence = random.choice(human_starters) + ' ' + sentence[0].lower() + sentence[1:]
        
        return sentence
    
    def _ai_evasion_paragraph_coherence(self, sentences):
        """Create paragraph coherence while evading AI detection"""
        if not sentences:
            return ""
        
        connected_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            current_sentence = sentences[i]
            
            # Use informal connectors frequently
            if random.random() < 0.6:
                connectors = self.ai_detection_evasion['informal_connectors']
                current_sentence = random.choice(connectors) + ', ' + current_sentence[0].lower() + current_sentence[1:]
            
            connected_sentences.append(current_sentence)
        
        return ' '.join(connected_sentences)
    
    def humanize_text(self, text, intensity='extreme'):
        """Main humanization function"""
        return self.ultra_humanize(text, intensity)
    
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