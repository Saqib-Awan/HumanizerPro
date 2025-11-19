import re
import random
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from utils.text_analyzer import TextAnalyzer

class UltraHumanizer:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        
        # Enhanced transformation databases
        self.ai_patterns_db = self._build_ai_patterns_database()
        self.human_patterns_db = self._build_human_patterns_database()
        self.synonym_chains = self._build_synonym_chains()
        self.sentence_expanders = self._build_sentence_expanders()
        self.paragraph_structures = self._build_paragraph_structures()
        
    def _build_ai_patterns_database(self):
        """Comprehensive database of AI writing patterns"""
        return {
            'formal_transitions': {
                'however': ['but', 'though', 'then again', 'that said', 'on the flip side'],
                'moreover': ['also', 'besides', 'what\'s more', 'on top of that'],
                'furthermore': ['plus', 'additionally', 'not to mention'],
                'additionally': ['also', 'as well', 'too'],
                'consequently': ['so', 'as a result', 'because of this'],
                'therefore': ['so', 'thus', 'that\'s why'],
                'thus': ['so', 'therefore', 'as a result'],
                'hence': ['so', 'therefore', 'that\'s why'],
                'nevertheless': ['anyway', 'still', 'even so'],
                'nonetheless': ['regardless', 'anyway', 'still']
            },
            'academic_phrases': {
                'it is important to note': ['keep in mind', 'remember that', 'note that'],
                'it is crucial to': ['we need to', 'we must', 'it\'s vital to'],
                'it is worth noting': ['it\'s worth remembering', 'don\'t forget'],
                'it should be emphasized': ['it\'s key to remember', 'the main point is'],
                'from this perspective': ['looking at it this way', 'from this angle'],
                'in this context': ['in this situation', 'under these circumstances'],
                'upon careful analysis': ['after looking closely', 'when you examine it'],
                'it becomes apparent': ['it becomes clear', 'you can see that']
            },
            'structural_patterns': {
                'in conclusion': ['to wrap up', 'overall', 'all things considered'],
                'in summary': ['to sum up', 'basically', 'long story short'],
                'to summarize': ['in short', 'put simply', 'the bottom line is'],
                'overall': ['all in all', 'by and large', 'generally speaking'],
                'as a result': ['so', 'because of this', 'that\'s why'],
                'in essence': ['basically', 'at its core', 'fundamentally']
            },
            'perfection_indicators': {
                'optimal': ['best', 'ideal', 'perfect'],
                'maximize': ['get the most out of', 'make the most of'],
                'minimize': ['reduce', 'cut down on', 'lessen'],
                'efficient': ['effective', 'productive', 'well-run'],
                'effectively': ['well', 'successfully', 'properly'],
                'significantly': ['greatly', 'considerably', 'substantially'],
                'considerably': ['quite a bit', 'a lot', 'significantly']
            }
        }
    
    def _build_human_patterns_database(self):
        """Database of human writing characteristics"""
        return {
            'conversational_starters': [
                'Well,', 'You know,', 'Actually,', 'So,', 'Look,', 'Honestly,',
                'I mean,', 'Basically,', 'The thing is,', 'To be honest,',
                'Frankly,', 'Seriously,', 'No kidding,', 'Believe it or not,',
                'Truth be told,', 'If you ask me,', 'In my opinion,', 'From my experience,'
            ],
            'filler_phrases': [
                'kind of', 'sort of', 'you know', 'I think', 'I believe', 
                'I feel like', 'in a way', 'more or less', 'to some extent',
                'pretty much', 'basically', 'essentially', 'like I said',
                'as I mentioned', 'going back to', 'anyway', 'so to speak'
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
                'additionally': 'also', 'thus': 'so', 'hence': 'so',
                'demonstrate': 'show', 'illustrate': 'show', 'utilize': 'use',
                'acquire': 'get', 'assist': 'help', 'require': 'need',
                'terminate': 'end', 'commence': 'start', 'approximately': 'about',
                'numerous': 'many', 'facilitate': 'make easier', 'objective': 'goal',
                'methodology': 'approach', 'utilization': 'use', 'termination': 'end'
            }
        }
    
    def _build_synonym_chains(self):
        """Advanced synonym chains for vocabulary variation"""
        return {
            'important': ['crucial', 'vital', 'key', 'essential', 'critical', 'major', 'significant', 'paramount'],
            'good': ['great', 'excellent', 'awesome', 'fantastic', 'wonderful', 'terrific', 'amazing', 'superb'],
            'bad': ['terrible', 'awful', 'horrible', 'poor', 'lousy', 'dreadful', 'unfortunate', 'subpar'],
            'big': ['large', 'huge', 'enormous', 'massive', 'giant', 'substantial', 'considerable', 'sizable'],
            'small': ['tiny', 'little', 'miniature', 'compact', 'petite', 'modest', 'limited', 'minuscule'],
            'show': ['demonstrate', 'illustrate', 'reveal', 'display', 'exhibit', 'present', 'indicate'],
            'help': ['assist', 'aid', 'support', 'facilitate', 'guide', 'advise', 'mentor'],
            'change': ['alter', 'modify', 'adjust', 'transform', 'adapt', 'revise', 'amend'],
            'make': ['create', 'produce', 'generate', 'develop', 'construct', 'build', 'fashion'],
            'use': ['utilize', 'employ', 'apply', 'operate', 'work with', 'handle', 'leverage'],
            'think': ['believe', 'feel', 'consider', 'suppose', 'reckon', 'figure', 'deem'],
            'get': ['obtain', 'acquire', 'receive', 'secure', 'gain', 'procure', 'attain'],
            'give': ['provide', 'offer', 'supply', 'furnish', 'donate', 'contribute', 'bestow'],
            'tell': ['inform', 'notify', 'advise', 'apprise', 'communicate', 'share', 'disclose'],
            'understand': ['comprehend', 'grasp', 'fathom', 'apprehend', 'discern', 'perceive'],
            'explain': ['clarify', 'elucidate', 'expound', 'interpret', 'describe', 'illustrate']
        }

    def _build_sentence_expanders(self):
        """Professional sentence expansion templates"""
        return {
            'add_description': [
                ", which is {description},",
                ", a {description} approach,",
                ", offering {description} benefits,"
            ],
            'add_purpose': [
                " in order to {purpose}",
                " with the goal of {purpose}",
                " aiming to {purpose}"
            ],
            'add_context': [
                " within the context of {context}",
                " when considering {context}",
                " especially in {context}"
            ],
            'add_result': [
                " resulting in {result}",
                " which leads to {result}",
                " ultimately {result}"
            ],
            'add_comparison': [
                " unlike {comparison}",
                " compared to {comparison}",
                " whereas {comparison}"
            ],
            'add_condition': [
                " provided that {condition}",
                " as long as {condition}",
                " when {condition}"
            ]
        }
    
    def _build_paragraph_structures(self):
        """Professional paragraph organization patterns"""
        return [
            # Problem-Solution
            ["situation_context", "problem_statement", "solution_approach", "expected_outcome"],
            
            # General to Specific
            ["general_statement", "specific_example", "detailed_explanation", "conclusion"],
            
            # Compare-Contrast
            ["topic_intro", "first_aspect", "second_aspect", "comparison_analysis"],
            
            # Sequential
            ["initial_state", "development_process", "current_situation", "future_implications"]
        ]

    def ultra_humanize(self, text, intensity='extreme'):
        """Professional humanization that expands sentences and maintains structure"""
        if not text or len(text.strip()) < 10:
            return text
        
        # Preserve original paragraph structure
        original_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        humanized_paragraphs = []
        
        for paragraph in original_paragraphs:
            humanized_paragraph = self._process_paragraph_professionally(paragraph, intensity)
            humanized_paragraphs.append(humanized_paragraph)
        
        # Reconstruct with original paragraph breaks
        return '\n\n'.join(humanized_paragraphs)
    
    def _process_paragraph_professionally(self, paragraph, intensity):
        """Process a paragraph with professional expansion and structure preservation"""
        sentences = sent_tokenize(paragraph)
        if not sentences:
            return paragraph
        
        humanized_sentences = []
        
        for sentence in sentences:
            # Apply professional transformations that EXPAND sentences
            humanized_sentence = self._professional_sentence_expansion(sentence)
            humanized_sentences.append(humanized_sentence)
        
        # Ensure paragraph coherence with expanded sentences
        coherent_paragraph = self._ensure_paragraph_coherence_professional(humanized_sentences)
        return coherent_paragraph
    
    def _professional_sentence_expansion(self, sentence):
        """Professionally expand and enhance a sentence"""
        if len(sentence.split()) < 3:  # Too short to work with
            return sentence
        
        # Step 1: Replace AI patterns with human alternatives
        sentence = self._replace_ai_patterns(sentence)
        
        # Step 2: Expand sentence professionally (instead of shortening)
        sentence = self._expand_sentence_professionally(sentence)
        
        # Step 3: Vary vocabulary professionally
        sentence = self._professional_vocabulary_variation(sentence)
        
        # Step 4: Add natural human elements (subtly)
        sentence = self._add_subtle_human_elements(sentence)
        
        return sentence
    
    def _expand_sentence_professionally(self, sentence):
        """Expand sentence length professionally while maintaining meaning"""
        words = sentence.split()
        
        # Only expand if sentence is too short or could benefit from elaboration
        if len(words) < 15 and random.random() < 0.7:
            expansion_type = random.choice(list(self.sentence_expanders.keys()))
            expansion_template = random.choice(self.sentence_expanders[expansion_type])
            
            # Apply appropriate expansion based on type
            if expansion_type == 'add_description':
                descriptions = ['highly effective', 'particularly useful', 'quite innovative', 'really practical']
                expansion = expansion_template.format(description=random.choice(descriptions))
                
            elif expansion_type == 'add_purpose':
                purposes = ['achieve better results', 'improve outcomes', 'enhance performance', 'streamline processes']
                expansion = expansion_template.format(purpose=random.choice(purposes))
                
            elif expansion_type == 'add_context':
                contexts = ['modern applications', 'today\'s environment', 'current scenarios', 'real-world situations']
                expansion = expansion_template.format(context=random.choice(contexts))
                
            elif expansion_type == 'add_result':
                results = ['significant improvements', 'better outcomes', 'enhanced efficiency', 'positive changes']
                expansion = expansion_template.format(result=random.choice(results))
                
            elif expansion_type == 'add_comparison':
                comparisons = ['traditional methods', 'older approaches', 'conventional techniques', 'standard practices']
                expansion = expansion_template.format(comparison=random.choice(comparisons))
                
            elif expansion_type == 'add_condition':
                conditions = ['implemented correctly', 'used properly', 'applied consistently', 'executed well']
                expansion = expansion_template.format(condition=random.choice(conditions))
            
            # Insert expansion at appropriate position
            if expansion_type in ['add_description', 'add_purpose']:
                # Insert after main verb or object
                if len(words) > 3:
                    insert_pos = min(4, len(words) - 1)
                    words = words[:insert_pos] + [expansion] + words[insert_pos:]
                else:
                    words.append(expansion)
            else:
                # Append to end for other expansion types
                words.append(expansion)
            
            sentence = ' '.join(words)
        
        return sentence
    
    def _replace_ai_patterns(self, sentence):
        """Replace AI patterns with natural human alternatives"""
        original_sentence = sentence
        
        # Replace formal transitions
        for formal, alternatives in self.ai_patterns_db['formal_transitions'].items():
            if re.search(r'\b' + re.escape(formal) + r'\b', sentence, re.IGNORECASE):
                replacement = random.choice(alternatives)
                sentence = re.sub(r'\b' + re.escape(formal) + r'\b', replacement, sentence, re.IGNORECASE)
        
        # Replace academic phrases
        for formal, alternatives in self.ai_patterns_db['academic_phrases'].items():
            if formal.lower() in sentence.lower():
                replacement = random.choice(alternatives)
                sentence = sentence.replace(formal, replacement)
        
        # Replace structural patterns
        for formal, alternatives in self.ai_patterns_db['structural_patterns'].items():
            if formal.lower() in sentence.lower():
                replacement = random.choice(alternatives)
                sentence = sentence.replace(formal, replacement)
        
        # Replace perfection indicators
        for formal, alternatives in self.ai_patterns_db['perfection_indicators'].items():
            if re.search(r'\b' + re.escape(formal) + r'\b', sentence, re.IGNORECASE):
                replacement = random.choice(alternatives)
                sentence = re.sub(r'\b' + re.escape(formal) + r'\b', replacement, sentence, re.IGNORECASE)
        
        # Ensure we didn't break the sentence
        if len(sentence.split()) < 3:
            return original_sentence
        
        return sentence
    
    def _professional_vocabulary_variation(self, sentence):
        """Professionally vary vocabulary without losing meaning"""
        words = word_tokenize(sentence)
        new_words = []
        
        for word in words:
            original_word = word.lower()
            new_word = word
            
            # Apply synonym replacement with context awareness
            if original_word in self.synonym_chains and random.random() < 0.3:  # Reduced probability
                synonyms = self.synonym_chains[original_word]
                # Choose a synonym that fits the context
                new_word = random.choice(synonyms)
            
            # Apply casual alternatives for formal words
            if original_word in self.human_patterns_db['casual_alternatives'] and random.random() < 0.5:
                new_word = self.human_patterns_db['casual_alternatives'][original_word]
            
            # Preserve capitalization
            if word[0].isupper():
                new_word = new_word.capitalize()
            
            new_words.append(new_word)
        
        return ' '.join(new_words)
    
    def _add_subtle_human_elements(self, sentence):
        """Add subtle human elements without being obvious"""
        words = sentence.split()
        
        # Add occasional conversational elements (very subtle)
        if random.random() < 0.2 and len(words) > 10:  # Reduced probability and increased length requirement
            # Add filler phrase
            filler = random.choice(self.human_patterns_db['filler_phrases'])
            insert_point = random.randint(3, len(words) - 4)  # More conservative insertion
            words.insert(insert_point, filler)
        
        # Add contractions naturally
        sentence = ' '.join(words)
        for formal, contraction in self.human_patterns_db['contractions'].items():
            if random.random() < 0.6:  # Reduced probability
                sentence = re.sub(r'\b' + formal + r'\b', contraction, sentence, re.IGNORECASE)
        
        return sentence
    
    def _ensure_paragraph_coherence_professional(self, sentences):
        """Ensure paragraph maintains professional coherence and flow"""
        if not sentences:
            return ""
        
        # Connect sentences naturally with professional transitions
        connected_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            current_sentence = sentences[i]
            
            # Add professional transitions (less frequent)
            if random.random() < 0.3:  # Reduced probability
                transitions = ['Additionally,', 'Furthermore,', 'Moreover,', 'Similarly,', 'Consequently,']
                current_sentence = random.choice(transitions) + ' ' + current_sentence[0].lower() + current_sentence[1:]
            
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