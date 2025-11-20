import re
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe NLTK imports with fallback
try:
    from nltk.tag import pos_tag
    from nltk.corpus import wordnet
    WORDNET_AVAILABLE = True
except:
    WORDNET_AVAILABLE = False
    logger.warning("WordNet not available, using fallback mode")

# Download required NLTK data safely
def download_nltk_data():
    """Download required NLTK data packages"""
    try:
        import nltk
        required_packages = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                except:
                    logger.warning(f"Could not download {package}")
    except Exception as e:
        logger.warning(f"NLTK data download issue: {e}")

# Try to download NLTK data
download_nltk_data()


@dataclass
class TransformationConfig:
    """Configuration for humanization intensity levels"""
    perplexity_target: float
    burstiness_variance: float
    semantic_shift_tolerance: float
    structure_variation_rate: float
    contraction_rate: float
    professional_voice_rate: float
    
    @classmethod
    def get_config(cls, level: str):
        configs = {
            'subtle': cls(1.2, 0.3, 0.1, 0.2, 0.5, 0.2),
            'moderate': cls(1.5, 0.5, 0.2, 0.4, 0.7, 0.4),
            'aggressive': cls(2.0, 0.7, 0.3, 0.6, 0.9, 0.6),
            'extreme': cls(2.0, 0.7, 0.3, 0.6, 0.9, 0.6)  # Added for compatibility
        }
        return configs.get(level, configs['moderate'])


class AdvancedHumanizer:
    """
    Professional-grade text humanizer using advanced NLP techniques
    Focuses on natural linguistic variation while preserving meaning and quality
    """
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stopwords if NLTK data not available
            self.stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but'}
            logger.warning("Using fallback stopwords")
        
        self._initialize_transformation_databases()
        
    def _initialize_transformation_databases(self):
        """Initialize sophisticated transformation databases"""
        
        # Professional vocabulary with context-aware replacements
        self.contextual_replacements = {
            'formal_to_professional': {
                'utilize': {'default': 'use', 'technical': 'apply', 'business': 'leverage'},
                'facilitate': {'default': 'enable', 'technical': 'streamline', 'business': 'drive'},
                'implement': {'default': 'put in place', 'technical': 'deploy', 'business': 'execute'},
                'optimal': {'default': 'best', 'technical': 'most efficient', 'business': 'ideal'},
                'parameters': {'default': 'settings', 'technical': 'configurations', 'business': 'criteria'},
                'methodology': {'default': 'approach', 'technical': 'framework', 'business': 'strategy'},
                'commence': {'default': 'begin', 'technical': 'initiate', 'business': 'launch'},
                'terminate': {'default': 'end', 'technical': 'conclude', 'business': 'finalize'},
                'approximately': {'default': 'about', 'technical': 'roughly', 'business': 'around'},
                'subsequently': {'default': 'later', 'technical': 'afterwards', 'business': 'then'},
                'demonstrate': {'default': 'show', 'technical': 'illustrate', 'business': 'prove'},
                'acquire': {'default': 'get', 'technical': 'obtain', 'business': 'secure'},
                'assist': {'default': 'help', 'technical': 'support', 'business': 'aid'},
                'require': {'default': 'need', 'technical': 'demand', 'business': 'call for'},
                'numerous': {'default': 'many', 'technical': 'multiple', 'business': 'several'},
            }
        }
        
        # Natural transition phrases that maintain professionalism
        self.professional_transitions = {
            'addition': [
                'Building on this,', 'What\'s more,', 'On top of that,', 
                'Beyond that,', 'Along with this,', 'In addition,', 'Plus,'
            ],
            'contrast': [
                'That said,', 'On the flip side,', 'In contrast,',
                'Conversely,', 'Yet,', 'Still,', 'However,'
            ],
            'causation': [
                'As a result,', 'This means', 'Consequently,',
                'Because of this,', 'This leads to', 'Which is why', 'So'
            ],
            'elaboration': [
                'Put simply,', 'In other words,', 'What this means is',
                'To clarify,', 'Specifically,', 'More precisely,', 'Essentially,'
            ],
            'emphasis': [
                'Notably,', 'Importantly,', 'Worth noting:',
                'Key point:', 'Critically,', 'Significantly,', 'The thing is,'
            ]
        }
        
        # Natural contractions (context-aware)
        self.smart_contractions = {
            r'\bit is\b': "it's", r'\bdo not\b': "don't", r'\bdoes not\b': "doesn't",
            r'\bcannot\b': "can't", r'\bwill not\b': "won't", r'\bhave not\b': "haven't",
            r'\bhas not\b': "hasn't", r'\bhad not\b': "hadn't", r'\bwould not\b': "wouldn't",
            r'\bshould not\b': "shouldn't", r'\bcould not\b': "couldn't",
            r'\bthat is\b': "that's", r'\bwhat is\b': "what's", r'\bwho is\b': "who's",
            r'\bthere is\b': "there's", r'\bthey are\b': "they're", r'\bwe are\b': "we're",
            r'\byou are\b': "you're", r'\bI am\b': "I'm", r'\bhe is\b': "he's",
            r'\bshe is\b': "she's", r'\blet us\b': "let's", r'\bit will\b': "it'll",
            r'\bI will\b': "I'll", r'\byou will\b': "you'll", r'\bthey will\b': "they'll"
        }
        
        # AI detection patterns to avoid/transform
        self.ai_patterns_to_avoid = {
            'overused_phrases': {
                'it is important to note that': 'Keep in mind that',
                'it is worth noting that': 'Worth mentioning:',
                'it should be emphasized that': 'Importantly,',
                'as previously mentioned': 'As mentioned earlier,',
                'in today\'s world': 'Today,',
                'in today\'s society': 'Nowadays,',
                'plays a crucial role': 'is essential',
                'plays a vital role': 'matters greatly',
                'it is essential to': 'We need to',
                'it is necessary to': 'We should',
                'in order to': 'to',
                'due to the fact that': 'because',
                'at this point in time': 'now',
                'for the purpose of': 'to',
                'in the event that': 'if',
                'with regard to': 'about',
                'in relation to': 'about',
                'on the other hand': 'but',
                'as a matter of fact': 'in fact',
            },
            'repetitive_starters': [
                r'^The\s+\w+\s+is\s+',
                r'^This\s+\w+\s+demonstrates\s+',
                r'^It\s+is\s+important\s+to\s+note\s+',
                r'^In\s+conclusion,\s+',
                r'^In\s+summary,\s+',
                r'^To\s+summarize,\s+',
            ]
        }
        
        # Professional voice elements
        self.professional_voice_elements = {
            'openers': [
                'Consider this:', 'Here\'s the key:', 'What matters most:',
                'The critical point:', 'Start here:', 'Think about this:',
                'Let\'s examine', 'Look at', 'Here\'s what happens:'
            ],
            'emphasis': [
                'The key aspect', 'What stands out', 'The main point',
                'What\'s crucial', 'The core issue', 'What really matters'
            ],
            'hedging': [
                'tends to', 'often', 'typically', 'generally',
                'in most cases', 'usually', 'commonly', 'frequently'
            ]
        }
    
    # BACKWARD COMPATIBILITY METHOD
    def humanize_text(self, text: str, intensity: str = 'moderate') -> str:
        """
        Backward compatible method for existing apps
        Calls the main humanize() method
        """
        return self.humanize(text, intensity)
    
    def humanize(self, text: str, intensity: str = 'moderate', 
                 preserve_technical: bool = True) -> str:
        """
        Main humanization function with quality preservation
        
        Args:
            text: Input text to humanize
            intensity: 'subtle', 'moderate', 'aggressive', or 'extreme'
            preserve_technical: Whether to preserve technical terms
            
        Returns:
            Humanized text maintaining professional quality
        """
        try:
            if not text or len(text.strip()) < 10:
                return text
            
            config = TransformationConfig.get_config(intensity)
            
            # Split into paragraphs and process
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            if not paragraphs:
                # Handle single line text
                paragraphs = [text.strip()]
            
            processed_paragraphs = []
            
            for i, paragraph in enumerate(paragraphs):
                try:
                    processed = self._process_paragraph(
                        paragraph, config, i, len(paragraphs), preserve_technical
                    )
                    processed_paragraphs.append(processed)
                except Exception as e:
                    logger.error(f"Error processing paragraph {i}: {e}")
                    processed_paragraphs.append(paragraph)  # Keep original on error
            
            # Ensure paragraph coherence
            result = self._ensure_coherence(processed_paragraphs)
            
            # Quality validation
            if self._quality_check(text, result):
                return result
            else:
                logger.warning("Quality check failed, applying fallback transformation")
                return self._fallback_humanization(text, config)
                
        except Exception as e:
            logger.error(f"Humanization error: {e}")
            # Return original text if everything fails
            return text
    
    def _process_paragraph(self, paragraph: str, config: TransformationConfig,
                          para_index: int, total_paras: int, 
                          preserve_technical: bool) -> str:
        """Process individual paragraph with multi-layer transformation"""
        
        try:
            sentences = sent_tokenize(paragraph)
        except:
            # Fallback sentence splitting
            sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
        
        if not sentences:
            return paragraph
        
        processed_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Skip very short sentences
            if len(sentence.split()) < 3:
                processed_sentences.append(sentence)
                continue
            
            # Multi-layer transformation
            transformed = sentence
            original = sentence  # Backup
            
            try:
                # Layer 1: Remove AI fingerprints
                transformed = self._remove_ai_patterns(transformed)
                
                # Layer 2: Context-aware vocabulary enhancement
                transformed = self._enhance_vocabulary(transformed, preserve_technical)
                
                # Layer 3: Structural variation
                if random.random() < config.structure_variation_rate:
                    transformed = self._vary_structure(transformed, i, len(sentences))
                
                # Layer 4: Natural contractions
                if random.random() < config.contraction_rate:
                    transformed = self._apply_smart_contractions(transformed)
                
                # Layer 5: Professional voice injection
                if random.random() < config.professional_voice_rate:
                    transformed = self._add_professional_voice(transformed, i, len(sentences))
                
                # Validate transformation
                if self._sentence_quality_check(original, transformed):
                    processed_sentences.append(transformed)
                else:
                    processed_sentences.append(original)
                    
            except Exception as e:
                logger.error(f"Sentence transformation error: {e}")
                processed_sentences.append(original)
        
        # Connect sentences naturally
        return self._connect_sentences(processed_sentences, config)
    
    def _remove_ai_patterns(self, sentence: str) -> str:
        """Remove common AI writing patterns"""
        
        # Remove overused phrases
        for phrase, replacement in self.ai_patterns_to_avoid['overused_phrases'].items():
            if phrase.lower() in sentence.lower():
                sentence = re.sub(
                    re.escape(phrase), 
                    replacement, 
                    sentence, 
                    flags=re.IGNORECASE
                )
        
        # Fix repetitive sentence starters
        for pattern in self.ai_patterns_to_avoid['repetitive_starters']:
            if re.match(pattern, sentence, re.IGNORECASE):
                sentence = self._restructure_opening(sentence)
        
        return sentence
    
    def _restructure_opening(self, sentence: str) -> str:
        """Restructure repetitive sentence openings"""
        words = sentence.split()
        
        if not words:
            return sentence
        
        # Handle "The X is..." structure
        if sentence.lower().startswith('the ') and len(words) > 3:
            if random.random() < 0.5:
                return 'This ' + ' '.join(words[1:])
            else:
                # Try to remove "The X is" structure if it makes sense
                if len(words) > 4 and words[2].lower() in ['is', 'are', 'was', 'were']:
                    return ' '.join(words[3:])
        
        # Handle "This X demonstrates..." structure
        if sentence.lower().startswith('this ') and 'demonstrates' in sentence.lower():
            sentence = re.sub(r'This \w+ demonstrates that', 'This shows', sentence, flags=re.IGNORECASE)
        
        # Handle "It is important to note"
        if sentence.lower().startswith('it is important'):
            sentence = re.sub(r'It is important to note that', 'Keep in mind:', sentence, flags=re.IGNORECASE)
        
        return sentence
    
    def _enhance_vocabulary(self, sentence: str, preserve_technical: bool) -> str:
        """Context-aware vocabulary enhancement"""
        
        context = self._detect_context(sentence)
        
        for formal_word, replacements in self.contextual_replacements['formal_to_professional'].items():
            pattern = r'\b' + re.escape(formal_word) + r'\b'
            if re.search(pattern, sentence, re.IGNORECASE):
                # Choose context-appropriate replacement
                if context in replacements:
                    replacement = replacements[context]
                else:
                    replacement = replacements['default']
                
                sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE, count=1)
        
        # Additional synonym replacement using WordNet if available
        if WORDNET_AVAILABLE and random.random() < 0.3:
            try:
                sentence = self._apply_wordnet_synonyms(sentence)
            except:
                pass
        
        return sentence
    
    def _apply_wordnet_synonyms(self, sentence: str) -> str:
        """Apply WordNet-based synonym replacement"""
        if not WORDNET_AVAILABLE:
            return sentence
        
        try:
            words = word_tokenize(sentence)
            pos_tags = pos_tag(words)
            
            enhanced_words = []
            
            for word, pos in pos_tags:
                # Only replace adjectives and verbs occasionally
                if (pos.startswith('JJ') or pos.startswith('VB')) and random.random() < 0.2:
                    synonym = self._get_contextual_synonym(word, pos)
                    enhanced_words.append(synonym if synonym else word)
                else:
                    enhanced_words.append(word)
            
            # Reconstruct sentence
            result = ' '.join(enhanced_words)
            
            # Fix spacing around punctuation
            result = re.sub(r'\s+([.,;:!?])', r'\1', result)
            result = re.sub(r'([.,;:!?])(\w)', r'\1 \2', result)
            
            return result
        except:
            return sentence
    
    def _get_contextual_synonym(self, word: str, pos: str) -> Optional[str]:
        """Get contextual synonym using WordNet"""
        if not WORDNET_AVAILABLE:
            return None
        
        try:
            from nltk.corpus import wordnet
            
            # Map POS tags to WordNet POS
            if pos.startswith('J'):
                wordnet_pos = wordnet.ADJ
            elif pos.startswith('V'):
                wordnet_pos = wordnet.VERB
            elif pos.startswith('N'):
                wordnet_pos = wordnet.NOUN
            elif pos.startswith('R'):
                wordnet_pos = wordnet.ADV
            else:
                return None
            
            synsets = wordnet.synsets(word, pos=wordnet_pos)
            if not synsets:
                return None
            
            # Get synonyms from first synset
            lemmas = synsets[0].lemmas()
            if len(lemmas) > 1:
                synonyms = [l.name().replace('_', ' ') for l in lemmas 
                           if l.name().lower() != word.lower()]
                if synonyms:
                    return random.choice(synonyms[:3])  # Limit to top 3
            
            return None
        except:
            return None
    
    def _detect_context(self, sentence: str) -> str:
        """Detect the context of the sentence (technical, business, general)"""
        technical_indicators = ['system', 'algorithm', 'data', 'function', 'code', 
                               'process', 'implementation', 'configuration', 'software',
                               'hardware', 'network', 'database', 'application']
        business_indicators = ['strategy', 'market', 'revenue', 'customer', 
                              'business', 'organization', 'team', 'goal', 'sales',
                              'management', 'company', 'client', 'service']
        
        sentence_lower = sentence.lower()
        
        tech_score = sum(1 for ind in technical_indicators if ind in sentence_lower)
        business_score = sum(1 for ind in business_indicators if ind in sentence_lower)
        
        if tech_score > business_score and tech_score > 0:
            return 'technical'
        elif business_score > tech_score and business_score > 0:
            return 'business'
        else:
            return 'default'
    
    def _vary_structure(self, sentence: str, sent_index: int, total_sents: int) -> str:
        """Apply structural variation while preserving meaning"""
        
        words = sentence.split()
        
        # Only vary sentences with sufficient length
        if len(words) < 7:
            return sentence
        
        try:
            # Different strategies based on sentence position
            if sent_index == 0:
                return self._create_strong_opening(sentence)
            elif sent_index == total_sents - 1:
                return self._create_conclusion(sentence)
            else:
                # Try to split compound sentences
                if random.random() < 0.4:
                    return self._split_compound_sentence(sentence)
        except:
            pass
        
        return sentence
    
    def _create_strong_opening(self, sentence: str) -> str:
        """Create engaging opening sentence"""
        
        # If it starts with generic words, add variety
        if re.match(r'^(The|This|It|There)\s+', sentence, re.IGNORECASE):
            if random.random() < 0.3:
                openers = self.professional_voice_elements['openers']
                opener = random.choice(openers)
                return f"{opener} {sentence[0].lower()}{sentence[1:]}"
        
        return sentence
    
    def _create_conclusion(self, sentence: str) -> str:
        """Create natural concluding sentence"""
        
        # Avoid mechanical conclusions
        mechanical_phrases = {
            'in conclusion': 'The bottom line:',
            'to summarize': 'In short:',
            'to conclude': 'Ultimately:',
            'in summary': 'To sum up:'
        }
        
        for phrase, replacement in mechanical_phrases.items():
            if phrase in sentence.lower():
                sentence = re.sub(phrase, replacement, sentence, flags=re.IGNORECASE)
        
        return sentence
    
    def _split_compound_sentence(self, sentence: str) -> str:
        """Split compound sentence for variety"""
        
        # Look for coordinating conjunctions
        conjunctions = {
            ' and ': ('Also,', 'Plus,'),
            ' but ': ('However,', 'Yet,'),
            ' so ': ('Therefore,', 'As a result,'),
        }
        
        for conj, replacements in conjunctions.items():
            if conj in sentence.lower():
                # Split only if both parts are substantial
                parts = sentence.split(conj, 1)
                if len(parts) == 2 and len(parts[0].split()) > 4 and len(parts[1].split()) > 4:
                    connector = random.choice(replacements)
                    return f"{parts[0].strip()}. {connector} {parts[1].strip()}"
        
        return sentence
    
    def _apply_smart_contractions(self, sentence: str) -> str:
        """Apply contractions intelligently based on context"""
        
        # Don't contract in very formal contexts
        formal_indicators = ['research', 'study', 'analysis', 'investigation', 
                            'examination', 'assessment']
        
        is_formal = any(ind in sentence.lower() for ind in formal_indicators)
        
        # Apply contractions with reduced probability in formal text
        probability = 0.4 if is_formal else 0.9
        
        if random.random() < probability:
            for pattern, contraction in self.smart_contractions.items():
                sentence = re.sub(pattern, contraction, sentence, flags=re.IGNORECASE)
        
        return sentence
    
    def _add_professional_voice(self, sentence: str, sent_index: int, total_sents: int) -> str:
        """Add professional human voice"""
        
        # Add transitions to middle sentences
        if sent_index > 0 and random.random() < 0.5:
            transition_type = random.choice(list(self.professional_transitions.keys()))
            transition = random.choice(self.professional_transitions[transition_type])
            
            # Don't add if sentence already starts with a transition
            common_transitions = ['however', 'moreover', 'furthermore', 'additionally', 
                                'therefore', 'consequently', 'thus', 'hence']
            
            if not any(sentence.lower().startswith(t) for t in common_transitions):
                sentence = f"{transition} {sentence[0].lower()}{sentence[1:]}"
        
        return sentence
    
    def _connect_sentences(self, sentences: List[str], config: TransformationConfig) -> str:
        """Connect sentences with natural flow"""
        
        if not sentences:
            return ""
        
        result = [sentences[0]]
        
        for i in range(1, len(sentences)):
            current = sentences[i]
            
            # Occasionally use informal connectors for natural flow
            if random.random() < 0.25:
                connectors = ['And', 'But', 'So', 'Yet', 'Plus']
                # Don't add if already has one
                if not any(current.startswith(c) for c in connectors):
                    connector = random.choice(connectors)
                    current = f"{connector} {current[0].lower()}{current[1:]}"
            
            result.append(current)
        
        return ' '.join(result)
    
    def _ensure_coherence(self, paragraphs: List[str]) -> str:
        """Ensure overall document coherence"""
        
        if not paragraphs:
            return ""
        
        result = [paragraphs[0]]
        
        for i in range(1, len(paragraphs)):
            current_para = paragraphs[i]
            
            # Add transitional element to some paragraphs
            if random.random() < 0.3 and len(paragraphs) > 2:
                transitions = [
                    'Moving forward,', 'Now,', 'Next,', 'Additionally,', 
                    'Beyond this,', 'What\'s more,'
                ]
                transition = random.choice(transitions)
                # Don't add if already starts with transition
                if not current_para.split()[0].rstrip(',') in [t.rstrip(',') for t in transitions]:
                    current_para = f"{transition} {current_para[0].lower()}{current_para[1:]}"
            
            result.append(current_para)
        
        return '\n\n'.join(result)
    
    def _sentence_quality_check(self, original: str, transformed: str) -> bool:
        """Check if transformed sentence maintains quality"""
        
        try:
            # Length check: shouldn't deviate too much
            orig_len = len(original.split())
            trans_len = len(transformed.split())
            
            if trans_len < max(3, orig_len * 0.5) or trans_len > orig_len * 2.5:
                return False
            
            # Should have proper capitalization
            if not transformed or not transformed[0].isupper():
                return False
            
            # Should end with punctuation
            if not transformed.rstrip().endswith(('.', '!', '?', ':', ';')):
                return False
            
            # Should not have excessive repetition
            words = transformed.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.35:  # Too much repetition
                    return False
            
            return True
        except:
            return False
    
    def _quality_check(self, original: str, transformed: str) -> bool:
        """Overall quality check"""
        
        try:
            # Length preservation (within reason)
            orig_len = len(original.split())
            trans_len = len(transformed.split())
            
            if trans_len < orig_len * 0.5 or trans_len > orig_len * 1.6:
                return False
            
            # Ensure it's not empty or too short
            if len(transformed.strip()) < 15:
                return False
            
            # Check for basic structure
            if not transformed[0].isupper():
                return False
            
            return True
        except:
            return False
    
    def _fallback_humanization(self, text: str, config: TransformationConfig) -> str:
        """Fallback to conservative humanization if quality check fails"""
        
        logger.info("Applying conservative fallback humanization")
        
        result = text
        
        try:
            # Apply safe transformations only
            
            # 1. Apply contractions
            for pattern, contraction in list(self.smart_contractions.items())[:10]:
                if random.random() < 0.7:
                    result = re.sub(pattern, contraction, result, flags=re.IGNORECASE)
            
            # 2. Replace obvious AI phrases
            for phrase, replacement in list(self.ai_patterns_to_avoid['overused_phrases'].items())[:5]:
                if phrase in result.lower():
                    result = re.sub(re.escape(phrase), replacement, result, flags=re.IGNORECASE)
            
            # 3. Simple vocabulary replacements
            simple_replacements = {
                'utilize': 'use', 'facilitate': 'help', 'implement': 'set up',
                'optimal': 'best', 'commence': 'begin', 'terminate': 'end'
            }
            for formal, casual in simple_replacements.items():
                result = re.sub(r'\b' + formal + r'\b', casual, result, flags=re.IGNORECASE)
            
        except Exception as e:
            logger.error(f"Fallback humanization error: {e}")
            return text
        
        return result
    
    def get_humanization_report(self, original_text: str, humanized_text: str) -> Dict:
        """
        Generate comprehensive comparison report
        Backward compatible method name
        """
        return self.get_analysis_report(original_text, humanized_text)
    
    def get_analysis_report(self, original: str, humanized: str) -> Dict:
        """Generate comprehensive analysis report"""
        
        def analyze_text(text):
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            try:
                words = word_tokenize(text)
            except:
                words = text.split()
            
            word_count = len(words)
            sentence_count = len(sentences) if sentences else 1
            
            return {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0,
                'unique_words': len(set(w.lower() for w in words)),
                'lexical_diversity': len(set(w.lower() for w in words)) / word_count if word_count > 0 else 0,
                'contractions': len(re.findall(r"\w+'\w+", text)),
                'questions': len(re.findall(r'\?', text)),
                'character_count': len(text)
            }
        
        try:
            original_stats = analyze_text(original)
            humanized_stats = analyze_text(humanized)
            
            return {
                'original': original_stats,
                'humanized': humanized_stats,
                'improvements': {
                    'lexical_diversity_change': round(humanized_stats['lexical_diversity'] - original_stats['lexical_diversity'], 3),
                    'sentence_variety': round(abs(humanized_stats['avg_sentence_length'] - original_stats['avg_sentence_length']), 2),
                    'naturalness_score': round(humanized_stats['contractions'] / max(humanized_stats['sentence_count'], 1), 2),
                    'word_count_change': humanized_stats['word_count'] - original_stats['word_count']
                },
                'quality_preserved': self._quality_check(original, humanized)
            }
        except Exception as e:
            logger.error(f"Analysis report error: {e}")
            return {
                'error': str(e),
                'original': {},
                'humanized': {},
                'improvements': {},
                'quality_preserved': False
            }


# Alias for backward compatibility
UltimateHumanizer = AdvancedHumanizer


# Usage example
if __name__ == "__main__":
    humanizer = AdvancedHumanizer()
    
    sample_text = """
    The implementation of artificial intelligence in modern systems is crucial. 
    It is important to note that these systems utilize advanced algorithms to facilitate 
    optimal performance. Subsequently, the parameters must be configured accordingly. 
    In conclusion, this methodology demonstrates significant improvements.
    """
    
    print("Original:")
    print(sample_text)
    print("\n" + "="*80 + "\n")
    
    # Test with humanize_text (backward compatible)
    result = humanizer.humanize_text(sample_text, intensity='moderate')
    print("Humanized (moderate):")
    print(result)
    print("\n" + "="*80 + "\n")
    
    # Get analysis
    report = humanizer.get_humanization_report(sample_text, result)
    print("Analysis Report:")
    for key, value in report.items():
        print(f"{key}: {value}")