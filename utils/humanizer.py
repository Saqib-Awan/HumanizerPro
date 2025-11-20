import re
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransformationConfig:
    """Configuration for humanization intensity levels"""
    perplexity_target: float
    burstiness_variance: float
    semantic_shift_tolerance: float
    structure_variation_rate: float
    contraction_rate: float
    personal_voice_rate: float
    
    @classmethod
    def get_config(cls, level: str):
        configs = {
            'subtle': cls(1.2, 0.3, 0.1, 0.2, 0.4, 0.1),
            'moderate': cls(1.5, 0.5, 0.2, 0.4, 0.6, 0.3),
            'aggressive': cls(2.0, 0.7, 0.3, 0.6, 0.8, 0.5)
        }
        return configs.get(level, configs['moderate'])


class AdvancedHumanizer:
    """
    Professional-grade text humanizer using advanced NLP techniques
    Focuses on natural linguistic variation while preserving meaning and quality
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
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
            }
        }
        
        # Natural transition phrases that maintain professionalism
        self.professional_transitions = {
            'addition': [
                'Building on this,', 'What\'s more,', 'On top of that,', 
                'Beyond that,', 'Along with this,', 'In addition,'
            ],
            'contrast': [
                'That said,', 'On the flip side,', 'In contrast,',
                'Conversely,', 'Yet,', 'Still,'
            ],
            'causation': [
                'As a result,', 'This means', 'Consequently,',
                'Because of this,', 'This leads to', 'Which is why'
            ],
            'elaboration': [
                'Put simply,', 'In other words,', 'What this means is',
                'To clarify,', 'Specifically,', 'More precisely,'
            ],
            'emphasis': [
                'Notably,', 'Importantly,', 'Worth noting:',
                'Key point:', 'Critically,', 'Significantly,'
            ]
        }
        
        # Sentence restructuring patterns
        self.restructuring_patterns = {
            'active_variations': [
                lambda s, v, o: f"{s} {v} {o}",
                lambda s, v, o: f"What {s} does is {v} {o}",
                lambda s, v, o: f"{s}'s approach is to {v} {o}",
            ],
            'emphasis_patterns': [
                lambda core: f"The key aspect is that {core}",
                lambda core: f"What stands out is {core}",
                lambda core: f"The critical factor: {core}",
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
            r'\bshe is\b': "she's", r'\blet us\b': "let's"
        }
        
        # AI detection patterns to avoid/transform
        self.ai_patterns_to_avoid = {
            'repetitive_starters': [
                r'^The\s+\w+\s+is\s+',
                r'^This\s+\w+\s+demonstrates\s+',
                r'^It\s+is\s+important\s+to\s+note\s+',
                r'^In\s+conclusion,\s+',
            ],
            'overused_phrases': [
                'it is important to note that',
                'it is worth noting that',
                'it should be emphasized that',
                'as previously mentioned',
                'in today\'s world',
                'in today\'s society',
                'plays a crucial role',
                'plays a vital role',
            ],
            'mechanical_transitions': [
                'firstly', 'secondly', 'thirdly',
                'in conclusion', 'to summarize', 'to conclude'
            ]
        }
        
    def humanize(self, text: str, intensity: str = 'moderate', 
                 preserve_technical: bool = True) -> str:
        """
        Main humanization function with quality preservation
        
        Args:
            text: Input text to humanize
            intensity: 'subtle', 'moderate', or 'aggressive'
            preserve_technical: Whether to preserve technical terms
            
        Returns:
            Humanized text maintaining professional quality
        """
        if not text or len(text.strip()) < 10:
            return text
            
        config = TransformationConfig.get_config(intensity)
        
        # Split into paragraphs and process
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        processed_paragraphs = []
        
        for i, paragraph in enumerate(paragraphs):
            processed = self._process_paragraph(
                paragraph, config, i, len(paragraphs), preserve_technical
            )
            processed_paragraphs.append(processed)
        
        # Ensure paragraph coherence
        result = self._ensure_coherence(processed_paragraphs)
        
        # Quality validation
        if self._quality_check(text, result):
            return result
        else:
            logger.warning("Quality check failed, applying fallback transformation")
            return self._fallback_humanization(text, config)
    
    def _process_paragraph(self, paragraph: str, config: TransformationConfig,
                          para_index: int, total_paras: int, 
                          preserve_technical: bool) -> str:
        """Process individual paragraph with multi-layer transformation"""
        
        sentences = sent_tokenize(paragraph)
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
                transformed = self._enhance_vocabulary(
                    transformed, preserve_technical
                )
                
                # Layer 3: Structural variation
                if random.random() < config.structure_variation_rate:
                    transformed = self._vary_structure(transformed, i, len(sentences))
                
                # Layer 4: Natural contractions
                if random.random() < config.contraction_rate:
                    transformed = self._apply_smart_contractions(transformed)
                
                # Layer 5: Professional voice injection
                if random.random() < config.personal_voice_rate:
                    transformed = self._add_professional_voice(
                        transformed, i, len(sentences)
                    )
                
                # Layer 6: Perplexity and burstiness adjustment
                transformed = self._adjust_linguistic_metrics(
                    transformed, config
                )
                
                # Validate transformation
                if self._sentence_quality_check(original, transformed):
                    processed_sentences.append(transformed)
                else:
                    processed_sentences.append(original)
                    
            except Exception as e:
                logger.error(f"Transformation error: {e}")
                processed_sentences.append(original)
        
        # Connect sentences naturally
        return self._connect_sentences(processed_sentences, config)
    
    def _remove_ai_patterns(self, sentence: str) -> str:
        """Remove common AI writing patterns"""
        
        # Remove overused phrases
        for phrase in self.ai_patterns_to_avoid['overused_phrases']:
            if phrase in sentence.lower():
                # Replace with more natural alternatives
                alternatives = self._get_natural_alternative(phrase)
                sentence = re.sub(
                    re.escape(phrase), 
                    alternatives, 
                    sentence, 
                    flags=re.IGNORECASE
                )
        
        # Fix repetitive sentence starters
        for pattern in self.ai_patterns_to_avoid['repetitive_starters']:
            if re.match(pattern, sentence, re.IGNORECASE):
                sentence = self._restructure_opening(sentence)
        
        return sentence
    
    def _get_natural_alternative(self, phrase: str) -> str:
        """Get natural alternative for AI phrases"""
        alternatives = {
            'it is important to note that': 'Keep in mind:',
            'it is worth noting that': 'Worth mentioning:',
            'it should be emphasized that': 'Importantly,',
            'as previously mentioned': 'As mentioned earlier,',
            'in today\'s world': 'Today,',
            'plays a crucial role': 'is essential',
            'plays a vital role': 'matters significantly',
        }
        return alternatives.get(phrase.lower(), phrase)
    
    def _restructure_opening(self, sentence: str) -> str:
        """Restructure repetitive sentence openings"""
        # Extract core content
        words = sentence.split()
        
        # Common restructuring patterns
        if sentence.lower().startswith('the '):
            # "The system is..." → "This system..." or direct statement
            if random.random() < 0.5:
                return 'This ' + ' '.join(words[1:])
            else:
                # Remove "The X is" structure
                return ' '.join(words[3:]) if len(words) > 3 else sentence
        
        if sentence.lower().startswith('this '):
            # Vary "This" starters
            variations = ['Here,', 'Looking at this,', 'Consider that']
            return random.choice(variations) + ' ' + ' '.join(words[1:])
        
        return sentence
    
    def _enhance_vocabulary(self, sentence: str, preserve_technical: bool) -> str:
        """Context-aware vocabulary enhancement"""
        
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)
        
        enhanced_words = []
        context = self._detect_context(sentence)
        
        for word, pos in pos_tags:
            # Check if word needs replacement
            if word.lower() in self.contextual_replacements['formal_to_professional']:
                replacements = self.contextual_replacements['formal_to_professional'][word.lower()]
                
                # Choose context-appropriate replacement
                if context in replacements:
                    enhanced_words.append(replacements[context])
                else:
                    enhanced_words.append(replacements['default'])
            else:
                # Try synonym replacement for variety (but carefully)
                if pos.startswith('VB') or pos.startswith('JJ'):  # Verbs and adjectives
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
    
    def _detect_context(self, sentence: str) -> str:
        """Detect the context of the sentence (technical, business, general)"""
        technical_indicators = ['system', 'algorithm', 'data', 'function', 'code', 
                               'process', 'implementation', 'configuration']
        business_indicators = ['strategy', 'market', 'revenue', 'customer', 
                              'business', 'organization', 'team', 'goal']
        
        sentence_lower = sentence.lower()
        
        tech_score = sum(1 for ind in technical_indicators if ind in sentence_lower)
        business_score = sum(1 for ind in business_indicators if ind in sentence_lower)
        
        if tech_score > business_score:
            return 'technical'
        elif business_score > tech_score:
            return 'business'
        else:
            return 'default'
    
    def _get_contextual_synonym(self, word: str, pos: str) -> Optional[str]:
        """Get contextual synonym using WordNet"""
        try:
            # Map POS tags to WordNet POS
            wordnet_pos = self._get_wordnet_pos(pos)
            if not wordnet_pos:
                return None
            
            synsets = wordnet.synsets(word, pos=wordnet_pos)
            if not synsets:
                return None
            
            # Get synonyms from first synset
            lemmas = synsets[0].lemmas()
            if len(lemmas) > 1:
                # Choose a synonym that's not the same as original
                synonyms = [l.name().replace('_', ' ') for l in lemmas 
                           if l.name().lower() != word.lower()]
                if synonyms:
                    return random.choice(synonyms)
            
            return None
        except:
            return None
    
    def _get_wordnet_pos(self, treebank_tag: str) -> Optional[str]:
        """Convert Treebank POS tag to WordNet POS tag"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    
    def _vary_structure(self, sentence: str, sent_index: int, 
                       total_sents: int) -> str:
        """Apply structural variation while preserving meaning"""
        
        words = sentence.split()
        
        # Only vary sentences with sufficient length
        if len(words) < 8:
            return sentence
        
        # Different strategies based on sentence position
        if sent_index == 0:
            # First sentence: stronger, more engaging opening
            return self._create_strong_opening(sentence)
        elif sent_index == total_sents - 1:
            # Last sentence: conclusive structure
            return self._create_conclusion(sentence)
        else:
            # Middle sentences: vary structure
            strategies = [
                self._split_compound_sentence,
                self._add_subordinate_clause,
                self._use_appositive,
                self._front_load_context
            ]
            
            strategy = random.choice(strategies)
            try:
                return strategy(sentence)
            except:
                return sentence
    
    def _create_strong_opening(self, sentence: str) -> str:
        """Create engaging opening sentence"""
        
        # If it starts with generic words, restructure
        if re.match(r'^(The|This|It|There)\s+', sentence, re.IGNORECASE):
            # Extract the core message
            words = sentence.split()
            
            # Create more direct opening
            openers = [
                "Here's the thing:",
                "Consider this:",
                "What's key:",
                "Start with this:"
            ]
            
            if random.random() < 0.3:
                return f"{random.choice(openers)} {sentence[0].lower()}{sentence[1:]}"
        
        return sentence
    
    def _create_conclusion(self, sentence: str) -> str:
        """Create natural concluding sentence"""
        
        # Avoid mechanical conclusions
        if any(phrase in sentence.lower() for phrase in 
               ['in conclusion', 'to summarize', 'to conclude']):
            # Replace with more natural conclusions
            natural_conclusions = [
                'Bottom line:',
                'The takeaway:',
                'What this means:',
                'In essence,'
            ]
            
            for phrase in ['in conclusion', 'to summarize', 'to conclude']:
                if phrase in sentence.lower():
                    sentence = re.sub(
                        phrase, 
                        random.choice(natural_conclusions),
                        sentence,
                        flags=re.IGNORECASE
                    )
        
        return sentence
    
    def _split_compound_sentence(self, sentence: str) -> str:
        """Split compound sentence for variety"""
        
        # Look for coordinating conjunctions
        conjunctions = [' and ', ' but ', ' or ', ' so ']
        
        for conj in conjunctions:
            if conj in sentence:
                parts = sentence.split(conj, 1)
                if len(parts) == 2 and len(parts[0].split()) > 3:
                    # Create two sentences with natural connector
                    connector = random.choice(['Also,', 'Plus,', 'Yet,', 'So,'])
                    return f"{parts[0].strip()}. {connector} {parts[1].strip()}"
        
        return sentence
    
    def _add_subordinate_clause(self, sentence: str) -> str:
        """Add subordinate clause for complexity"""
        
        words = sentence.split()
        
        if len(words) > 10 and random.random() < 0.5:
            # Insert subordinate clause
            subordinators = [
                'which', 'that', 'where', 'when'
            ]
            
            insert_point = len(words) // 2
            subordinator = random.choice(subordinators)
            
            # This is simplified - real implementation would be more sophisticated
            return sentence
        
        return sentence
    
    def _use_appositive(self, sentence: str) -> str:
        """Use appositive phrases for sophistication"""
        # Simplified implementation
        return sentence
    
    def _front_load_context(self, sentence: str) -> str:
        """Front-load contextual information"""
        
        # Look for clauses that can be moved to front
        if ', which' in sentence or ', that' in sentence:
            # This is a simplified version
            pass
        
        return sentence
    
    def _apply_smart_contractions(self, sentence: str) -> str:
        """Apply contractions intelligently based on context"""
        
        # Don't contract in very formal contexts
        formal_indicators = ['research', 'study', 'analysis', 'data', 
                            'results', 'findings', 'investigation']
        
        sentence_lower = sentence.lower()
        is_formal = any(ind in sentence_lower for ind in formal_indicators)
        
        # Apply contractions with reduced probability in formal text
        probability = 0.3 if is_formal else 0.8
        
        if random.random() < probability:
            for pattern, contraction in self.smart_contractions.items():
                sentence = re.sub(pattern, contraction, sentence, flags=re.IGNORECASE)
        
        return sentence
    
    def _add_professional_voice(self, sentence: str, sent_index: int,
                                total_sents: int) -> str:
        """Add professional human voice without being too casual"""
        
        # Different approaches based on position
        if sent_index == 0:
            # Opening: set professional tone
            professional_openers = [
                'Let\'s examine', 'Consider', 'Look at', 
                'Think about', 'Start with'
            ]
            
            if random.random() < 0.3 and not sentence.startswith(tuple(professional_openers)):
                opener = random.choice(professional_openers)
                sentence = f"{opener} {sentence[0].lower()}{sentence[1:]}"
        
        else:
            # Middle: add professional transitions
            if random.random() < 0.4:
                transition_type = random.choice(list(self.professional_transitions.keys()))
                transition = random.choice(self.professional_transitions[transition_type])
                sentence = f"{transition} {sentence[0].lower()}{sentence[1:]}"
        
        return sentence
    
    def _adjust_linguistic_metrics(self, sentence: str, 
                                   config: TransformationConfig) -> str:
        """Adjust perplexity and burstiness"""
        
        words = sentence.split()
        
        # Adjust sentence length for burstiness
        # Human writing has varying sentence lengths
        target_variance = config.burstiness_variance
        
        # This is a simplified representation
        # Real implementation would involve more sophisticated analysis
        
        return sentence
    
    def _connect_sentences(self, sentences: List[str], 
                          config: TransformationConfig) -> str:
        """Connect sentences with natural flow"""
        
        if not sentences:
            return ""
        
        result = [sentences[0]]
        
        for i in range(1, len(sentences)):
            current = sentences[i]
            
            # Occasionally use informal connectors for natural flow
            if random.random() < 0.3 and not current.startswith(('And', 'But', 'So', 'Yet')):
                connectors = ['And', 'But', 'So', 'Yet', 'Plus']
                connector = random.choice(connectors)
                current = f"{connector} {current[0].lower()}{current[1:]}"
            
            result.append(current)
        
        return ' '.join(result)
    
    def _ensure_coherence(self, paragraphs: List[str]) -> str:
        """Ensure overall document coherence"""
        
        if not paragraphs:
            return ""
        
        # Add paragraph transitions where appropriate
        result = [paragraphs[0]]
        
        for i in range(1, len(paragraphs)):
            current_para = paragraphs[i]
            
            # Add transitional element to some paragraphs
            if random.random() < 0.4:
                transitions = [
                    'Moving forward,', 'Now,', 'Next,', 'Beyond this,',
                    'Additionally,', 'What\'s more,', 'On another note,'
                ]
                transition = random.choice(transitions)
                current_para = f"{transition} {current_para[0].lower()}{current_para[1:]}"
            
            result.append(current_para)
        
        return '\n\n'.join(result)
    
    def _sentence_quality_check(self, original: str, transformed: str) -> bool:
        """Check if transformed sentence maintains quality"""
        
        # Length check: shouldn't deviate too much
        orig_len = len(original.split())
        trans_len = len(transformed.split())
        
        if trans_len < orig_len * 0.5 or trans_len > orig_len * 2:
            return False
        
        # Should have proper capitalization and punctuation
        if not transformed[0].isupper():
            return False
        
        if not transformed.rstrip().endswith(('.', '!', '?', ':')):
            return False
        
        # Should not have excessive repetition
        words = transformed.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        if unique_ratio < 0.4:  # Too much repetition
            return False
        
        return True
    
    def _quality_check(self, original: str, transformed: str) -> bool:
        """Overall quality check"""
        
        # Length preservation (within reason)
        orig_len = len(original.split())
        trans_len = len(transformed.split())
        
        if trans_len < orig_len * 0.6 or trans_len > orig_len * 1.4:
            logger.warning(f"Length deviation: {orig_len} -> {trans_len}")
            return False
        
        # Ensure it's not empty or too short
        if len(transformed.strip()) < 20:
            return False
        
        # Check for paragraph structure preservation
        orig_paras = len([p for p in original.split('\n\n') if p.strip()])
        trans_paras = len([p for p in transformed.split('\n\n') if p.strip()])
        
        if abs(orig_paras - trans_paras) > 1:
            logger.warning(f"Paragraph structure changed significantly")
            return False
        
        return True
    
    def _fallback_humanization(self, text: str, 
                              config: TransformationConfig) -> str:
        """Fallback to conservative humanization if quality check fails"""
        
        logger.info("Applying fallback humanization")
        
        # Apply only safe transformations
        result = text
        
        # Apply contractions
        for pattern, contraction in self.smart_contractions.items():
            if random.random() < 0.6:
                result = re.sub(pattern, contraction, result, flags=re.IGNORECASE)
        
        # Replace obvious AI phrases
        for phrase in self.ai_patterns_to_avoid['overused_phrases']:
            if phrase in result.lower():
                alternative = self._get_natural_alternative(phrase)
                result = re.sub(
                    re.escape(phrase), 
                    alternative, 
                    result, 
                    flags=re.IGNORECASE
                )
        
        return result
    
    def get_analysis_report(self, original: str, humanized: str) -> Dict:
        """
        Generate comprehensive analysis report
        
        Returns detailed metrics about the transformation
        """
        
        def analyze_text(text):
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            return {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
                'unique_words': len(set(w.lower() for w in words)),
                'lexical_diversity': len(set(w.lower() for w in words)) / len(words) if words else 0,
                'contractions': len(re.findall(r"\w+'\w+", text)),
                'questions': len(re.findall(r'\?', text)),
            }
        
        original_stats = analyze_text(original)
        humanized_stats = analyze_text(humanized)
        
        return {
            'original': original_stats,
            'humanized': humanized_stats,
            'improvements': {
                'lexical_diversity_change': humanized_stats['lexical_diversity'] - original_stats['lexical_diversity'],
                'sentence_variety': abs(humanized_stats['avg_sentence_length'] - original_stats['avg_sentence_length']),
                'naturalness_score': humanized_stats['contractions'] / max(humanized_stats['sentence_count'], 1),
            },
            'quality_preserved': self._quality_check(original, humanized)
        }


# Usage example
if __name__ == "__main__":
    humanizer = AdvancedHumanizer()
    
    sample_text = """
    The implementation of artificial intelligence in modern systems is crucial. 
    It is important to note that these systems utilize advanced algorithms to facilitate 
    optimal performance. Subsequently, the parameters must be configured accordingly. 
    In conclusion, this methodology demonstrates significant improvements.
    """
    
    # Different intensity levels
    subtle = humanizer.humanize(sample_text, intensity='subtle')
    moderate = humanizer.humanize(sample_text, intensity='moderate')
    aggressive = humanizer.humanize(sample_text, intensity='aggressive')
    
    print("SUBTLE:")
    print(subtle)
    print("\nMODERATE:")
    print(moderate)
    print("\nAGGRESSIVE:")
    print(aggressive)
    
    # Get analysis
    report = humanizer.get_analysis_report(sample_text, moderate)
    print("\nANALYSIS REPORT:")
    print(report)