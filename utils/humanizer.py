import re
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from utils.text_analyzer import TextAnalyzer

class ProfessionalHumanizerPro:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        self.ai_patterns = self._build_comprehensive_ai_patterns()
        self.professional_vocabulary = self._build_professional_vocabulary()
        self.syntactic_patterns = self._build_syntactic_patterns()
        self.discourse_markers = self._build_discourse_markers()
        
    def _build_comprehensive_ai_patterns(self):
        """Comprehensive AI pattern database with professional alternatives"""
        return {
            'transitions': {
                'however': ['yet', 'still', 'nevertheless', 'despite this', 'even so'],
                'moreover': ['in addition', 'what\'s more', 'further', 'beyond this'],
                'furthermore': ['additionally', 'in addition', 'what\'s more', 'beyond that'],
                'therefore': ['consequently', 'thus', 'as a result', 'for this reason'],
                'consequently': ['as a result', 'accordingly', 'hence', 'for this reason'],
                'thus': ['therefore', 'accordingly', 'hence', 'as such'],
                'hence': ['therefore', 'accordingly', 'thus', 'as such'],
                'indeed': ['in fact', 'actually', 'certainly', 'undoubtedly'],
                'nonetheless': ['nevertheless', 'even so', 'yet', 'still'],
            },
            'academic_phrases': {
                'it is important to note': ['it\'s worth noting', 'notably', 'significantly', 'crucially'],
                'it is crucial to': ['it\'s essential to', 'it\'s vital that', 'critically'],
                'it is worth noting': ['notably', 'it\'s significant that', 'importantly'],
                'in conclusion': ['ultimately', 'in the end', 'finally', 'to conclude'],
                'in summary': ['in brief', 'to summarize', 'overall', 'in essence'],
                'to summarize': ['in short', 'briefly', 'in essence', 'overall'],
                'it should be noted': ['notably', 'significantly', 'importantly'],
                'it can be seen': ['clearly', 'evidently', 'one can observe', 'this shows'],
            },
            'formal_verbs': {
                'utilize': ['use', 'employ', 'apply', 'implement'],
                'facilitate': ['enable', 'support', 'assist', 'help'],
                'implement': ['apply', 'execute', 'establish', 'deploy'],
                'leverage': ['use', 'exploit', 'capitalize on', 'harness'],
                'commence': ['begin', 'start', 'initiate', 'launch'],
                'terminate': ['end', 'conclude', 'finish', 'complete'],
                'ascertain': ['determine', 'establish', 'verify', 'confirm'],
                'endeavor': ['attempt', 'try', 'strive', 'work'],
            },
            'redundant_phrases': {
                'very unique': 'unique',
                'completely finished': 'finished',
                'absolutely essential': 'essential',
                'totally complete': 'complete',
                'extremely critical': 'critical',
                'highly important': 'important',
            }
        }
    
    def _build_professional_vocabulary(self):
        """Professional alternatives that maintain formality"""
        return {
            'enhancers': {
                'good': ['effective', 'valuable', 'beneficial', 'advantageous', 'productive'],
                'bad': ['problematic', 'challenging', 'concerning', 'detrimental', 'adverse'],
                'big': ['substantial', 'significant', 'considerable', 'extensive', 'major'],
                'small': ['minimal', 'modest', 'limited', 'marginal', 'slight'],
                'important': ['critical', 'essential', 'vital', 'key', 'significant'],
                'new': ['novel', 'innovative', 'emerging', 'recent', 'contemporary'],
                'old': ['established', 'traditional', 'conventional', 'longstanding'],
            },
            'precision_words': {
                'a lot of': ['numerous', 'substantial', 'considerable', 'significant'],
                'many': ['numerous', 'multiple', 'various', 'several'],
                'few': ['limited', 'minimal', 'scarce', 'handful of'],
                'some': ['certain', 'particular', 'specific', 'select'],
            }
        }
    
    def _build_syntactic_patterns(self):
        """Advanced syntactic transformation patterns"""
        return {
            'sentence_openers': [
                'Notably,', 'Significantly,', 'Critically,', 'Importantly,',
                'Interestingly,', 'Remarkably,', 'Essentially,', 'Fundamentally,'
            ],
            'transitional_phrases': [
                'In this context,', 'From this perspective,', 'In light of this,',
                'Building on this,', 'With this in mind,', 'Given these factors,'
            ],
            'hedging_devices': [
                'appears to', 'tends to', 'seems to', 'likely', 'potentially',
                'arguably', 'presumably', 'generally', 'typically'
            ],
            'emphasis_markers': [
                'particularly', 'especially', 'notably', 'specifically',
                'primarily', 'chiefly', 'mainly', 'predominantly'
            ]
        }
    
    def _build_discourse_markers(self):
        """Discourse markers for natural flow"""
        return {
            'addition': ['Furthermore', 'Additionally', 'In addition', 'Moreover'],
            'contrast': ['However', 'Conversely', 'On the other hand', 'In contrast'],
            'cause_effect': ['Consequently', 'As a result', 'Therefore', 'Thus'],
            'example': ['For instance', 'For example', 'Specifically', 'In particular'],
            'clarification': ['That is', 'In other words', 'Namely', 'Specifically'],
        }

    def humanize_professional(self, text, maintain_formality=True):
        """Main humanization method with professional quality preservation"""
        if not text or len(text.strip()) < 10:
            return text
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        processed_paragraphs = []
        
        for para_idx, paragraph in enumerate(paragraphs):
            humanized = self._process_paragraph_advanced(paragraph, para_idx, maintain_formality)
            processed_paragraphs.append(humanized)
        
        return '\n\n'.join(processed_paragraphs)
    
    def _process_paragraph_advanced(self, paragraph, para_index, maintain_formality):
        """Advanced paragraph processing with context awareness"""
        sentences = sent_tokenize(paragraph)
        if not sentences:
            return paragraph
        
        transformed_sentences = []
        
        for sent_idx, sentence in enumerate(sentences):
            context = {
                'is_first': sent_idx == 0,
                'is_last': sent_idx == len(sentences) - 1,
                'paragraph_position': para_index,
                'sentence_position': sent_idx
            }
            
            transformed = self._transform_sentence_professional(
                sentence, context, maintain_formality
            )
            transformed_sentences.append(transformed)
        
        return ' '.join(transformed_sentences)
    
    def _transform_sentence_professional(self, sentence, context, maintain_formality):
        """Professional sentence transformation with quality preservation"""
        if len(sentence.split()) < 4:
            return sentence
        
        original = sentence
        working_sentence = sentence
        
        # Step 1: Replace AI-typical patterns with professional alternatives
        working_sentence = self._replace_ai_patterns(working_sentence, maintain_formality)
        
        # Step 2: Enhance vocabulary precision
        working_sentence = self._enhance_vocabulary(working_sentence)
        
        # Step 3: Vary sentence structure
        working_sentence = self._vary_structure(working_sentence, context)
        
        # Step 4: Add contextual discourse markers
        working_sentence = self._add_discourse_markers(working_sentence, context)
        
        # Step 5: Apply subtle linguistic variations
        working_sentence = self._apply_linguistic_variations(working_sentence)
        
        # Quality validation
        if self._validate_transformation(original, working_sentence, maintain_formality):
            return working_sentence
        
        # Fallback to safer transformation
        return self._safe_transform(original, maintain_formality)
    
    def _replace_ai_patterns(self, sentence, maintain_formality):
        """Replace AI-typical patterns while maintaining professionalism"""
        # Replace formal transitions
        for pattern, alternatives in self.ai_patterns['transitions'].items():
            pattern_regex = r'\b' + re.escape(pattern) + r'\b'
            if re.search(pattern_regex, sentence, re.IGNORECASE):
                replacement = random.choice(alternatives)
                sentence = re.sub(pattern_regex, replacement, sentence, 
                                flags=re.IGNORECASE, count=1)
        
        # Replace academic phrases
        for phrase, alternatives in self.ai_patterns['academic_phrases'].items():
            if phrase.lower() in sentence.lower():
                replacement = random.choice(alternatives)
                sentence = sentence.replace(phrase, replacement)
                sentence = sentence.replace(phrase.capitalize(), replacement.capitalize())
        
        # Replace formal verbs
        for formal, alternatives in self.ai_patterns['formal_verbs'].items():
            pattern_regex = r'\b' + re.escape(formal) + r'\b'
            if re.search(pattern_regex, sentence, re.IGNORECASE):
                # Choose more professional alternatives when maintaining formality
                replacement = random.choice(alternatives[:2] if maintain_formality else alternatives)
                sentence = re.sub(pattern_regex, replacement, sentence, 
                                flags=re.IGNORECASE, count=1)
        
        # Remove redundant phrases
        for redundant, concise in self.ai_patterns['redundant_phrases'].items():
            sentence = sentence.replace(redundant, concise)
            sentence = sentence.replace(redundant.capitalize(), concise.capitalize())
        
        return sentence
    
    def _enhance_vocabulary(self, sentence):
        """Enhance vocabulary with precise, professional alternatives"""
        words = sentence.split()
        
        # Replace vague words with precise alternatives
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            
            # Check enhancers
            if word_lower in self.professional_vocabulary['enhancers']:
                alternatives = self.professional_vocabulary['enhancers'][word_lower]
                replacement = random.choice(alternatives)
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                words[i] = word.replace(word_lower, replacement)
        
        sentence = ' '.join(words)
        
        # Replace imprecise phrases
        for vague, precise_list in self.professional_vocabulary['precision_words'].items():
            if vague in sentence.lower():
                replacement = random.choice(precise_list)
                sentence = re.sub(r'\b' + re.escape(vague) + r'\b', replacement, 
                                sentence, flags=re.IGNORECASE)
        
        return sentence
    
    def _vary_structure(self, sentence, context):
        """Vary sentence structure for natural variation"""
        words = sentence.split()
        
        # Don't modify very short sentences
        if len(words) < 6:
            return sentence
        
        # Occasionally invert sentence structure (20% probability)
        if random.random() < 0.2 and not context['is_first']:
            # Try to move adverbial phrases to the beginning
            adverb_match = re.search(r',\s+(\w+ly)\s+', sentence)
            if adverb_match:
                adverb = adverb_match.group(1)
                sentence = f"{adverb.capitalize()}, {sentence.replace(', ' + adverb, '')}"
        
        # Add subordinate clauses occasionally (15% probability)
        if random.random() < 0.15 and len(words) > 8:
            hedging = random.choice(self.syntactic_patterns['hedging_devices'])
            # Insert hedging device before main verb
            verb_patterns = [' is ', ' are ', ' was ', ' were ', ' has ', ' have ']
            for verb_pattern in verb_patterns:
                if verb_pattern in sentence:
                    sentence = sentence.replace(verb_pattern, f' {hedging} {verb_pattern.strip()} ', 1)
                    break
        
        return sentence
    
    def _add_discourse_markers(self, sentence, context):
        """Add contextual discourse markers for flow"""
        # Only add to non-first sentences occasionally
        if context['sentence_position'] > 0 and random.random() < 0.25:
            # Don't add if sentence already starts with a marker
            first_word = sentence.split()[0].rstrip(',')
            if first_word not in ['However', 'Moreover', 'Furthermore', 'Therefore', 
                                 'Thus', 'Consequently', 'Additionally', 'Indeed']:
                marker = random.choice(self.syntactic_patterns['transitional_phrases'])
                sentence = f"{marker} {sentence[0].lower()}{sentence[1:]}"
        
        return sentence
    
    def _apply_linguistic_variations(self, sentence):
        """Apply subtle linguistic variations"""
        # Add emphasis markers occasionally (20% probability)
        if random.random() < 0.2:
            emphasis = random.choice(self.syntactic_patterns['emphasis_markers'])
            # Insert before key adjectives or nouns
            words = sentence.split()
            for i in range(1, len(words) - 1):
                if words[i].lower() in ['important', 'critical', 'significant', 
                                       'essential', 'key', 'relevant']:
                    words.insert(i, emphasis)
                    sentence = ' '.join(words)
                    break
        
        # Vary punctuation slightly
        sentence = self._vary_punctuation(sentence)
        
        return sentence
    
    def _vary_punctuation(self, sentence):
        """Vary punctuation for natural variation"""
        # Replace some periods with semicolons where appropriate (10% probability)
        if random.random() < 0.1:
            # Look for coordinating conjunctions that could use semicolons
            conjunctions = [' and ', ' but ', ' yet ', ' or ']
            for conj in conjunctions:
                if conj in sentence and sentence.count(conj) == 1:
                    parts = sentence.split(conj)
                    if len(parts[0].split()) > 4 and len(parts[1].split()) > 4:
                        sentence = f"{parts[0]};{parts[1]}"
                        break
        
        return sentence
    
    def _validate_transformation(self, original, transformed, maintain_formality):
        """Validate that transformation maintains quality and meaning"""
        # Length check (should be within reasonable bounds)
        orig_len = len(original.split())
        trans_len = len(transformed.split())
        
        if trans_len < orig_len * 0.75 or trans_len > orig_len * 1.4:
            return False
        
        # Meaning preservation check
        if not self._check_meaning_preservation(original, transformed):
            return False
        
        # Formality check
        if maintain_formality and not self._check_formality_level(transformed):
            return False
        
        return True
    
    def _check_meaning_preservation(self, original, transformed):
        """Check if core meaning is preserved"""
        # Extract key content words
        orig_words = set(w.lower() for w in original.split() 
                        if len(w) > 3 and w.isalpha())
        trans_words = set(w.lower() for w in transformed.split() 
                         if len(w) > 3 and w.isalpha())
        
        # Calculate overlap
        if not orig_words:
            return True
        
        overlap = len(orig_words & trans_words) / len(orig_words)
        return overlap > 0.5
    
    def _check_formality_level(self, sentence):
        """Check if sentence maintains professional formality"""
        informal_markers = ['gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'yeah', 'nope']
        return not any(marker in sentence.lower() for marker in informal_markers)
    
    def _safe_transform(self, sentence, maintain_formality):
        """Safe fallback transformation"""
        # Apply only the most reliable transformations
        sentence = self._replace_ai_patterns(sentence, maintain_formality)
        sentence = self._enhance_vocabulary(sentence)
        return sentence
    
    def humanize_text(self, text, maintain_formality=True):
        """Alias for main humanization method"""
        return self.humanize_professional(text, maintain_formality)
    
    def get_humanization_report(self, original_text, humanized_text):
        """Generate comprehensive humanization report"""
        original_analysis = self.analyzer.analyze_text(original_text)
        humanized_analysis = self.analyzer.analyze_text(humanized_text)
        
        improvements = {
            'readability_change': humanized_analysis['flesch_reading_ease'] - 
                                original_analysis['flesch_reading_ease'],
            'lexical_diversity_change': humanized_analysis['lexical_diversity'] - 
                                      original_analysis['lexical_diversity'],
            'sentence_variety': abs(humanized_analysis['avg_sentence_length'] - 
                                  original_analysis['avg_sentence_length']),
            'word_count_change': humanized_analysis['word_count'] - 
                               original_analysis['word_count'],
            'grammar_improvement': original_analysis['grammar_errors'] - 
                                 humanized_analysis['grammar_errors'],
            'formality_score': self._calculate_formality_score(humanized_text),
            'ai_pattern_reduction': self._calculate_ai_pattern_reduction(
                original_text, humanized_text
            )
        }
        
        return {
            'original_analysis': original_analysis,
            'humanized_analysis': humanized_analysis,
            'improvements': improvements,
            'quality_score': self._calculate_overall_quality(improvements)
        }
    
    def _calculate_formality_score(self, text):
        """Calculate formality score of text"""
        formal_indicators = ['therefore', 'however', 'moreover', 'consequently']
        informal_indicators = ['gonna', 'wanna', 'kinda', 'yeah']
        
        words = text.lower().split()
        formal_count = sum(1 for word in words if word in formal_indicators)
        informal_count = sum(1 for word in words if word in informal_indicators)
        
        total_words = len(words)
        if total_words == 0:
            return 0.5
        
        return (formal_count - informal_count) / total_words + 0.5
    
    def _calculate_ai_pattern_reduction(self, original, humanized):
        """Calculate reduction in AI patterns"""
        ai_patterns = ['however', 'moreover', 'furthermore', 'therefore', 'thus',
                      'consequently', 'it is important', 'it is crucial']
        
        original_count = sum(original.lower().count(pattern) for pattern in ai_patterns)
        humanized_count = sum(humanized.lower().count(pattern) for pattern in ai_patterns)
        
        if original_count == 0:
            return 0
        
        reduction = (original_count - humanized_count) / original_count
        return max(0, min(1, reduction))
    
    def _calculate_overall_quality(self, improvements):
        """Calculate overall quality score"""
        scores = []
        
        # Readability (should not change drastically)
        readability_impact = abs(improvements['readability_change'])
        readability_score = max(0, 1 - (readability_impact / 20))
        scores.append(readability_score)
        
        # Lexical diversity (slight increase is good)
        diversity_score = min(1, max(0, 0.5 + improvements['lexical_diversity_change'] * 2))
        scores.append(diversity_score)
        
        # AI pattern reduction (high is good)
        scores.append(improvements.get('ai_pattern_reduction', 0.5))
        
        # Formality (should be maintained)
        formality = improvements.get('formality_score', 0.5)
        formality_score = formality if formality > 0.4 else 0.3
        scores.append(formality_score)
        
        return sum(scores) / len(scores)