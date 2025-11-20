import re
import random
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag

class Humanizer:
    def __init__(self):
        """Initialize the humanizer with all necessary components"""
        # Download required NLTK data
        self._setup_nltk()
        
        # AI-typical phrases that scream "I'm AI!"
        self.ai_phrases = {
            "it's important to note that": ["note that", "keep in mind", "remember", ""],
            "it's worth noting that": ["notably", "interestingly", "by the way", ""],
            "in today's digital age": ["nowadays", "today", "these days", "currently"],
            "in conclusion": ["to wrap up", "overall", "bottom line", "so"],
            "furthermore": ["also", "plus", "and", "what's more"],
            "however": ["but", "yet", "though", "still"],
            "therefore": ["so", "thus", "that's why", "hence"],
            "additionally": ["also", "plus", "and", "too", "besides"],
            "consequently": ["as a result", "so", "because of this"],
            "nevertheless": ["still", "even so", "yet", "but"],
            "moreover": ["besides", "also", "plus", "on top of that"],
            "thus": ["so", "therefore", "that's why"],
            "indeed": ["in fact", "actually", "really", "truly"],
            "significantly": ["greatly", "a lot", "substantially", "majorly"],
            "it is essential to": ["you must", "you need to", "make sure to", "don't forget to"],
            "individuals": ["people", "folks", "everyone", "users"],
            "utilize": ["use", "employ", "leverage"],
            "obtain": ["get", "acquire", "grab", "gain"],
            "assist": ["help", "aid", "support"],
            "regarding": ["about", "concerning", "on", "around"],
            "in order to": ["to", "so you can", ""],
            "due to the fact that": ["because", "since", "as"],
            "at this point in time": ["now", "currently", "right now", "today"],
            "for the purpose of": ["for", "to", "so you can"],
            "in the event that": ["if", "when", "should"],
            "prior to": ["before", "ahead of"],
            "subsequent to": ["after", "following", "later"],
            "a number of": ["several", "many", "some", "a few"],
            "in the near future": ["soon", "shortly", "before long"],
            "at the present time": ["now", "currently", "today"],
            "it is recommended that": ["you should", "try to", "consider", "I'd recommend"],
            "one must": ["you should", "you need to", "make sure to"],
        }
        
        # Contractions for natural flow
        self.contractions = {
            "do not": "don't", "does not": "doesn't", "did not": "didn't",
            "is not": "isn't", "are not": "aren't", "was not": "wasn't",
            "were not": "weren't", "have not": "haven't", "has not": "hasn't",
            "had not": "hadn't", "will not": "won't", "would not": "wouldn't",
            "should not": "shouldn't", "could not": "couldn't", "cannot": "can't",
            "it is": "it's", "that is": "that's", "there is": "there's",
            "what is": "what's", "who is": "who's", "where is": "where's",
            "I am": "I'm", "you are": "you're", "we are": "we're",
            "they are": "they're", "I have": "I've", "you have": "you've",
            "we have": "we've", "they have": "they've", "I will": "I'll",
            "you will": "you'll", "we will": "we'll", "they will": "they'll",
        }
        
        # Sentence starters variety
        self.sentence_starters = [
            "Look,", "Well,", "So,", "Now,", "Anyway,", "Actually,", 
            "Honestly,", "Frankly,", "To be fair,", "Here's the thing:"
        ]
        
        # Filler phrases for natural flow
        self.natural_fillers = [
            "you know", "I mean", "basically", "pretty much", "kind of",
            "sort of", "more or less", "in a way", "at the end of the day"
        ]
    
    def _setup_nltk(self):
        """Download required NLTK resources"""
        resources = ['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{resource}')
                except LookupError:
                    try:
                        nltk.data.find(f'taggers/{resource}')
                    except LookupError:
                        try:
                            nltk.download(resource, quiet=True)
                        except:
                            pass
    
    def humanize_text(self, text, intensity="medium"):
        """
        Main humanization engine
        
        Args:
            text: Input text to humanize
            intensity: 'low', 'medium', 'high', or 'maximum'
        
        Returns:
            Humanized text
        """
        if not text or len(text.strip()) == 0:
            return text
        
        result = text
        
        # LEVEL 1: Basic humanization (always applied)
        result = self._remove_ai_phrases(result)
        result = self._add_contractions(result)
        result = self._fix_spacing(result)
        
        # LEVEL 2: Medium intensity
        if intensity in ["medium", "high", "maximum"]:
            result = self._replace_synonyms(result, ratio=0.25)
            result = self._vary_sentence_length(result)
            result = self._add_transition_variety(result)
        
        # LEVEL 3: High intensity
        if intensity in ["high", "maximum"]:
            result = self._replace_synonyms(result, ratio=0.40)
            result = self._restructure_sentences(result)
            result = self._add_natural_flow(result)
            result = self._vary_punctuation(result)
        
        # LEVEL 4: Maximum intensity (nuclear option)
        if intensity == "maximum":
            result = self._deep_paraphrase(result)
            result = self._add_human_quirks(result)
            result = self._inject_personality(result)
            result = self._final_polish(result)
        
        return result.strip()
    
    def _remove_ai_phrases(self, text):
        """Eliminate typical AI-generated phrases"""
        result = text
        for phrase, replacements in self.ai_phrases.items():
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            matches = pattern.findall(result)
            for match in matches:
                replacement = random.choice(replacements)
                # Handle capitalization
                if match[0].isupper() and replacement:
                    replacement = replacement[0].upper() + replacement[1:]
                result = pattern.sub(replacement, result, count=1)
        
        # Clean up double spaces
        result = re.sub(r'\s+', ' ', result)
        return result
    
    def _add_contractions(self, text):
        """Make text conversational with contractions"""
        result = text
        for full, contraction in self.contractions.items():
            # Apply with 85% probability for natural variation
            if random.random() < 0.85:
                pattern = re.compile(r'\b' + re.escape(full) + r'\b', re.IGNORECASE)
                result = pattern.sub(contraction, result)
        return result
    
    def _get_synonyms(self, word, pos=None):
        """Get contextual synonyms from WordNet"""
        synonyms = set()
        
        # Get all synsets for the word
        for syn in wordnet.synsets(word):
            # Filter by POS if provided
            if pos and not syn.pos() == pos:
                continue
            
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                # Avoid the word itself and overly complex synonyms
                if synonym.lower() != word.lower() and len(synonym) < 20:
                    synonyms.add(synonym)
        
        return list(synonyms)[:5]  # Limit to top 5
    
    def _replace_synonyms(self, text, ratio=0.3):
        """Intelligently replace words with synonyms"""
        try:
            sentences = sent_tokenize(text)
            result_sentences = []
            
            for sentence in sentences:
                words = word_tokenize(sentence)
                tagged = pos_tag(words)
                new_words = []
                
                for i, (word, pos) in enumerate(tagged):
                    # Skip short words and punctuation
                    if len(word) <= 3 or word in string.punctuation:
                        new_words.append(word)
                        continue
                    
                    # Replace nouns, verbs, adjectives, adverbs
                    if pos.startswith(('NN', 'VB', 'JJ', 'RB')) and random.random() < ratio:
                        synonyms = self._get_synonyms(word, pos[0].lower())
                        if synonyms:
                            new_words.append(random.choice(synonyms))
                        else:
                            new_words.append(word)
                    else:
                        new_words.append(word)
                
                # Reconstruct sentence
                reconstructed = self._reconstruct_sentence(new_words, sentence)
                result_sentences.append(reconstructed)
            
            return ' '.join(result_sentences)
        except:
            return text
    
    def _reconstruct_sentence(self, words, original):
        """Rebuild sentence with proper punctuation and spacing"""
        if not words:
            return original
        
        result = []
        for i, word in enumerate(words):
            if i == 0:
                result.append(word)
            elif word in string.punctuation or word.startswith("'"):
                # No space before punctuation
                if result:
                    result[-1] += word
            else:
                result.append(word)
        
        sentence = ' '.join(result)
        
        # Ensure proper capitalization
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        
        return sentence
    
    def _vary_sentence_length(self, text):
        """Create burstiness - mix of short and long sentences"""
        sentences = sent_tokenize(text)
        result = []
        i = 0
        
        while i < len(sentences):
            sent_length = len(word_tokenize(sentences[i]))
            
            # Merge short sentences (< 8 words)
            if sent_length < 8 and i < len(sentences) - 1 and random.random() < 0.4:
                connectors = [", and", ", so", ", but", " –", ";"]
                merged = sentences[i].rstrip('.!?') + random.choice(connectors) + " " + sentences[i+1].lower()
                result.append(merged)
                i += 2
            # Split long sentences (> 35 words)
            elif sent_length > 35 and random.random() < 0.5:
                split_sentences = self._split_long_sentence(sentences[i])
                result.extend(split_sentences)
                i += 1
            else:
                result.append(sentences[i])
                i += 1
        
        return ' '.join(result)
    
    def _split_long_sentence(self, sentence):
        """Split overly long sentences"""
        # Look for natural break points
        break_points = [
            (r',\s+and\s+', '. '),
            (r',\s+but\s+', '. However, '),
            (r',\s+which\s+', '. This '),
            (r',\s+because\s+', '. That\'s because '),
            (r'\s+and\s+', '. Also, '),
        ]
        
        for pattern, replacement in break_points:
            if re.search(pattern, sentence, re.IGNORECASE):
                parts = re.split(pattern, sentence, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    part1 = parts[0].strip().rstrip(',') + '.'
                    part2 = parts[1].strip()
                    # Capitalize first letter
                    if part2 and part2[0].islower():
                        part2 = part2[0].upper() + part2[1:]
                    return [part1, part2]
        
        return [sentence]
    
    def _add_transition_variety(self, text):
        """Vary transition words for natural flow"""
        sentences = sent_tokenize(text)
        
        # Don't modify if too few sentences
        if len(sentences) < 3:
            return text
        
        # Randomly add casual starters to some sentences
        for i in range(1, len(sentences)):
            if random.random() < 0.15:  # 15% chance
                sentences[i] = random.choice(self.sentence_starters) + " " + sentences[i].lower()
        
        return ' '.join(sentences)
    
    def _restructure_sentences(self, text):
        """Restructure sentences for variety"""
        sentences = sent_tokenize(text)
        result = []
        
        for sentence in sentences:
            if random.random() < 0.3:  # 30% chance to restructure
                restructured = self._apply_restructuring(sentence)
                result.append(restructured)
            else:
                result.append(sentence)
        
        return ' '.join(result)
    
    def _apply_restructuring(self, sentence):
        """Apply various restructuring techniques"""
        # Move trailing adverbs to the front
        adverb_pattern = r'^(.*?)\s+(\w+ly)([.!?])$'
        match = re.match(adverb_pattern, sentence)
        if match and random.random() < 0.5:
            main_part, adverb, punct = match.groups()
            return f"{adverb.capitalize()}, {main_part.lower()}{punct}"
        
        # Move prepositional phrases
        prep_pattern = r'^(.*?),\s+(in|on|at|for|with|by)\s+([^,]+),(.*)$'
        match = re.match(prep_pattern, sentence, re.IGNORECASE)
        if match and random.random() < 0.5:
            before, prep, phrase, after = match.groups()
            return f"{prep.capitalize()} {phrase}, {before.lower()}{after}"
        
        return sentence
    
    def _add_natural_flow(self, text):
        """Add natural conversational elements"""
        sentences = sent_tokenize(text)
        
        # Add occasional filler phrases
        if len(sentences) > 2 and random.random() < 0.2:
            idx = random.randint(1, len(sentences) - 1)
            filler = random.choice(self.natural_fillers)
            
            # Insert at beginning or middle of sentence
            words = sentences[idx].split()
            if len(words) > 4:
                insert_pos = random.randint(1, min(4, len(words) - 1))
                words.insert(insert_pos, filler + ",")
                sentences[idx] = ' '.join(words)
        
        return ' '.join(sentences)
    
    def _vary_punctuation(self, text):
        """Vary punctuation for more natural feel"""
        result = text
        
        # Occasionally use em dashes instead of commas
        if random.random() < 0.2:
            result = re.sub(r',\s+which\s+', ' – which ', result, count=1)
            result = re.sub(r',\s+and\s+', ' – and ', result, count=1)
        
        # Use semicolons occasionally
        if random.random() < 0.15:
            result = re.sub(r'\.\s+([A-Z])', r'; \1', result, count=1)
        
        return result
    
    def _deep_paraphrase(self, text):
        """Aggressive paraphrasing for maximum humanization"""
        sentences = sent_tokenize(text)
        result = []
        
        for sentence in sentences:
            # Multiple passes of synonym replacement
            paraphrased = self._replace_synonyms(sentence, ratio=0.6)
            
            # Restructure
            paraphrased = self._apply_restructuring(paraphrased)
            
            # Change active/passive voice indicators
            paraphrased = self._adjust_voice(paraphrased)
            
            result.append(paraphrased)
        
        return ' '.join(result)
    
    def _adjust_voice(self, sentence):
        """Adjust voice patterns"""
        # Convert some passive to active (simplified)
        passive_patterns = [
            (r'is\s+(\w+ed)\s+by', r'gets \1 by'),
            (r'was\s+(\w+ed)\s+by', r'got \1 by'),
            (r'are\s+(\w+ed)\s+by', r'get \1 by'),
        ]
        
        for pattern, replacement in passive_patterns:
            if random.random() < 0.4:
                sentence = re.sub(pattern, replacement, sentence)
        
        return sentence
    
    def _add_human_quirks(self, text):
        """Add human writing quirks and imperfections"""
        sentences = sent_tokenize(text)
        
        # Occasionally start with conjunctions (breaking grammar "rules")
        if len(sentences) > 1 and random.random() < 0.2:
            conjunctions = ["And", "But", "So", "Yet", "Or"]
            idx = random.randint(1, len(sentences) - 1)
            sentences[idx] = random.choice(conjunctions) + " " + sentences[idx].lower()
        
        # Add parenthetical asides
        if len(sentences) > 2 and random.random() < 0.25:
            asides = [
                "(at least in my experience)",
                "(in most cases)",
                "(though this varies)",
                "(generally speaking)",
                "(from what I've seen)",
                "(surprisingly enough)",
            ]
            idx = random.randint(0, len(sentences) - 1)
            sentences[idx] = sentences[idx].rstrip('.!?') + " " + random.choice(asides) + "."
        
        return ' '.join(sentences)
    
    def _inject_personality(self, text):
        """Inject subtle personality into the text"""
        result = text
        
        # Add occasional emphasis
        emphasis_words = ["really", "actually", "definitely", "absolutely", "certainly", "quite"]
        sentences = sent_tokenize(result)
        
        if sentences and random.random() < 0.3:
            idx = random.randint(0, len(sentences) - 1)
            words = sentences[idx].split()
            if len(words) > 4:
                insert_pos = random.randint(1, 3)
                words.insert(insert_pos, random.choice(emphasis_words))
                sentences[idx] = ' '.join(words)
        
        result = ' '.join(sentences)
        
        # Occasional rhetorical questions
        if random.random() < 0.15:
            rhetorical = [
                "Right?",
                "Don't you think?",
                "Wouldn't you agree?",
                "Make sense?",
            ]
            result = result.rstrip('.') + ", " + random.choice(rhetorical)
        
        return result
    
    def _final_polish(self, text):
        """Final touches for perfect humanization"""
        result = text
        
        # Fix any double punctuation
        result = re.sub(r'([.!?])\1+', r'\1', result)
        
        # Fix spacing issues
        result = self._fix_spacing(result)
        
        # Ensure proper sentence endings
        if not result.endswith(('.', '!', '?')):
            result += '.'
        
        # Remove any artifacts
        result = re.sub(r'\s+([,.])', r'\1', result)
        result = re.sub(r'\s+', ' ', result)
        
        return result.strip()
    
    def _fix_spacing(self, text):
        """Fix spacing and punctuation issues"""
        # Fix spaces before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Ensure space after punctuation
        text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix quotes
        text = re.sub(r'\s+"', ' "', text)
        text = re.sub(r'"\s+', '" ', text)
        
        return text.strip()
    
    def get_humanization_report(self, original_text, humanized_text):
        """Generate humanization comparison report"""
        return {
            'original_length': len(original_text),
            'humanized_length': len(humanized_text),
            'original_words': len(original_text.split()),
            'humanized_words': len(humanized_text.split()),
            'changes_made': abs(len(original_text) - len(humanized_text)),
            'ai_phrases_removed': self._count_ai_phrases(original_text) - self._count_ai_phrases(humanized_text),
            'contractions_added': humanized_text.count("'") - original_text.count("'"),
        }

    def _count_ai_phrases(self, text):
        """Count AI phrases in text"""
        count = 0
        text_lower = text.lower()
        for phrase in self.ai_phrases.keys():
            count += text_lower.count(phrase.lower())
        return count


# Example usage
if __name__ == "__main__":
    humanizer = Humanizer()
    
    sample_text = """
    It is important to note that artificial intelligence has significantly transformed 
    the way individuals interact with technology. Furthermore, it is essential to 
    understand that these changes will continue to evolve. In today's digital age, 
    one must adapt to these technological advancements.
    """
    
    print("ORIGINAL:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    for intensity in ["low", "medium", "high", "maximum"]:
        print(f"{intensity.upper()} INTENSITY:")
        print(humanizer.humanize_text(sample_text, intensity=intensity))
        print("\n" + "="*50 + "\n")