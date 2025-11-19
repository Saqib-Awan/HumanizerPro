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
        self.human_touch_phrases = self._build_human_touch_phrases()
        
    def _build_comprehensive_ai_patterns(self):
        return {
            'transitions': {
                'however': ['but', 'yet', 'still', 'even so', 'that said', 'having said that', 'on the flip side'],
                'moreover': ['also', 'plus', 'what\'s more', 'on top of that', 'besides'],
                'furthermore': ['also', 'what\'s more', 'additionally', 'beyond that', 'and another thing'],
                'therefore': ['so', 'as a result', 'this means', 'which is why', 'that\'s why'],
                'consequently': ['so', 'as a result', 'this leads to', 'which explains why'],
                'thus': ['so', 'this way', 'accordingly', 'in turn'],
                'hence': ['so', 'this is why', 'that\'s the reason'],
                'indeed': ['in fact', 'actually', 'truth is', 'to be honest'],
                'nonetheless': ['still', 'even so', 'all the same', 'be that as it may'],
            },
            'academic_phrases': {
                'it is important to note': ['one thing to keep in mind', 'worth pointing out', 'something I always emphasize', 'notably'],
                'it is crucial to': ['we must', 'it\'s critical that', 'the key is to', 'you have to'],
                'it is worth noting': ['interestingly', 'curiously', 'something stands out', 'notably'],
                'in conclusion': ['in the end', 'ultimately', 'when all is said and done', 'looking at the bigger picture'],
                'in summary': ['all in all', 'to sum up', 'bottom line', 'in short'],
                'it can be seen that': ['clearly', 'you can see', 'it\'s evident', 'this shows'],
                'it is evident that': ['clearly', 'obviously', 'you can tell', 'it\'s pretty clear'],
            },
            'formal_verbs': {
                'utilize': ['use', 'make use of', 'go with', 'turn to'],
                'facilitate': ['help', 'make easier', 'support', 'enable'],
                'implement': ['put in place', 'roll out', 'carry out', 'apply'],
                'leverage': ['take advantage of', 'use', 'tap into', 'capitalize on'],
                'commence': ['start', 'kick off', 'get going', 'begin'],
                'terminate': ['end', 'wrap up', 'close out', 'finish'],
                'ascertain': ['find out', 'figure out', 'confirm', 'check'],
                'endeavor': ['try', 'work to', 'aim to', 'strive'],
            }
        }

    def _build_professional_vocabulary(self):
        return {
            'enhancers': {
                'good': ['solid', 'strong', 'valuable', 'useful', 'effective', 'practical'],
                'bad': ['troubling', 'concerning', 'problematic', 'unfortunate', 'challenging'],
                'big': ['major', 'huge', 'significant', 'substantial', 'serious'],
                'small': ['minor', 'limited', 'modest', 'slight', 'barely noticeable'],
                'very': ['really', 'truly', 'extremely', 'quite', 'remarkably', 'surprisingly'],
                'important': ['critical', 'key', 'vital', 'crucial', 'essential', 'pivotal'],
            }
        }

    def _build_syntactic_patterns(self):
        return {
            'openers': ['Look,', 'Frankly,', 'Honestly,', 'To be clear,', 'Here\'s the thing,', 'One point I\'d make,'],
            'parentheticals': ['— I think —', ' (at least in my view)', ' — or so it seems —', ' (and this is key)', ' — surprisingly —'],
            'contractions': ['don\'t', 'can\'t', 'won\'t', 'it\'s', 'that\'s', 'there\'s', 'you\'ll', 'we\'re'],
            'fillers': ['well,', 'you know,', 'sort of,', 'kind of,', 'in a way,'],
        }

    def _build_human_touch_phrases(self):
        return [
            "I've found that", "From what I've seen", "In my experience", "One thing that's clear",
            "What stands out to me is", "Something worth considering", "The reality is", "Truth be told",
            "If I'm being honest", "At the end of the day", "When you really think about it"
        ]

    def _build_discourse_markers(self):
        return {
            'addition': ['Also,', 'Plus,', 'And another thing,', 'On top of that,'],
            'contrast': ['But', 'That said,', 'Still,', 'On the other hand,', 'Having said that,'],
            'cause_effect': ['So', 'This is why', 'That\'s why', 'As a result', 'Which means'],
            'example': ['For example,', 'Take this:', 'Like when', 'Think about'],
        }

    def humanize_professional(self, text, maintain_formality=True):
        if not text or len(text.strip()) < 10:
            return text

        random.seed(hash(text) % 2**32)  # Consistent but varied per input
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        result = []

        for para_idx, para in enumerate(paragraphs):
            sentences = sent_tokenize(para)
            processed = []

            for i, sent in enumerate(sentences):
                ctx = {
                    'first_in_para': i == 0,
                    'last_in_para': i == len(sentences) - 1,
                    'para_idx': para_idx,
                    'sent_idx': i
                }
                processed.append(self._transform_sentence_enhanced(sent.strip(), ctx, maintain_formality))
            
            # Add natural paragraph rhythm: vary sentence lengths heavily
            processed = self._apply_burstiness(processed)
            result.append(' '.join(processed))

        final_text = '\n\n'.join(result)
        
        # Final polish pass
        final_text = self._final_natural_polish(final_text)
        return final_text

    def _transform_sentence_enhanced(self, sentence, ctx, maintain_formality):
        if len(sentence) < 15:
            return sentence

        orig = sentence
        s = sentence

        # 1. Replace robotic transitions
        s = self._replace_ai_patterns(s, maintain_formality)

        # 2. Inject human touch (sparingly)
        if random.random() < 0.18 and not ctx['first_in_para']:
            touch = random.choice(self._build_human_touch_phrases())
            s = touch + " " + s[0].lower() + s[1:]

        # 3. Add parentheticals or asides (key for bypassing detectors)
        if random.random() < 0.22 and len(s) > 60:
            aside = random.choice(self.syntactic_patterns['parentheticals'])
            pos = random.randint(30, min(80, len(s)-20))
            s = s[:pos] + aside + s[pos:]

        # 4. Use contractions aggressively
        s = self._apply_contractions(s)

        # 5. Occasional sentence opener
        if not ctx['first_in_para'] and random.random() < 0.3:
            opener = random.choice(['That said,', 'Still,', 'Look,', 'Here\'s the thing,'])
            s = opener + " " + s[0].lower() + s[1:]

        # 6. Vary structure
        s = self._vary_structure_aggressively(s)

        # 7. Final safety check
        if self._validate_transformation(orig, s, maintain_formality):
            return s.capitalize() if s else orig
        else:
            return self._safe_transform(orig, maintain_formality)

    def _apply_contractions(self, text):
        contractions = {
            ' do not ': ' don\'t ',
            ' cannot ': ' can\'t ',
            ' will not ': ' won\'t ',
            ' it is ': ' it\'s ',
            ' that is ': ' that\'s ',
            ' there is ': ' there\'s ',
            ' you will ': ' you\'ll ',
            ' we are ': ' we\'re ',
            ' I have ': ' I\'ve ',
            ' would have ': ' \'d have ',
            ' should have ': ' should\'ve '
        }
        for k, v in contractions.items():
            text = text.replace(k, v)
        return text

    def _vary_structure_aggressively(self, s):
        # Fragment occasionally
        if random.random() < 0.12 and len(s) > 80:
            parts = s.split(',', 1)
            if len(parts) > 1:
                s = parts[0] + ". " + parts[1].strip().capitalize()

        # Start with conjunction sometimes
        if random.random() < 0.15:
            starters = ['And', 'But', 'So', 'Or', 'Nor', 'Yet']
            if not s.startswith(tuple(starters)):
                s = random.choice(starters) + " " + s[0].lower() + s[1:]

        return s

    def _apply_burstiness(self, sentences):
        # Force high burstiness: mix very short + very long sentences
        if len(sentences) < 3:
            return sentences

        new_order = []
        for s in sentences:
            length = len(s.split())
            if length > 25 and random.random() < 0.4:
                # Break long sentence
                mid = len(s) // 2
                break_at = s.rfind(',', 0, mid)
                if break_at == -1:
                    break_at = mid
                if break_at > 20:
                    new_order.append(s[:break_at+1])
                    remainder = s[break_at+1:].strip()
                    if remainder:
                        new_order.append(remainder.capitalize())
                else:
                    new_order.append(s)
            else:
                new_order.append(s)

        return new_order

    def _final_natural_polish(self, text):
        # Replace double spaces, fix capitalization after fragments
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), text)
        text = re.sub(r'\?\s+([a-z])', lambda m: '? ' + m.group(1).upper(), text)
        text = re.sub(r'!\s+([a-z])', lambda m: '! ' + m.group(1).upper(), text)
        return text.strip()

    # Keep all original method names intact for compatibility
    def _replace_ai_patterns(self, sentence, maintain_formality):
        for category in ['transitions', 'academic_phrases', 'formal_verbs']:
            patterns = self.ai_patterns[category]
            for old, options in patterns.items():
                regex = re.compile(r'\b' + re.escape(old) + r'\b', re.IGNORECASE)
                if regex.search(sentence):
                    replacement = random.choice(options)
                    sentence = regex.sub(replacement, sentence, count=1)
        return sentence

    def _enhance_vocabulary(self, sentence):
        for old, options in self.professional_vocabulary['enhancers'].items():
            sentence = re.sub(r'\b' + old + r'\b', lambda m: random.choice(options), sentence, flags=re.IGNORECASE)
        return sentence

    def _validate_transformation(self, original, transformed, maintain_formality):
        if abs(len(transformed) - len(original)) > 0.5 * len(original):
            return False
        if not self._check_meaning_preservation(original, transformed):
            return False
        return True

    def _check_meaning_preservation(self, o, t):
        o_words = set(re.findall(r'\b\w{4,}\b', o.lower()))
        t_words = set(re.findall(r'\b\w{4,}\b', t.lower()))
        if not o_words:
            return True
        return len(o_words & t_words) / len(o_words) > 0.55

    def _safe_transform(self, sentence, maintain_formality):
        s = self._replace_ai_patterns(sentence, maintain_formality)
        s = self._enhance_vocabulary(s)
        s = self._apply_contractions(s)
        return s

    # Public aliases — unchanged
    def humanize_text(self, text, maintain_formality=True):
        return self.humanize_professional(text, maintain_formality)

    def get_humanization_report(self, original_text, humanized_text):
        # Keep original implementation or enhance later
        return {"status": "success", "note": "Enhanced first-pass humanization active"}