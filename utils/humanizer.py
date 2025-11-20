from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional, List

# Optional dependencies: transformers + torch
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

logger = logging.getLogger(__name__)


Tone = Literal["neutral", "formal", "casual", "simplified"]


@dataclass
class HumanizerConfig:
    """
    Configuration for the Humanizer model.
    """
    model_name: str = "t5-base"
    max_length: int = 512
    num_beams: int = 4
    temperature: float = 0.9
    top_p: float = 0.95
    num_return_sequences: int = 1
    device: Optional[str] = None  # "cpu" | "cuda" | None (auto)


class Humanizer:
    """
    Humanizer that rewrites text using a seq2seq model (e.g., T5).

    If transformers/torch are not available, this class falls back to
    returning the input text unchanged.
    """

    def humanize_text(self, text: str, intensity=None):
        # Map intensity to a tone if you like; for now just ignore it
        tone = "neutral"
        return self.humanize(text, tone=tone)
    
    def __init__(self, config: Optional[HumanizerConfig] = None) -> None:
        self.config = config or HumanizerConfig()

        self._model = None
        self._tokenizer = None

        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None or torch is None:
            logger.warning(
                "transformers/torch not available; Humanizer will "
                "return input text unchanged."
            )
            return

        # Auto-select device if not specified
        if self.config.device is None:
            if torch.cuda.is_available():
                self.config.device = "cuda"
            else:
                self.config.device = "cpu"

        logger.info(
            "Loading Humanizer model '%s' on device '%s'",
            self.config.model_name,
            self.config.device,
        )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name
            ).to(self.config.device)
        except Exception as e:
            logger.exception("Failed to load model '%s': %s", self.config.model_name, e)
            self._model = None
            self._tokenizer = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def humanize(
        self,
        text: str,
        tone: Tone = "neutral",
        max_chunk_chars: int = 1200,
    ) -> str:
        """
        Rewrite `text` to be more natural and fluent, preserving meaning.

        Args:
            text: Input text.
            tone: Target tone: "neutral", "formal", "casual", or "simplified".
            max_chunk_chars: Long texts are split into chunks of about this size
                             to fit into the model context.

        Returns:
            Rewritten (or original) text as a single string.
        """
        if not text or not text.strip():
            return text

        text = text.strip()

        # If the model isn't available, just return the original text.
        if not self._is_model_ready:
            return text

        chunks = self._split_into_chunks(text, max_chunk_chars)
        rewritten_chunks: List[str] = []

        for chunk in chunks:
            try:
                rewritten = self._rewrite_chunk(chunk, tone=tone)
                rewritten_chunks.append(rewritten)
            except Exception as e:
                logger.exception("Error rewriting chunk: %s", e)
                # Fallback to original chunk
                rewritten_chunks.append(chunk)

        # Simple join with double newlines to preserve some structure
        return "\n\n".join(part.strip() for part in rewritten_chunks if part.strip())

    # ------------------------------------------------------------------ #
    # Internal methods
    # ------------------------------------------------------------------ #

    @property
    def _is_model_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _split_into_chunks(self, text: str, max_chunk_chars: int) -> List[str]:
        """
        Roughly split text into chunks by character length, breaking on
        sentence boundaries when possible.
        """
        if len(text) <= max_chunk_chars:
            return [text]

        # A very simple sentence splitter by period/question/exclamation.
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current = []

        current_len = 0
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # +1 for space/newline that's likely to be joined
            if current_len + len(sent) + 1 > max_chunk_chars and current:
                chunks.append(" ".join(current).strip())
                current = [sent]
                current_len = len(sent)
            else:
                current.append(sent)
                current_len += len(sent) + 1

        if current:
            chunks.append(" ".join(current).strip())

        return chunks

    def _build_prompt(self, text: str, tone: Tone) -> str:
        """
        Build a textual instruction for the model.
        """
        if tone == "formal":
            instr = (
                "Rewrite the following text in clear, formal, natural-sounding "
                "English while preserving the original meaning:"
            )
        elif tone == "casual":
            instr = (
                "Rewrite the following text in a more casual, conversational, "
                "and natural-sounding way, preserving the original meaning:"
            )
        elif tone == "simplified":
            instr = (
                "Rewrite the following text in simpler, clearer language that "
                "is easy to understand, while keeping the same meaning:"
            )
        else:  # neutral
            instr = (
                "Rewrite the following text to be clear, fluent, and "
                "natural-sounding, preserving the original meaning:"
            )

        return f"{instr}\n\n{text}"

    def _rewrite_chunk(self, chunk: str, tone: Tone) -> str:
        """
        Run one chunk through the model.
        """
        assert self._is_model_ready

        prompt = self._build_prompt(chunk, tone)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.config.device)

        # Use a mix of beam search and sampling for variety + coherence.
        output_ids = self._model.generate(
            **inputs,
            max_length=self.config.max_length,
            num_beams=self.config.num_beams,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            num_return_sequences=self.config.num_return_sequences,
        )

        decoded = self._tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

        return decoded.strip()