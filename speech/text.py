import re
from .contracts import SpeechError


def split_text(text: str, max_words: int = 20, max_chars: int = 180) -> list[str]:
    if not isinstance(text, str) or not text.strip() or len(text) > 10000:
        raise SpeechError("Provide 1–10000 characters of speech text")
    if max_words < 1 or max_chars < 1:
        raise ValueError("Chunk limits must be positive")
    if any(ord(c) < 32 and c not in '\n\r\t' for c in text):
        raise SpeechError("Text contains unsupported control characters")
    chunks = []
    for sentence in re.split(r'(?<=[.!?])\s+|[\r\n]+', text):
        words = []
        for word in sentence.split():
            if len(word) > max_chars:
                raise SpeechError("A word exceeds the chunk length limit")
            if words and (len(words) >= max_words or len(' '.join(words + [word])) > max_chars):
                chunks.append(' '.join(words))
                words = []
            words.append(word)
        if words:
            chunks.append(' '.join(words))
    if not chunks or not any(c.isalnum() for c in text):
        raise SpeechError("Text contains no spoken content")
    if len(chunks) > 128:
        raise SpeechError("Request exceeds 128 chunks")
    return chunks


def duration_budget(text: str, ceiling: float) -> float:
    # Generous speech allowance, bounded independently for every chunk.
    return min(ceiling, max(6.0, len(text.split()) * .9 + 2.0))
