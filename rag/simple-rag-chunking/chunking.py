"""
chunking.py
-----------
Five commonly used text-chunking strategies, written from scratch in plain
Python so beginners can read exactly what each one does (no hidden magic
inside a library).

    1. fixed_size_chunking      -> cut every N characters (with overlap)
    2. sentence_chunking        -> group N sentences together
    3. paragraph_chunking       -> one chunk per paragraph
    4. recursive_chunking       -> try big separators first, then smaller ones
    5. semantic_chunking        -> cut where the *meaning* changes (uses embeddings)

Every function takes a string and returns a list of strings (the chunks).
"""

import re
from typing import Callable, List

import numpy as np


# ---------------------------------------------------------------------------
# Helper: split text into sentences
# ---------------------------------------------------------------------------
def split_into_sentences(text: str) -> List[str]:
    """
    Very small sentence splitter.

    It splits after '.', '!' or '?' when they are followed by whitespace.
    It is NOT perfect (e.g. "Dr. Smith" will be split after "Dr."), but it is
    easy to understand and good enough for learning. Production systems often
    use libraries such as nltk or spaCy for this step.
    """
    text = re.sub(r"\s+", " ", text).strip()          # collapse newlines/spaces
    sentences = re.split(r"(?<=[.!?])\s+", text)      # split after . ! ?
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# 1. Fixed-size chunking
# ---------------------------------------------------------------------------
def fixed_size_chunking(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Cut the text into pieces of exactly `chunk_size` characters.
    Each new chunk starts `chunk_size - chunk_overlap` characters after the
    previous one, so neighbouring chunks share `chunk_overlap` characters.

    Example (chunk_size=10, chunk_overlap=3):
        "abcdefghijklmnopqrst"
        -> "abcdefghij", "hijklmnopq", "opqrst"
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - chunk_overlap
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):          # reached the end of the document
            break
        start += step
    return chunks


# ---------------------------------------------------------------------------
# 2. Sentence-based chunking
# ---------------------------------------------------------------------------
def sentence_chunking(text: str, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> List[str]:
    """
    Split the text into sentences, then group every `sentences_per_chunk`
    sentences into one chunk. `overlap_sentences` sentences are repeated at
    the start of the next chunk so context is not lost at the boundary.
    """
    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be greater than 0")
    if overlap_sentences >= sentences_per_chunk:
        raise ValueError("overlap_sentences must be smaller than sentences_per_chunk")

    sentences = split_into_sentences(text)
    chunks = []
    step = sentences_per_chunk - overlap_sentences
    for start in range(0, len(sentences), step):
        group = sentences[start:start + sentences_per_chunk]
        chunks.append(" ".join(group))
        if start + sentences_per_chunk >= len(sentences):
            break
    return chunks


# ---------------------------------------------------------------------------
# 3. Paragraph-based chunking
# ---------------------------------------------------------------------------
def paragraph_chunking(text: str, min_chars: int = 100, max_chars: int = 1500) -> List[str]:
    """
    Use the author's own structure: a blank line ends a paragraph.

    Two small safety rules keep the chunks useful:
      * very short paragraphs (< min_chars, e.g. a heading like "3. Sick Leave")
        are glued onto the NEXT paragraph, so a heading stays with its content;
      * very long paragraphs (> max_chars) are broken up by sentences so a
        single chunk never becomes huge.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # Rule 1: merge tiny paragraphs (headings) into the following paragraph
    merged, carry = [], ""
    for para in paragraphs:
        para = (carry + "\n" + para).strip() if carry else para
        if len(para) < min_chars:
            carry = para
        else:
            merged.append(para)
            carry = ""
    if carry:                                   # leftover at the very end
        if merged:
            merged[-1] = merged[-1] + "\n" + carry
        else:
            merged.append(carry)

    # Rule 2: break oversized paragraphs on sentence boundaries
    chunks = []
    for para in merged:
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        current = ""
        for sentence in split_into_sentences(para):
            if current and len(current) + 1 + len(sentence) > max_chars:
                chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# 4. Recursive character chunking
# ---------------------------------------------------------------------------
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _merge_pieces(pieces: List[str], separator: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Greedily pack small pieces together until a chunk is (almost) full."""
    chunks, current, current_len = [], [], 0
    sep_len = len(separator)

    for piece in pieces:
        added = len(piece) + (sep_len if current else 0)
        if current and current_len + added > chunk_size:
            chunks.append(separator.join(current))
            # Keep the last few pieces as overlap for the next chunk
            while current and (current_len > chunk_overlap or
                               current_len + sep_len + len(piece) > chunk_size):
                current_len -= len(current[0]) + (sep_len if len(current) > 1 else 0)
                current.pop(0)
            added = len(piece) + (sep_len if current else 0)
        current.append(piece)
        current_len += added

    if current:
        chunks.append(separator.join(current))
    return chunks


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    separator, remaining = separators[0], separators[1:]

    # Last resort: no separator left, cut by characters
    if separator == "":
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

    pieces = []
    for part in text.split(separator):
        if not part.strip():
            continue
        if len(part) > chunk_size:
            # This part is still too big -> try the NEXT (smaller) separator on it
            pieces.extend(_recursive_split(part, chunk_size, chunk_overlap, remaining))
        else:
            pieces.append(part)
    return _merge_pieces(pieces, separator, chunk_size, chunk_overlap)


def recursive_chunking(text: str, chunk_size: int = 500, chunk_overlap: int = 50,
                       separators: List[str] = None) -> List[str]:
    """
    The "smart default" used by many frameworks (e.g. LangChain's
    RecursiveCharacterTextSplitter).

    It first tries to split on the biggest natural boundary (blank line =
    paragraph). Any piece that is still bigger than `chunk_size` is split on
    the next boundary (single newline), then sentences (". "), then words
    (" "), and only as a last resort on raw characters ("").
    Small pieces are then packed back together up to `chunk_size`.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    separators = separators or DEFAULT_SEPARATORS
    return [c.strip() for c in _recursive_split(text, chunk_size, chunk_overlap, separators) if c.strip()]


# ---------------------------------------------------------------------------
# 5. Semantic chunking
# ---------------------------------------------------------------------------
def semantic_chunking(text: str, embed_fn: Callable[[List[str]], List[List[float]]],
                      breakpoint_percentile: int = 90, buffer_size: int = 1) -> List[str]:
    """
    Cut the text where the TOPIC changes, instead of after a fixed length.

    Steps:
      1. Split the text into sentences.
      2. For each sentence build a small "window" = the sentence plus
         `buffer_size` neighbours on each side (this smooths out noise).
      3. Embed every window (turn it into a vector of numbers).
      4. Measure how different each window is from the next one
         (cosine distance = 1 - cosine similarity).
      5. The biggest jumps (above the `breakpoint_percentile` percentile of all
         distances) are treated as topic changes -> start a new chunk there.

    `embed_fn` is any function that turns a list of strings into a list of
    vectors. In the app we pass the same HuggingFace model used for search.
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= 2:
        return [" ".join(sentences)] if sentences else []

    # Step 2: sentence windows
    windows = []
    for i in range(len(sentences)):
        lo, hi = max(0, i - buffer_size), min(len(sentences), i + buffer_size + 1)
        windows.append(" ".join(sentences[lo:hi]))

    # Step 3: embeddings (normalised, so dot product == cosine similarity)
    vectors = np.array(embed_fn(windows), dtype=float)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Step 4: distance between each neighbouring pair
    similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)
    distances = 1 - similarities

    # Step 5: breakpoints where the distance is unusually large
    threshold = np.percentile(distances, breakpoint_percentile)
    breakpoints = [i for i, d in enumerate(distances) if d > threshold]

    chunks, start = [], 0
    for bp in breakpoints:
        chunks.append(" ".join(sentences[start:bp + 1]))
        start = bp + 1
    chunks.append(" ".join(sentences[start:]))
    return [c for c in chunks if c.strip()]
