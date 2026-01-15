"""
VoiceBpe tokenizer for ACE-Step lyric encoding.

Simplified port from ACE-Step that supports:
- Full English text normalization (abbreviations, symbols, basic number expansion)
- Basic support for other languages (lowercase + collapse whitespace)
- Optional enhanced support for Chinese/Korean with pypinyin/hangul_romanize

The tokenizer uses a BPE vocabulary trained for multilingual lyrics.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Union

# Optional imports for advanced language support
try:
    from num2words import num2words
    HAS_NUM2WORDS = True
except ImportError:
    HAS_NUM2WORDS = False

try:
    import pypinyin
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False

try:
    from hangul_romanize import Transliter
    from hangul_romanize.rule import academic
    HAS_HANGUL = True
except ImportError:
    HAS_HANGUL = False

# Default vocab path
DEFAULT_VOCAB_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "vocab.json"
)

# Whitespace normalization
_whitespace_re = re.compile(r"\s+")

# English abbreviations
_abbreviations_en = [
    (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "missus"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

# English symbols
_symbols_en = [
    (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
    for x in [
        ("&", " and "),
        ("@", " at "),
        ("%", " percent "),
        ("#", " hash "),
        ("$", " dollar "),
        ("£", " pound "),
        ("°", " degree "),
    ]
]

# Number patterns
_number_re = re.compile(r"[0-9]+")
_ordinal_re = re.compile(r"([0-9]+)(st|nd|rd|th)")
_comma_number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")

# Simple number words for basic expansion (no num2words dependency)
_simple_numbers = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty", "30": "thirty",
    "40": "forty", "50": "fifty", "60": "sixty", "70": "seventy",
    "80": "eighty", "90": "ninety", "100": "hundred",
}

_ordinal_words = {
    "1": "first", "2": "second", "3": "third", "4": "fourth", "5": "fifth",
    "6": "sixth", "7": "seventh", "8": "eighth", "9": "ninth", "10": "tenth",
    "11": "eleventh", "12": "twelfth", "13": "thirteenth", "14": "fourteenth",
    "15": "fifteenth", "16": "sixteenth", "17": "seventeenth", "18": "eighteenth",
    "19": "nineteenth", "20": "twentieth",
}


def _remove_commas(m):
    """Remove commas from number strings."""
    text = m.group(0)
    if "," in text:
        text = text.replace(",", "")
    return text


def _expand_number_simple(m):
    """Simple number expansion without num2words."""
    num_str = m.group(0)
    if num_str in _simple_numbers:
        return _simple_numbers[num_str]

    # For larger numbers, just spell out digits
    if HAS_NUM2WORDS:
        try:
            return num2words(int(num_str), lang="en")
        except Exception:
            pass

    # Fallback: spell out each digit (always return words, never raw digits)
    return " ".join(_simple_numbers.get(d, "zero") for d in num_str)


def _expand_ordinal_simple(m):
    """Simple ordinal expansion."""
    num_str = m.group(1)
    if num_str in _ordinal_words:
        return _ordinal_words[num_str]

    if HAS_NUM2WORDS:
        try:
            return num2words(int(num_str), ordinal=True, lang="en")
        except Exception:
            pass

    # Fallback: expand the number portion and add "th"
    # Note: We can't pass `m` directly because _expand_number_simple expects
    # the match group(0) to be the number, but m has "Xst/nd/rd/th" pattern
    if num_str in _simple_numbers:
        return _simple_numbers[num_str] + "th"
    # Spell out digits
    return " ".join(_simple_numbers.get(d, "zero") for d in num_str) + "th"


def expand_abbreviations(text: str, lang: str = "en") -> str:
    """Expand common abbreviations."""
    if lang == "en":
        for regex, replacement in _abbreviations_en:
            text = re.sub(regex, replacement, text)
    return text


def expand_symbols(text: str, lang: str = "en") -> str:
    """Expand symbols to words."""
    if lang == "en":
        for regex, replacement in _symbols_en:
            text = re.sub(regex, replacement, text)
            text = text.replace("  ", " ")  # Remove double spaces
    return text.strip()


def expand_numbers(text: str, lang: str = "en") -> str:
    """Expand numbers to words."""
    if lang == "en":
        # Remove commas from numbers
        text = re.sub(_comma_number_re, _remove_commas, text)
        # Expand ordinals first (1st, 2nd, etc.)
        text = re.sub(_ordinal_re, _expand_ordinal_simple, text)
        # Then expand regular numbers
        text = re.sub(_number_re, _expand_number_simple, text)
    return text


def lowercase(text: str) -> str:
    """Convert to lowercase."""
    return text.lower()


def collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace to single space."""
    return re.sub(_whitespace_re, " ", text)


def chinese_transliterate(text: str) -> str:
    """Convert Chinese to pinyin if pypinyin is available."""
    if not HAS_PYPINYIN:
        return text

    return "".join(
        [
            p[0]
            for p in pypinyin.pinyin(
                text,
                style=pypinyin.Style.TONE3,
                heteronym=False,
                neutral_tone_with_five=True,
            )
        ]
    )


def korean_transliterate(text: str) -> str:
    """Convert Korean to romanized form if hangul_romanize is available."""
    if not HAS_HANGUL:
        return text

    r = Transliter(academic)
    return r.translit(text)


def preprocess_text(text: str, lang: str) -> str:
    """
    Preprocess text for tokenization.

    Full support for English with abbreviations, symbols, and numbers.
    Basic support for other languages (lowercase + collapse whitespace).
    Enhanced support for Chinese/Korean if optional dependencies are installed.
    """
    # Remove quotes
    text = text.replace('"', "")

    if lang == "en":
        # Full English normalization
        text = lowercase(text)
        text = expand_numbers(text, lang)
        text = expand_abbreviations(text, lang)
        text = expand_symbols(text, lang)
    elif lang == "zh":
        # Chinese: convert to pinyin if available
        text = lowercase(text)
        if HAS_PYPINYIN:
            text = chinese_transliterate(text)
    elif lang == "ko":
        # Korean: romanize if available
        text = lowercase(text)
        if HAS_HANGUL:
            text = korean_transliterate(text)
    else:
        # Basic normalization for other languages
        text = lowercase(text)

    text = collapse_whitespace(text)
    return text


class VoiceBpeTokenizer:
    """
    BPE tokenizer for ACE-Step lyrics.

    Uses a pretrained BPE vocabulary to tokenize lyrics text.
    Supports multiple languages with varying levels of text normalization.

    Example:
        >>> tokenizer = VoiceBpeTokenizer()
        >>> tokens = tokenizer.encode("Hello world", lang="en")
        >>> text = tokenizer.decode(tokens)
    """

    def __init__(self, vocab_file: Optional[str] = None):
        """
        Initialize tokenizer.

        Args:
            vocab_file: Path to vocab.json file. Defaults to bundled vocab.
        """
        if vocab_file is None:
            vocab_file = DEFAULT_VOCAB_FILE

        self.tokenizer = None
        self.vocab_file = vocab_file

        if vocab_file is not None and os.path.exists(vocab_file):
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(vocab_file)
            except ImportError:
                raise ImportError(
                    "The 'tokenizers' package is required for lyric encoding. "
                    "Install it with: pip install tokenizers"
                )

        # Character limits per language (for warning purposes)
        self.char_limits = {
            "en": 10000,
            "de": 253,
            "fr": 273,
            "es": 239,
            "it": 213,
            "pt": 203,
            "pl": 224,
            "zh": 82,
            "ar": 166,
            "cs": 186,
            "ru": 182,
            "nl": 251,
            "tr": 226,
            "ja": 71,
            "hu": 224,
            "ko": 95,
        }

    def check_input_length(self, txt: str, lang: str) -> None:
        """Check if input text exceeds recommended length for language."""
        lang = lang.split("-")[0]  # Remove region code
        limit = self.char_limits.get(lang, 250)
        if len(txt) > limit:
            print(
                f"Warning: Text length ({len(txt)}) exceeds recommended limit "
                f"({limit}) for language '{lang}'. This might cause truncation."
            )

    def encode(self, txt: str, lang: str = "en") -> List[int]:
        """
        Encode text to token IDs.

        Args:
            txt: Text to encode
            lang: Language code (en, zh, ko, etc.)

        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not initialized. Make sure vocab.json exists "
                f"at {self.vocab_file}"
            )

        # Remove region code
        lang = lang.split("-")[0]

        # Check length
        self.check_input_length(txt, lang)

        # Preprocess
        txt = preprocess_text(txt, lang)

        # Handle Chinese language code
        lang_code = "zh-cn" if lang == "zh" else lang

        # Format for tokenizer: [lang]text with spaces as [SPACE]
        txt = f"[{lang_code}]{txt}"
        txt = txt.replace(" ", "[SPACE]")

        return self.tokenizer.encode(txt).ids

    def decode(self, seq: Union[List[int], "mx.array"], skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs back to text.

        Args:
            seq: Token IDs (list or array)
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        # Convert MLX array to list if needed
        if hasattr(seq, "tolist"):
            seq = seq.tolist()

        txt = self.tokenizer.decode(seq, skip_special_tokens=skip_special_tokens).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")

        return txt

    def batch_decode(
        self,
        sequences: List[List[int]],
        skip_special_tokens: bool = False,
    ) -> List[str]:
        """Decode a batch of token sequences."""
        return [self.decode(seq, skip_special_tokens) for seq in sequences]

    def __len__(self) -> int:
        """Return vocabulary size."""
        if self.tokenizer is None:
            return 0
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self) -> int:
        """Get total number of tokens in vocabulary."""
        if self.tokenizer is None:
            return 0
        return self.tokenizer.get_vocab_size()


# Singleton instance for convenience
_default_tokenizer: Optional[VoiceBpeTokenizer] = None


def get_tokenizer() -> VoiceBpeTokenizer:
    """Get or create the default tokenizer instance."""
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = VoiceBpeTokenizer()
    return _default_tokenizer


if __name__ == "__main__":
    # Quick test
    tokenizer = VoiceBpeTokenizer()

    test_cases = [
        ("Hello Mr. Smith, how are you?", "en"),
        ("I have 50% battery and $20", "en"),
        ("This is a 1st test", "en"),
        ("Dancing through the night", "en"),
    ]

    for text, lang in test_cases:
        tokens = tokenizer.encode(text, lang)
        decoded = tokenizer.decode(tokens)
        print(f"Original: {text}")
        print(f"Tokens: {tokens[:20]}..." if len(tokens) > 20 else f"Tokens: {tokens}")
        print(f"Decoded: {decoded}")
        print()
