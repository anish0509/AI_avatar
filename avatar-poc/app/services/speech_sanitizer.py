"""Strips markdown formatting and converts LaTeX math notation into speakable
plain text, so the TTS engine's "read verbatim" instruction (see
realtime_tts.py's VERBATIM_INSTRUCTIONS) never causes it to speak raw '**',
'$', '\\frac', etc. aloud. Regex-based, deliberately not a real markdown/LaTeX
parser -- covers the constructs this project's math/CS RAG corpus and
general LLM answers actually produce, with a safe fallback (strip stray
backslashes/braces) for anything unrecognized rather than leaving TTS to
mangle it. Pure logic, no API calls -- fully unit tested (see
tests/test_speech_sanitizer.py).
"""

import re

_BLOCK_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH = re.compile(r"\$([^$\n]+?)\$")

_GREEK_LETTERS = (
    "alpha", "beta", "gamma", "Gamma", "delta", "Delta", "epsilon", "zeta",
    "eta", "theta", "Theta", "iota", "kappa", "lambda", "Lambda", "mu", "nu",
    "xi", "Xi", "pi", "Pi", "rho", "sigma", "Sigma", "tau", "upsilon", "phi",
    "Phi", "chi", "psi", "Psi", "omega", "Omega",
)
_GREEK_PATTERN = re.compile(r"\\(" + "|".join(_GREEK_LETTERS) + r")\b")

# Applied in order to the isolated text between a pair of $/$$ delimiters.
# Order matters: braced frac/sqrt/sup/sub must consume their own braces
# BEFORE the generic escaped-brace and fallback rules run, or those rules
# would strip the grouping braces those patterns still need to match.
_LATEX_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}"), r"\1 over \2"),
    (re.compile(r"\\sqrt\{([^{}]+)\}"), r"the square root of \1"),
    (re.compile(r"\^\{([^{}]+)\}"), r" to the \1"),
    (re.compile(r"\^([A-Za-z0-9])"), r" to the \1"),
    (re.compile(r"_\{([^{}]+)\}"), r" sub \1"),
    (re.compile(r"_([A-Za-z0-9])"), r" sub \1"),
    (_GREEK_PATTERN, r"\1"),
    (re.compile(r"\\times\b"), " times "),
    (re.compile(r"\\cdot\b"), " times "),
    (re.compile(r"\\leq\b"), " less than or equal to "),
    (re.compile(r"\\geq\b"), " greater than or equal to "),
    (re.compile(r"\\neq\b"), " not equal to "),
    (re.compile(r"\\rightarrow\b"), " maps to "),
    (re.compile(r"\\to\b"), " maps to "),
    (re.compile(r"\\forall\b"), " for all "),
    (re.compile(r"\\exists\b"), " there exists "),
    (re.compile(r"\\in\b"), " in "),
    (re.compile(r"\\l?dots\b"), " and so on"),
    (re.compile(r"\\left"), ""),
    (re.compile(r"\\right"), ""),
    (re.compile(r"\\\{"), " "),
    (re.compile(r"\\\}"), " "),
    (re.compile(r"(?<=\s)\|(?=\s)"), " such that "),
]

# Safe fallback for any LaTeX command not covered above (e.g. \mathbb{R}):
# strip the backslash and any leftover grouping braces rather than letting
# the TTS engine read them aloud as literal punctuation.
_FALLBACK_BACKSLASH = re.compile(r"\\+")
_FALLBACK_BRACES = re.compile(r"[{}]")

_HEADER = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
_STRIKETHROUGH = re.compile(r"~~(.+?)~~")
_INLINE_CODE = re.compile(r"`([^`]+?)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
# Bold/italic require non-whitespace immediately inside the delimiters, so
# "3 * 4 * 5" (multiplication, space-padded) is never mistaken for italics.
_BOLD_STAR = re.compile(r"\*\*(\S(?:.*?\S)?)\*\*")
_BOLD_UNDERSCORE = re.compile(r"__(\S(?:.*?\S)?)__")
_ITALIC_STAR = re.compile(r"\*(\S(?:.*?\S)?)\*")
_ITALIC_UNDERSCORE = re.compile(r"(?<!\w)_(\S(?:.*?\S)?)_(?!\w)")

_WHITESPACE = re.compile(r"\s+")


def sanitize_for_speech(text: str) -> str:
    """Clean one chunked sentence before it's spoken (and, since the browser
    only ever renders plain textContent today, also before it's displayed)."""
    text = _convert_latex(text)
    text = _strip_markdown(text)
    return _normalize_whitespace(text)


def _convert_latex(text: str) -> str:
    text = _BLOCK_MATH.sub(lambda m: _convert_latex_body(m.group(1)), text)
    return _INLINE_MATH.sub(lambda m: _convert_latex_body(m.group(1)), text)


def _convert_latex_body(content: str) -> str:
    for pattern, replacement in _LATEX_RULES:
        content = pattern.sub(replacement, content)
    content = _FALLBACK_BACKSLASH.sub("", content)
    return _FALLBACK_BRACES.sub("", content)


def _strip_markdown(text: str) -> str:
    text = _HEADER.sub("", text)
    text = _LIST_MARKER.sub("", text)
    text = _STRIKETHROUGH.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _BOLD_STAR.sub(r"\1", text)
    text = _BOLD_UNDERSCORE.sub(r"\1", text)
    text = _ITALIC_STAR.sub(r"\1", text)
    return _ITALIC_UNDERSCORE.sub(r"\1", text)


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()
