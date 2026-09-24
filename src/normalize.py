"""Normalization for business names and addresses.

Keeps BOTH the raw and normalized/expanded forms — aggressive normalization
can destroy signal (e.g. collapsing "Reliance Digital" and "Reliance Fresh"
into the same token soup), so we return several representations and let the
feature layer decide what's useful.
"""
import re
import string

NAME_ABBR = {
    r"\bpvt\b": "private", r"\bltd\b": "limited", r"\bcorp\b": "corporation",
    r"\bco\b": "company", r"\binc\b": "incorporated", r"&": " and ",
}
ADDR_ABBR = {
    r"\brd\b": "road", r"\bst\b": "street", r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard", r"\bapt\b": "apartment", r"\bfl\b": "floor",
    r"\bsec\b": "sector", r"\bmg\b": "mahatma gandhi", r"\bnr\b": "near",
}
PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))
POSTAL_RE = re.compile(r"\b\d{5,6}\b")


def _basic_clean(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.lower().strip()
    t = t.translate(PUNCT_TABLE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _expand(text: str, mapping: dict) -> str:
    t = text
    for pat, repl in mapping.items():
        t = re.sub(pat, repl, t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_name(raw: str) -> dict:
    clean = _basic_clean(raw)
    expanded = _expand(clean, NAME_ABBR)
    tokens = expanded.split()
    return {
        "raw": raw or "",
        "clean": clean,
        "expanded": expanded,
        "alnum": re.sub(r"[^a-z0-9]", "", expanded),
        "tokens": tokens,
        "sorted_tokens": sorted(tokens),
        "first_token": tokens[0] if tokens else "",
    }


def normalize_address(raw: str) -> dict:
    clean = _basic_clean(raw)
    expanded = _expand(clean, ADDR_ABBR)
    tokens = expanded.split()
    postal = POSTAL_RE.findall(raw or "")
    numeric_tokens = {t for t in tokens if t.isdigit()}
    return {
        "raw": raw or "",
        "clean": clean,
        "expanded": expanded,
        "tokens": tokens,
        "token_set": set(tokens),
        "postal": postal[0] if postal else "",
        "house_numbers": numeric_tokens,
    }


def normalize_row(name: str, address: str) -> dict:
    n = normalize_name(name)
    a = normalize_address(address)
    return {"name": n, "address": a}
