"""Multi-strategy blocking: UNION of several candidate sets, not intersection.

Any one of these firing is enough to make a pair a candidate — the ML model
downstream decides which candidates are real matches. Recall ceiling comes
from here, so err toward over-generating rather than under-generating.

Country is a genuine OR clause (capped), never a hard filter: a true match
with a blank/inconsistent country label (or one side genuinely missing it)
must still be reachable through name/address/postal signals alone.
"""
from collections import defaultdict, Counter
import pandas as pd
from normalize import normalize_row

MIN_TOKEN_LEN = 3  # ignore super-short tokens as blocking keys (too generic)
MANUAL_STOPWORDS = {"private", "limited", "company", "corporation", "and",
                     "the", "of", "pvt", "ltd", "co", "inc"}
MAX_TOKEN_DOC_FREQ = 0.02   # drop tokens appearing in >2% of the other-source rows
COUNTRY_BLOCK_CAP = 500     # only use "same country" as a candidate source when
                            # that country's pool is small enough not to explode volume


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    norm = df.apply(lambda r: normalize_row(r["business_name"], r["business_address"]), axis=1)
    df["_name"] = norm.apply(lambda d: d["name"])
    df["_addr"] = norm.apply(lambda d: d["address"])
    df["_country_norm"] = df["country"].astype(str).str.strip().str.lower()
    return df


def _frequent_tokens(df: pd.DataFrame, token_col_fn, max_doc_freq=MAX_TOKEN_DOC_FREQ) -> set:
    """Tokens appearing in more than max_doc_freq of rows — computed from the
    actual data instead of guessed, so it adapts to whatever's frequent in
    the real dataset (city names, "restaurant", etc.), not just legal suffixes."""
    n = len(df)
    if n == 0:
        return set()
    counts = Counter()
    for _, row in df.iterrows():
        counts.update(set(token_col_fn(row)))
    return {tok for tok, c in counts.items() if c / n > max_doc_freq}


def _index_by_country(df: pd.DataFrame) -> dict:
    idx = defaultdict(list)
    for i, row in df.iterrows():
        idx[row["_country_norm"]].append(i)
    return idx


def _index_tokens(df: pd.DataFrame, token_col_fn, extra_stopwords: set) -> dict:
    idx = defaultdict(set)
    stop = MANUAL_STOPWORDS | extra_stopwords
    for i, row in df.iterrows():
        for tok in token_col_fn(row):
            if len(tok) >= MIN_TOKEN_LEN and tok not in stop:
                idx[tok].add(i)
    return idx


def generate_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    """Return candidate (s1_idx, other_idx) pairs for one other-source df.

    Blocking keys used (UNION):
      A. same country (capped)
      B. shared informative name token
      C. shared distinctive address token / postal code
      D. exact normalized name
      E. exact postal code (strong address signal)
    """
    s1 = _prep(s1_df)
    oth = _prep(other_df)

    name_stop = _frequent_tokens(oth, lambda r: r["_name"]["tokens"])
    addr_stop = _frequent_tokens(oth, lambda r: r["_addr"]["tokens"])

    country_idx = _index_by_country(oth)
    name_tok_idx = _index_tokens(oth, lambda r: r["_name"]["tokens"], name_stop)
    addr_tok_idx = _index_tokens(oth, lambda r: r["_addr"]["tokens"], addr_stop)
    exact_name_idx = defaultdict(set)
    postal_idx = defaultdict(set)
    for i, row in oth.iterrows():
        exact_name_idx[row["_name"]["expanded"]].add(i)
        if row["_addr"]["postal"]:
            postal_idx[row["_addr"]["postal"]].add(i)

    pairs = set()
    for i, row in s1.iterrows():
        candidates = set()

        # B: name tokens (stopwords now derived from actual token frequency,
        # not a fixed manual list)
        for tok in row["_name"]["tokens"]:
            if len(tok) >= MIN_TOKEN_LEN and tok not in MANUAL_STOPWORDS and tok not in name_stop:
                candidates |= name_tok_idx.get(tok, set())

        # C: address tokens
        for tok in row["_addr"]["tokens"]:
            if len(tok) >= MIN_TOKEN_LEN and tok not in MANUAL_STOPWORDS and tok not in addr_stop:
                candidates |= addr_tok_idx.get(tok, set())

        # D: exact normalized name
        candidates |= exact_name_idx.get(row["_name"]["expanded"], set())

        # E: postal code match
        if row["_addr"]["postal"]:
            candidates |= postal_idx.get(row["_addr"]["postal"], set())

        # A: same country — genuine OR clause now (not an AND filter), so a
        # true match with a blank/odd country label is still reachable via
        # B–E above. Only fire this block when the country pool is small
        # enough that it won't itself blow up candidate volume.
        country_pool = country_idx.get(row["_country_norm"], [])
        if 0 < len(country_pool) <= COUNTRY_BLOCK_CAP:
            candidates |= set(country_pool)

        for j in candidates:
            pairs.add((i, j))

    if not pairs:
        return pd.DataFrame(columns=["s1_idx", "other_idx"])
    return pd.DataFrame(list(pairs), columns=["s1_idx", "other_idx"])