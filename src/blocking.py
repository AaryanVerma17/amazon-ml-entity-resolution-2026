"""Multi-strategy blocking: UNION of several candidate sets, not intersection.

Any one of these firing is enough to make a pair a candidate — the ML model
downstream decides which candidates are real matches. Recall ceiling comes
from here, so err toward over-generating rather than under-generating.
"""
from collections import defaultdict
import pandas as pd
from normalize import normalize_row

MIN_TOKEN_LEN = 3  # ignore super-short tokens as blocking keys (too generic)
STOPWORD_TOKENS = {"private", "limited", "company", "corporation", "and",
                    "the", "of", "pvt", "ltd", "co", "inc"}


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    norm = df.apply(lambda r: normalize_row(r["business_name"], r["business_address"]), axis=1)
    df["_name"] = norm.apply(lambda d: d["name"])
    df["_addr"] = norm.apply(lambda d: d["address"])
    return df


def _index_by_country(df: pd.DataFrame) -> dict:
    idx = defaultdict(list)
    for i, row in df.iterrows():
        idx[row["country"]].append(i)
    return idx


def _index_tokens(df: pd.DataFrame, token_col_fn) -> dict:
    idx = defaultdict(set)
    for i, row in df.iterrows():
        for tok in token_col_fn(row):
            if len(tok) >= MIN_TOKEN_LEN and tok not in STOPWORD_TOKENS:
                idx[tok].add(i)
    return idx


def generate_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    """Return candidate (s1_idx, other_idx) pairs for one other-source df.

    Blocking keys used (UNION):
      A. same country
      B. shared informative name token
      C. shared distinctive address token / postal code
      D. exact normalized name
      E. exact postal code (strong address signal)
    """
    s1 = _prep(s1_df)
    oth = _prep(other_df)

    country_idx = _index_by_country(oth)
    name_tok_idx = _index_tokens(oth, lambda r: r["_name"]["tokens"])
    addr_tok_idx = _index_tokens(oth, lambda r: r["_addr"]["tokens"])
    exact_name_idx = defaultdict(set)
    postal_idx = defaultdict(set)
    for i, row in oth.iterrows():
        exact_name_idx[row["_name"]["expanded"]].add(i)
        if row["_addr"]["postal"]:
            postal_idx[row["_addr"]["postal"]].add(i)

    pairs = set()
    for i, row in s1.iterrows():
        candidates = set()
        # A: country block, but only combined with at least one weak signal
        # to keep volume sane — pure-country would be everything.
        country_pool = country_idx.get(row["country"], [])

        # B: name tokens
        for tok in row["_name"]["tokens"]:
            if len(tok) >= MIN_TOKEN_LEN and tok not in STOPWORD_TOKENS:
                candidates |= name_tok_idx.get(tok, set())

        # C: address tokens
        for tok in row["_addr"]["tokens"]:
            if len(tok) >= MIN_TOKEN_LEN and tok not in STOPWORD_TOKENS:
                candidates |= addr_tok_idx.get(tok, set())

        # D: exact normalized name
        candidates |= exact_name_idx.get(row["_name"]["expanded"], set())

        # E: postal code match
        if row["_addr"]["postal"]:
            candidates |= postal_idx.get(row["_addr"]["postal"], set())

        # restrict to same-country pool (cheap precision boost; drop this
        # line if country labels turn out to be unreliable in the real data)
        country_set = set(country_pool)
        candidates = candidates & country_set if country_set else candidates

        for j in candidates:
            pairs.add((i, j))

    if not pairs:
        return pd.DataFrame(columns=["s1_idx", "other_idx"])
    return pd.DataFrame(list(pairs), columns=["s1_idx", "other_idx"])
