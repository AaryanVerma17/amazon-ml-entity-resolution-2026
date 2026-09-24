"""Pairwise feature engineering for candidate (S1, other) pairs."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURE_COLS = [
    "name_lev_ratio", "name_token_sort", "name_token_set",
    "name_jaccard", "name_token_overlap", "name_exact",
    "name_same_first_token", "name_char_ngram_sim",
    "addr_lev_ratio", "addr_token_sort", "addr_token_set",
    "addr_jaccard", "addr_token_overlap", "addr_exact",
    "postal_match", "house_number_overlap",
    "same_country",
    "name_x_addr", "name_plus_addr", "name_minus_addr_abs",
    "name_tfidf_cosine", "addr_tfidf_cosine",
]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _token_overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _tfidf_cosine_batch(texts_a, texts_b):
    """Fit one TF-IDF on the union, return per-row cosine similarity."""
    all_text = list(texts_a) + list(texts_b)
    if not any(all_text):
        return np.zeros(len(texts_a))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    try:
        mat = vec.fit_transform(all_text)
    except ValueError:
        return np.zeros(len(texts_a))
    n = len(texts_a)
    a_mat, b_mat = mat[:n], mat[n:]
    num = np.asarray(a_mat.multiply(b_mat).sum(axis=1)).ravel()
    a_norm = np.sqrt(np.asarray(a_mat.multiply(a_mat).sum(axis=1)).ravel())
    b_norm = np.sqrt(np.asarray(b_mat.multiply(b_mat).sum(axis=1)).ravel())
    denom = a_norm * b_norm
    denom[denom == 0] = 1.0
    return num / denom


def build_features(pairs: pd.DataFrame, s1_norm: list, other_norm: list,
                    s1_country: list, other_country: list) -> pd.DataFrame:
    """pairs has columns s1_idx, other_idx (positions into the norm lists)."""
    rows = []
    name_a_texts, name_b_texts, addr_a_texts, addr_b_texts = [], [], [], []

    for _, p in pairs.iterrows():
        i, j = int(p["s1_idx"]), int(p["other_idx"])
        n1, n2 = s1_norm[i]["name"], other_norm[j]["name"]
        a1, a2 = s1_norm[i]["address"], other_norm[j]["address"]

        name_a_texts.append(n1["expanded"]); name_b_texts.append(n2["expanded"])
        addr_a_texts.append(a1["expanded"]); addr_b_texts.append(a2["expanded"])

        name_lev = Levenshtein.normalized_similarity(n1["expanded"], n2["expanded"])
        addr_lev = Levenshtein.normalized_similarity(a1["expanded"], a2["expanded"])
        name_sort = fuzz.token_sort_ratio(n1["expanded"], n2["expanded"]) / 100
        name_set = fuzz.token_set_ratio(n1["expanded"], n2["expanded"]) / 100
        addr_sort = fuzz.token_sort_ratio(a1["expanded"], a2["expanded"]) / 100
        addr_set = fuzz.token_set_ratio(a1["expanded"], a2["expanded"]) / 100
        name_jac = _jaccard(set(n1["tokens"]), set(n2["tokens"]))
        addr_jac = _jaccard(a1["token_set"], a2["token_set"])
        name_ov = _token_overlap(set(n1["tokens"]), set(n2["tokens"]))
        addr_ov = _token_overlap(a1["token_set"], a2["token_set"])

        rows.append({
            "s1_idx": i, "other_idx": j,
            "name_lev_ratio": name_lev, "name_token_sort": name_sort,
            "name_token_set": name_set, "name_jaccard": name_jac,
            "name_token_overlap": name_ov,
            "name_exact": float(n1["expanded"] == n2["expanded"] and n1["expanded"] != ""),
            "name_same_first_token": float(n1["first_token"] == n2["first_token"] and n1["first_token"] != ""),
            "addr_lev_ratio": addr_lev, "addr_token_sort": addr_sort,
            "addr_token_set": addr_set, "addr_jaccard": addr_jac,
            "addr_token_overlap": addr_ov,
            "addr_exact": float(a1["expanded"] == a2["expanded"] and a1["expanded"] != ""),
            "postal_match": float(a1["postal"] == a2["postal"] and a1["postal"] != ""),
            "house_number_overlap": _jaccard(a1["house_numbers"], a2["house_numbers"]),
            "same_country": float(s1_country[i] == other_country[j]),
        })

    feat = pd.DataFrame(rows)
    if feat.empty:
        return feat

    feat["name_x_addr"] = feat["name_lev_ratio"] * feat["addr_lev_ratio"]
    feat["name_plus_addr"] = feat["name_lev_ratio"] + feat["addr_lev_ratio"]
    feat["name_minus_addr_abs"] = (feat["name_lev_ratio"] - feat["addr_lev_ratio"]).abs()
    feat["name_char_ngram_sim"] = feat["name_lev_ratio"]  # placeholder alias

    feat["name_tfidf_cosine"] = _tfidf_cosine_batch(name_a_texts, name_b_texts)
    feat["addr_tfidf_cosine"] = _tfidf_cosine_batch(addr_a_texts, addr_b_texts)

    return feat
