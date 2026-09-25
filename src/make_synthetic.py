"""
Amazon ML Challenge 2026 - scalable single-file entity-resolution pipeline.

Design goals for a local Windows machine with ~12.5M training records:
- Training preparation never touches the test data.
- Large artifacts live outside the repository (default: D:\\AmazonMLArtifacts).
- DuckDB is used as the disk-backed working store.
- Candidate generation is performed only for a bounded S1 workset during CV.
- The final candidate table is exactly what the matcher scores.
- Validation is entity-level macro F0.5, matching the challenge definition.
- Test preparation/inference is a separate explicit stage.

Recommended first run:
    python src/pipeline.py --stage audit
    python src/pipeline.py --stage prepare-train
    python src/pipeline.py --stage validate
    # only after validation is satisfactory:
    python src/pipeline.py --stage prepare-test
    python src/pipeline.py --stage predict

The pipeline uses only challenge-provided data. No external lookup/API/geocoding.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = r"6ab10eb3b23ba_student_resource\student_resource\dataset"
DEFAULT_ARTIFACT_DIR = r"D:\AmazonMLArtifacts"
DEFAULT_OUTPUT_DIR = "output"
SEED = 42
VAL_FRACTION = 0.20
# First real run: bounded workset. Increase later after the pipeline is proven.
DEFAULT_WORK_S1 = 40_000
DEFAULT_TRAIN_S1 = 30_000
DEFAULT_VAL_S1 = 10_000
MAX_CANDIDATES_PER_S1 = 150
MAX_CANDIDATES_PER_BLOCK = 40
MAX_TOKEN_POSTINGS = 500
MAX_NAME_POSTINGS = 1000
MAX_FIRST2_POSTINGS = 1000
MAX_POSTAL_POSTINGS = 5000
MAX_ADDR_POSTINGS = 2000
MAX_COMPOSITE_POSTINGS = 3000
FEATURE_BATCH_SIZE = 25_000
MAX_TRAIN_ROWS = 1_500_000
NEGATIVES_PER_S1 = 6
MAX_TOKEN_DOC_FREQ = 0.0015
MIN_TOKEN_LEN = 3
MIN_SHARED_NAME_TOKENS = 2   # require 2+ shared rare name tokens: one weak token isn't enough evidence
MIN_SHARED_ADDR_TOKENS = 1   # addresses have fewer useful tokens; 1 is fine
XGB_N_JOBS = 2               # keep this low - full CPU-1 was overheating a laptop for no real speed gain
BLOCKING_VERSION = "v4_token_composite"  # bump this whenever the blocking logic changes, so
                                          # validation/best_threshold.json records which experiment produced a score
THRESHOLDS = np.round(np.arange(0.30, 0.991, 0.02), 2)

# Every candidate row carries one 0/1 flag per blocking strategy that found it.
# KEY_MAP drives the exact/composite blocks: flag -> (columns forming the key, posting cap).
ALL_FLAGS = [
    "exact_name", "exact_address", "postal", "first2",
    "name_postal", "addr_postal", "name_token", "address_token",
]
KEY_MAP = {
    "exact_name": (("name_norm",), MAX_NAME_POSTINGS),
    "exact_address": (("addr_norm",), MAX_ADDR_POSTINGS),
    "postal": (("postal",), MAX_POSTAL_POSTINGS),
    "first2": (("name_first2",), MAX_FIRST2_POSTINGS),
    "name_postal": (("name_norm", "postal"), MAX_COMPOSITE_POSTINGS),
    "addr_postal": (("addr_norm", "postal"), MAX_COMPOSITE_POSTINGS),
}
# Block-strength weights used to rank candidates before truncating to
# MAX_CANDIDATES_PER_S1, instead of an arbitrary ORDER BY other_id.
BLOCK_STRENGTH = {
    "exact_name": 100, "exact_address": 95, "name_postal": 90, "addr_postal": 85,
    "postal": 70, "name_token": 65, "address_token": 60, "first2": 40,
}
# Multi-signal bonus: a candidate found by more than one independent block is much
# more likely to be a true match than one found by a single weak block, so it should
# outrank a single strong-looking hit when the per-S1 cap forces a cutoff.
MULTI_SIGNAL_BONUS = 20

FEATURE_COLS = [
    "name_lev", "name_token_sort", "name_token_set", "name_jaccard", "name_overlap",
    "name_exact", "name_first_token", "name_char3_jaccard", "name_char4_jaccard",
    "addr_lev", "addr_token_sort", "addr_token_set", "addr_jaccard", "addr_overlap",
    "addr_exact", "postal_match", "house_overlap", "same_country", "name_addr_product",
    "name_addr_mean", "name_addr_gap", "same_name_first2", "same_address_first2",
    "block_exact_name", "block_exact_address", "block_postal", "block_name_token",
    "block_address_token", "block_first2", "block_name_postal", "block_addr_postal",
]

STOPWORDS = {
    "private", "limited", "company", "corporation", "incorporated", "and", "the", "of",
    "pvt", "ltd", "co", "inc",
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs(artifact_dir: Path, output_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in ["train", "test", "validation", "models"]:
        (artifact_dir / p).mkdir(parents=True, exist_ok=True)


def connect_db(artifact_dir: Path) -> duckdb.DuckDBPyConnection:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tmp = artifact_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(artifact_dir / "entity_resolution.duckdb"))
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute(f"PRAGMA temp_directory='{qpath(tmp)}'")
    try:
        con.execute("PRAGMA max_temp_directory_size='100GB'")
    except Exception:
        pass
    return con

# ---------------------------------------------------------------------------
# Normalization expressions - executed by DuckDB, not Python row-by-row.
# ---------------------------------------------------------------------------

def name_expr(col: str) -> str:
    x = f"lower(coalesce({col}, ''))"
    x = f"regexp_replace({x}, '[^a-z0-9]+', ' ', 'g')"
    for a, b in [("pvt", "private"), ("ltd", "limited"), ("corp", "corporation"),
                 ("co", "company"), ("inc", "incorporated")]:
        x = f"regexp_replace({x}, '\\\\b{a}\\\\b', '{b}', 'g')"
    return f"trim(regexp_replace({x}, '\\\\s+', ' ', 'g'))"


def address_expr(col: str) -> str:
    x = f"lower(coalesce({col}, ''))"
    x = f"regexp_replace({x}, '[^a-z0-9]+', ' ', 'g')"
    for a, b in [("rd", "road"), ("st", "street"), ("ave", "avenue"),
                 ("blvd", "boulevard"), ("apt", "apartment"), ("fl", "floor"),
                 ("sec", "sector"), ("nr", "near")]:
        x = f"regexp_replace({x}, '\\\\b{a}\\\\b', '{b}', 'g')"
    return f"trim(regexp_replace({x}, '\\\\s+', ' ', 'g'))"


def postal_expr(col: str) -> str:
    return f"coalesce(regexp_extract(coalesce({col}, ''), '([0-9]{{5,6}})', 1), '')"


def first_token(col: str) -> str:
    return f"split_part({col}, ' ', 1)"


def first2(col: str) -> str:
    return f"array_to_string(list_slice(string_split({col}, ' '), 1, 2), ' ')"


def normalize_table(con: duckdb.DuckDBPyConnection, table: str, path: Path) -> None:
    con.execute(f"DROP TABLE IF EXISTS {table}")
    n = name_expr("business_name")
    a = address_expr("business_address")
    p = postal_expr("business_address")
    con.execute(f"""
        CREATE TABLE {table} AS
        SELECT
            entity_id,
            coalesce(business_name, '') AS business_name,
            coalesce(business_address, '') AS business_address,
            coalesce(country, '') AS country,
            {n} AS name_norm,
            regexp_replace({n}, '[^a-z0-9]', '', 'g') AS name_alnum,
            {a} AS addr_norm,
            {p} AS postal,
            {first_token(n)} AS name_first,
            {first2(n)} AS name_first2,
            {first_token(a)} AS addr_first,
            {first2(a)} AS addr_first2
        FROM read_csv('{qpath(path)}', delim='\\t', header=true, all_varchar=true, nullstr='')
    """)
    con.execute(f"CREATE INDEX IF NOT EXISTS {table}_id_idx ON {table}(entity_id)")
    con.execute(f"CREATE INDEX IF NOT EXISTS {table}_name_idx ON {table}(name_norm)")
    con.execute(f"CREATE INDEX IF NOT EXISTS {table}_addr_idx ON {table}(addr_norm)")
    con.execute(f"CREATE INDEX IF NOT EXISTS {table}_postal_idx ON {table}(postal)")


def token_table(con: duckdb.DuckDBPyConnection, source: str, field: str, table: str) -> None:
    con.execute(f"DROP TABLE IF EXISTS {table}")
    stop = ",".join("'" + x.replace("'", "''") + "'" for x in STOPWORDS)
    con.execute(f"""
        CREATE TABLE {table} AS
        SELECT entity_id, token
        FROM {source}, UNNEST(string_split({field}, ' ')) AS u(token)
        WHERE length(token) >= {MIN_TOKEN_LEN}
          AND token <> ''
          AND token NOT IN ({stop})
    """)
    con.execute(f"CREATE INDEX IF NOT EXISTS {table}_token_idx ON {table}(token)")
    con.execute(f"CREATE INDEX IF NOT EXISTS {table}_id_idx ON {table}(entity_id)")


def _table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return bool(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?", [name]).fetchone()[0])

# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def load_gt(path: Path) -> Dict[str, set]:
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {s: {x for x in ids.split(",") if x} for s, ids in zip(gt.source1_entity_id, gt.matched_entity_ids)}

def load_gt_subset(path: Path, ids: set[str]) -> Dict[str, set]:
    if not ids:
        return {}
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=["source1_entity_id", "matched_entity_ids"])
    df = df[df.source1_entity_id.isin(ids)]
    return {s: {x for x in vals.split(",") if x} for s, vals in zip(df.source1_entity_id, df.matched_entity_ids)}


def audit(data_dir: Path, artifact_dir: Path) -> None:
    con = connect_db(artifact_dir)
    files = {
        "train_s1": data_dir / "train/train_source1.tsv",
        "train_s2": data_dir / "train/train_source2.tsv",
        "train_s3": data_dir / "train/train_source3.tsv",
        "test_s1": data_dir / "test/test_source1.tsv",
        "test_s2": data_dir / "test/test_source2.tsv",
        "test_s3": data_dir / "test/test_source3.tsv",
    }
    summary = {}
    for key, path in files.items():
        r = con.execute(f"""
            SELECT count(*), count(DISTINCT entity_id),
                   count(*) FILTER (WHERE coalesce(country,'')=''), count(DISTINCT country)
            FROM read_csv('{qpath(path)}', delim='\\t', header=true, all_varchar=true, nullstr='')
        """).fetchone()
        summary[key] = {"rows": int(r[0]), "unique_ids": int(r[1]), "blank_country": int(r[2]), "countries": int(r[3])}
        log(f"{key}: {summary[key]}")
    gt = load_gt(data_dir / "train/train_ground_truth.tsv")
    cards = pd.Series([len(v) for v in gt.values()])
    g = {
        "s1_rows": len(gt), "singletons": int((cards == 0).sum()),
        "non_singletons": int((cards > 0).sum()), "total_true_links": int(cards.sum()),
        "max_links_per_s1": int(cards.max()), "duplicate_s1": len(gt) - len(set(gt)),
        "unique_matched_ids": len({x for v in gt.values() for x in v}),
        "cardinality_distribution": {str(int(k)): int(v) for k, v in cards.value_counts().sort_index().items()},
    }
    log(f"ground truth: {g}")
    write_json(artifact_dir / "validation/audit.json", {"sources": summary, "ground_truth": g})
    con.close()

# ---------------------------------------------------------------------------
# Train preparation: TRAIN ONLY. No test files are touched.
# ---------------------------------------------------------------------------

def select_workset(con: duckdb.DuckDBPyConnection, artifact_dir: Path, total_limit: int, val_limit: int) -> None:
    """Create deterministic bounded train/validation S1 worksets using DuckDB hash.
    This avoids pulling 2.2M IDs into Python memory."""
    con.execute("DROP TABLE IF EXISTS work_s1")
    con.execute(f"""
        CREATE TABLE work_s1 AS
        SELECT entity_id, CASE WHEN abs(hash(entity_id)) % 100 < 20 THEN 'val' ELSE 'train' END AS split
        FROM train_s1
        WHERE abs(hash(entity_id)) % 100 < 20
           OR abs(hash(entity_id)) % 100 >= 20
        QUALIFY row_number() OVER (
            PARTITION BY CASE WHEN abs(hash(entity_id)) % 100 < 20 THEN 'val' ELSE 'train' END
            ORDER BY hash(entity_id)
        ) <= CASE WHEN abs(hash(entity_id)) % 100 < 20 THEN {val_limit} ELSE {total_limit - val_limit} END
    """)
    con.execute("CREATE INDEX IF NOT EXISTS work_s1_id_idx ON work_s1(entity_id)")
    counts = con.execute("SELECT split, count(*) FROM work_s1 GROUP BY split ORDER BY split").fetchall()
    log(f"bounded CV workset: {counts}")
    write_json(artifact_dir / "validation/workset.json", {"train_limit": total_limit-val_limit, "val_limit": val_limit, "counts": {str(a): int(b) for a,b in counts}})


def prepare_train(data_dir: Path, artifact_dir: Path, work_s1_limit: int, val_s1_limit: int) -> None:
    con = connect_db(artifact_dir)
    train_dir = artifact_dir / "train"
    if not all(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?", [t]).fetchone()[0] for t in ["train_s1","train_s2","train_s3"]):
        for table, rel in [("train_s1", "train/train_source1.tsv"), ("train_s2", "train/train_source2.tsv"), ("train_s3", "train/train_source3.tsv")]:
            log(f"normalizing {table}")
            normalize_table(con, table, data_dir / rel)
    else:
        log("train normalized tables already exist; reusing")
    select_workset(con, artifact_dir, work_s1_limit, val_s1_limit)
    # Only the bounded S1 workset is tokenized on the left. Target sources are fully indexed.
    if not con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?", ["work_s1_name_tokens"]).fetchone()[0]:
        log("building work_s1_name_tokens")
        token_table(con, "(SELECT s.* FROM train_s1 s JOIN work_s1 w ON w.entity_id=s.entity_id)", "name_norm", "work_s1_name_tokens")
    if not con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?", ["work_s1_addr_tokens"]).fetchone()[0]:
        log("building work_s1_addr_tokens")
        token_table(con, "(SELECT s.* FROM train_s1 s JOIN work_s1 w ON w.entity_id=s.entity_id)", "addr_norm", "work_s1_addr_tokens")
    for table in ["train_s2", "train_s3"]:
        tok_name = f"{table}_name_tokens"
        tok_addr = f"{table}_addr_tokens"
        if not con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?", [tok_name]).fetchone()[0]:
            log(f"building {tok_name}")
            token_table(con, table, "name_norm", tok_name)
        if not con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?", [tok_addr]).fetchone()[0]:
            log(f"building {tok_addr}")
            token_table(con, table, "addr_norm", tok_addr)
    write_json(train_dir / "prepared.json", {"status": "ready", "data_dir": str(data_dir.resolve())})
    con.close()
    log("prepare-train complete; test data was not read")

# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------

def _create_empty_candidate_table(con: duckdb.DuckDBPyConnection, name: str) -> None:
    con.execute(f"DROP TABLE IF EXISTS {name}")
    flag_cols = ",\n               ".join(f"CAST(0 AS INTEGER) {f}" for f in ALL_FLAGS)
    con.execute(f"""
        CREATE TABLE {name} AS
        SELECT CAST(NULL AS VARCHAR) s1_id, CAST(NULL AS VARCHAR) other_id,
               CAST(NULL AS VARCHAR) AS "source",
               {flag_cols}
        WHERE FALSE
    """)


def _insert_block_hits(con: duckdb.DuckDBPyConnection, final: str, label: str, flag: str) -> None:
    """Append the rows currently in temp table _block_pairs into the raw candidate
    table, stamping every flag column with 0 except the one that found this block."""
    values = ",".join("1" if f == flag else "0" for f in ALL_FLAGS)
    con.execute(f"""
        INSERT INTO {final} (s1_id,other_id,"source",{",".join(ALL_FLAGS)})
        SELECT s1_id,other_id,'{label}',{values}
        FROM _block_pairs
    """)


def _append_exact_block(con: duckdb.DuckDBPyConnection, other: str, label: str,
                        final: str, flag: str,
                        cap: int = MAX_CANDIDATES_PER_BLOCK) -> None:
    """Run one bounded exact/composite-key block and append it to a small raw candidate
    table. Each block caps its own posting lists AND its own per-S1 result count
    immediately, so DuckDB never has to hold a large intermediate relation - this is
    the key memory fix from the original OOM. `flag` selects columns + posting cap
    from KEY_MAP (e.g. a single column for exact_name/postal, or a pair of columns
    concatenated for the composite name_postal/addr_postal blocks)."""
    cols, key_cap = KEY_MAP[flag]
    key_bare = " || '|' || ".join(cols)                       # against {other} directly
    key_o = " || '|' || ".join(f"o.{c}" for c in cols)         # against the join alias
    condition = " AND ".join([f"s.{c} <> ''" for c in cols] + [f"o.{c}=s.{c}" for c in cols] + ["o.country=s.country"])
    con.execute("DROP TABLE IF EXISTS _block_pairs")
    con.execute(f"""
        CREATE TEMP TABLE _block_pairs AS
        WITH allowed_keys AS (
            SELECT {key_bare} AS k
            FROM {other}
            GROUP BY {key_bare}
            HAVING count(*) <= {key_cap}
        )
        SELECT s.entity_id AS s1_id, o.entity_id AS other_id
        FROM work_s1 w
        JOIN train_s1 s ON s.entity_id=w.entity_id
        JOIN {other} o ON {condition}
        JOIN allowed_keys ak ON ak.k = {key_o}
        QUALIFY row_number() OVER (PARTITION BY s.entity_id ORDER BY o.entity_id) <= {cap}
    """)
    n = int(con.execute("SELECT count(*) FROM _block_pairs").fetchone()[0])
    log(f"  {label} [{flag}]: {n:,} bounded pairs")
    if n:
        _insert_block_hits(con, final, label, flag)


def _append_token_block(con: duckdb.DuckDBPyConnection, other: str, label: str,
                        final: str, token_type: str,
                        cap: int = MAX_CANDIDATES_PER_BLOCK) -> None:
    """Rare-token blocking: only tokens with a posting list <= MAX_TOKEN_POSTINGS on
    the target side are ever joined on, so common tokens (limited, company, mumbai...)
    never create a large join - this is the recall workhorse without the OOM risk.
    A single shared rare token is weak evidence on its own for names (lots of
    single-word overlaps are coincidental), so name blocking requires
    MIN_SHARED_NAME_TOKENS; addresses have fewer usable tokens so 1 is kept."""
    tok = 'name' if token_type == 'name' else 'addr'
    flag = 'name_token' if token_type == 'name' else 'address_token'
    min_shared = MIN_SHARED_NAME_TOKENS if token_type == 'name' else MIN_SHARED_ADDR_TOKENS
    left_tokens = f"work_s1_{tok}_tokens"
    right_tokens = f"{other}_{tok}_tokens"
    con.execute("DROP TABLE IF EXISTS _block_pairs")
    con.execute(f"""
        CREATE TEMP TABLE _block_pairs AS
        WITH allowed AS (
            SELECT token
            FROM {right_tokens}
            GROUP BY token
            HAVING count(*) <= {MAX_TOKEN_POSTINGS}
        ), hits AS (
            SELECT w.entity_id AS s1_id, ot.entity_id AS other_id,
                   count(*) AS shared_tokens
            FROM work_s1 w
            JOIN train_s1 s ON s.entity_id=w.entity_id
            JOIN {left_tokens} lt ON lt.entity_id=s.entity_id
            JOIN allowed a ON a.token=lt.token
            JOIN {right_tokens} ot ON ot.token=lt.token
            JOIN {other} o ON o.entity_id=ot.entity_id AND o.country=s.country
            GROUP BY w.entity_id, ot.entity_id
            HAVING count(*) >= {min_shared}
        )
        SELECT s1_id, other_id
        FROM hits
        QUALIFY row_number() OVER
            (PARTITION BY s1_id ORDER BY shared_tokens DESC, other_id) <= {cap}
    """)
    n = int(con.execute("SELECT count(*) FROM _block_pairs").fetchone()[0])
    log(f"  {label} [{flag}]: {n:,} bounded pairs")
    if n:
        _insert_block_hits(con, final, label, flag)


def block_pair(con: duckdb.DuckDBPyConnection, other: str, label: str, final: str,
               use_extended_blocks: bool = True) -> None:
    _create_empty_candidate_table(con, f"_raw_{label.lower()}")
    raw = f"_raw_{label.lower()}"
    log(f"blocking {label}: exact + composite blocks")
    for flag in ["exact_name", "exact_address", "postal", "first2", "name_postal", "addr_postal"]:
        _append_exact_block(con, other, label, raw, flag)

    if use_extended_blocks:
        log(f"blocking {label}: rare-token blocks")
        _append_token_block(con, other, label, raw, 'name')
        _append_token_block(con, other, label, raw, 'addr')
    else:
        log(f"blocking {label}: rare-token blocks skipped (--no-extended-blocks)")

    agg_cols = ",\n               ".join(f"max({f}) {f}" for f in ALL_FLAGS)
    con.execute(f"DROP TABLE IF EXISTS {final}")
    con.execute(f"""
        CREATE TABLE {final} AS
        SELECT s1_id, other_id, '{label}' AS "source",
               {agg_cols}
        FROM {raw}
        GROUP BY s1_id, other_id
    """)
    # Rank by cumulative block strength plus a bonus for candidates confirmed by more
    # than one independent block (not an arbitrary ORDER BY other_id), so when an S1
    # has more hits than the cap, the strongest, best-corroborated candidates survive.
    strength_expr = " + ".join(f"{BLOCK_STRENGTH[f]}*{f}" for f in ALL_FLAGS)
    signal_count_expr = " + ".join(ALL_FLAGS)
    con.execute(f"""
        CREATE OR REPLACE TABLE {final}_ranked AS
        SELECT s1_id, other_id, "source", {",".join(ALL_FLAGS)}
        FROM (
            SELECT *, ({strength_expr}) + {MULTI_SIGNAL_BONUS}*GREATEST(({signal_count_expr}) - 1, 0) AS strength
            FROM {final}
        )
        QUALIFY row_number() OVER (PARTITION BY s1_id ORDER BY strength DESC, other_id) <= {MAX_CANDIDATES_PER_S1}
    """)
    con.execute(f"DROP TABLE {final}")
    con.execute(f"ALTER TABLE {final}_ranked RENAME TO {final}")
    con.execute(f"DROP TABLE {raw}")
    con.execute("DROP TABLE IF EXISTS _block_pairs")
    n = int(con.execute(f"SELECT count(*) FROM {final}").fetchone()[0])
    log(f"  {final}: {n:,} final candidates")

def block_train(artifact_dir: Path, use_extended_blocks: bool = True) -> None:
    con = connect_db(artifact_dir)
    for other, label in [("train_s2", "S2"), ("train_s3", "S3")]:
        log(f"blocking workset -> {other}")
        block_pair(con, other, label, f"cand_{label.lower()}", use_extended_blocks=use_extended_blocks)
    con.execute("DROP TABLE IF EXISTS train_candidates")
    con.execute("CREATE TABLE train_candidates AS SELECT * FROM cand_s2 UNION ALL SELECT * FROM cand_s3")
    stats = con.execute("SELECT count(*), count(DISTINCT s1_id) FROM train_candidates").fetchone()
    log(f"train candidates: {int(stats[0]):,} pairs across {int(stats[1]):,} S1")
    con.close()

# ---------------------------------------------------------------------------
# Candidate recall / features / model
# ---------------------------------------------------------------------------

def candidate_recall(con: duckdb.DuckDBPyConnection, gt_path: Path, split: str) -> Dict[str, float]:
    """Compute recall in DuckDB, broken down by source (S2/S3) plus overall, without
    materializing all 2.2M ground-truth rows in Python. A true link is attributed to
    S2 or S3 by which source table its other_id actually belongs to."""
    con.execute("DROP TABLE IF EXISTS _truth")
    con.execute(f"""CREATE TEMP TABLE _truth AS
        SELECT gt.source1_entity_id AS s1_id, trim(x.id) AS other_id,
               CASE WHEN s2.entity_id IS NOT NULL THEN 'S2'
                    WHEN s3.entity_id IS NOT NULL THEN 'S3'
                    ELSE 'UNK' END AS src
        FROM read_csv('{qpath(gt_path)}', delim='\\t', header=true, all_varchar=true, nullstr='') gt
        JOIN work_s1 w ON w.entity_id=gt.source1_entity_id AND w.split='{split}'
        CROSS JOIN UNNEST(string_split(gt.matched_entity_ids, ',')) AS x(id)
        LEFT JOIN train_s2 s2 ON s2.entity_id=trim(x.id)
        LEFT JOIN train_s3 s3 ON s3.entity_id=trim(x.id)
        WHERE trim(x.id) <> ''
    """)
    results: Dict[str, float] = {}
    for src in ("S2", "S3"):
        total = int(con.execute(f"SELECT count(*) FROM _truth WHERE src='{src}'").fetchone()[0])
        hit = int(con.execute(f"""
            SELECT count(*) FROM _truth t
            JOIN train_candidates c ON c.s1_id=t.s1_id AND c.other_id=t.other_id
            WHERE t.src='{src}'
        """).fetchone()[0])
        results[src] = (hit / total) if total else 0.0
    total_all = int(con.execute("SELECT count(*) FROM _truth").fetchone()[0])
    hit_all = int(con.execute("SELECT count(*) FROM _truth t JOIN train_candidates c ON c.s1_id=t.s1_id AND c.other_id=t.other_id").fetchone()[0])
    results["overall"] = (hit_all / total_all) if total_all else 0.0
    return results

def token_set(s: str) -> set:
    return set(s.split()) if s else set()


def char_ngrams(s: str,n:int)->set:
    if not s:return set()
    return {s} if len(s)<n else {s[i:i+n] for i in range(len(s)-n+1)}


def jac(a:set,b:set)->float:
    if not a and not b:return 1.0
    if not a or not b:return 0.0
    return len(a&b)/len(a|b)


def ov(a:set,b:set)->float:
    if not a or not b:return 0.0
    return len(a&b)/min(len(a),len(b))


def pair_features(df: pd.DataFrame) -> pd.DataFrame:
    out=[]
    for r in df.itertuples(index=False):
        n1,n2=r.name_norm,r.other_name_norm; a1,a2=r.addr_norm,r.other_addr_norm
        nt1,nt2=token_set(n1),token_set(n2); at1,at2=token_set(a1),token_set(a2)
        nl=fuzz.ratio(n1,n2)/100; al=fuzz.ratio(a1,a2)/100
        ns=fuzz.token_sort_ratio(n1,n2)/100; nset=fuzz.token_set_ratio(n1,n2)/100
        ats=fuzz.token_sort_ratio(a1,a2)/100; aset=fuzz.token_set_ratio(a1,a2)/100
        out.append({
            "s1_id":r.s1_id,"other_id":r.other_id,"source":r.source,
            "name_lev":nl,"name_token_sort":ns,"name_token_set":nset,"name_jaccard":jac(nt1,nt2),"name_overlap":ov(nt1,nt2),
            "name_exact":float(bool(n1) and n1==n2),"name_first_token":float(bool(n1) and n1.split()[0]==n2.split()[0]),
            "name_char3_jaccard":jac(char_ngrams(n1,3),char_ngrams(n2,3)),"name_char4_jaccard":jac(char_ngrams(n1,4),char_ngrams(n2,4)),
            "addr_lev":al,"addr_token_sort":ats,"addr_token_set":aset,"addr_jaccard":jac(at1,at2),"addr_overlap":ov(at1,at2),
            "addr_exact":float(bool(a1) and a1==a2),"postal_match":float(bool(r.postal) and r.postal==r.other_postal),
            "house_overlap":jac({x for x in a1.split() if x.isdigit()},{x for x in a2.split() if x.isdigit()}),
            "same_country":float(r.country==r.other_country),"name_addr_product":nl*al,"name_addr_mean":(nl+al)/2,"name_addr_gap":abs(nl-al),
            "same_name_first2":float(bool(r.name_first2) and r.name_first2==r.other_name_first2),
            "same_address_first2":float(bool(r.addr_first2) and r.addr_first2==r.other_addr_first2),
            "block_exact_name":float(r.exact_name),"block_exact_address":float(r.exact_address),"block_postal":float(r.postal),
            "block_name_token":float(r.name_token),"block_address_token":float(r.address_token),"block_first2":float(r.first2),
            "block_name_postal":float(getattr(r,'name_postal',0)),"block_addr_postal":float(getattr(r,'addr_postal',0)),
        })
    return pd.DataFrame(out)


def build_features(con: duckdb.DuckDBPyConnection, candidate_table:str, out_path:Path, split: str|None=None) -> int:
    if out_path.exists(): out_path.unlink()
    where = f"AND w.split=\'{split}\'" if split else ""
    con.execute("DROP TABLE IF EXISTS _feature_work")
    con.execute(f"""CREATE TEMP TABLE _feature_work AS
        SELECT c.*, row_number() OVER (ORDER BY c.s1_id,c.other_id) rn
        FROM {candidate_table} c JOIN work_s1 w ON w.entity_id=c.s1_id WHERE 1=1 {where}
    """)
    count=int(con.execute("SELECT count(*) FROM _feature_work").fetchone()[0])
    total=0; first=True
    for offset in range(0,count,FEATURE_BATCH_SIZE):
        q=f"""
        WITH b AS (SELECT * FROM _feature_work WHERE rn>{offset} AND rn<={offset+FEATURE_BATCH_SIZE})
        SELECT b.*, s.name_norm,s.addr_norm,s.postal,s.country,s.name_first2,s.addr_first2,
               CASE WHEN b."source"=\'S2\' THEN o2.name_norm ELSE o3.name_norm END other_name_norm,
               CASE WHEN b."source"=\'S2\' THEN o2.addr_norm ELSE o3.addr_norm END other_addr_norm,
               CASE WHEN b."source"=\'S2\' THEN o2.postal ELSE o3.postal END other_postal,
               CASE WHEN b."source"=\'S2\' THEN o2.country ELSE o3.country END other_country,
               CASE WHEN b."source"=\'S2\' THEN o2.name_first2 ELSE o3.name_first2 END other_name_first2,
               CASE WHEN b."source"=\'S2\' THEN o2.addr_first2 ELSE o3.addr_first2 END other_addr_first2
        FROM b JOIN train_s1 s ON s.entity_id=b.s1_id
        LEFT JOIN train_s2 o2 ON b."source"=\'S2\' AND o2.entity_id=b.other_id
        LEFT JOIN train_s3 o3 ON b."source"=\'S3\' AND o3.entity_id=b.other_id
        """
        df=con.execute(q).fetchdf()
        if df.empty: continue
        feat=pair_features(df)
        feat.to_csv(out_path,sep="	",index=False,mode="a",header=first)
        first=False; total+=len(feat)
        if total % 100000 < FEATURE_BATCH_SIZE: log(f"features {total:,}/{count:,}")
    return total

def sample_train(path:Path, gt:Dict[str,set], train_ids:set)->pd.DataFrame:
    pos=[]; neg=[]
    for ch in pd.read_csv(path,sep='\t',chunksize=50_000):
        ch=ch[ch.s1_id.isin(train_ids)]
        if ch.empty:continue
        ch['label']=[int(o in gt.get(s,set())) for s,o in zip(ch.s1_id,ch.other_id)]
        p=ch[ch.label==1]
        if not p.empty:pos.append(p)
        n=ch[ch.label==0].copy()
        if not n.empty:
            n['hard']=n.name_exact*100+n.addr_exact*90+n.postal_match*80+n.name_token_sort*30+n.addr_token_sort*20
            neg.append(n.sort_values(['s1_id','hard'],ascending=[True,False]).groupby('s1_id',sort=False).head(NEGATIVES_PER_S1).drop(columns='hard'))
    p=pd.concat(pos,ignore_index=True) if pos else pd.DataFrame(); n=pd.concat(neg,ignore_index=True) if neg else pd.DataFrame()
    if p.empty and n.empty:return pd.DataFrame()
    x=pd.concat([p,n],ignore_index=True).drop_duplicates(['s1_id','other_id'])
    if len(x)>MAX_TRAIN_ROWS:
        rng=np.random.default_rng(SEED); x=x.iloc[rng.choice(len(x),MAX_TRAIN_ROWS,replace=False)]
    return x


def train_model(df:pd.DataFrame)->XGBClassifier:
    y=df.label.astype(np.int8); pos=max(int(y.sum()),1); neg=max(int((y==0).sum()),1)
    model=XGBClassifier(n_estimators=300,max_depth=7,learning_rate=.05,min_child_weight=3,subsample=.85,colsample_bytree=.9,
                        reg_lambda=2,objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_jobs=XGB_N_JOBS,
                        scale_pos_weight=neg/pos,random_state=SEED)
    model.fit(df[FEATURE_COLS].astype(np.float32),y); return model


def macro_f05(true:Dict[str,set],pred:Dict[str,set])->float:
    scores=[]
    for s,t in true.items():
        p=pred.get(s,set())
        if not t:scores.append(1.0 if not p else 0.0);continue
        if not p:scores.append(0.0);continue
        tp=len(t&p); precision=tp/len(p); recall=tp/len(t)
        scores.append(0.0 if tp==0 else 1.25*precision*recall/(.25*precision+recall))
    return float(np.mean(scores)) if scores else 0.0


def evaluate(model:XGBClassifier, feat_path:Path, gt:Dict[str,set], split:str)->Tuple[float,float,pd.DataFrame]:
    pred_parts=[]
    # Read only rows whose S1 belongs to the validation workset. The feature file is already bounded.
    ids={s for s in gt if False}
    # derive IDs from the artifact file in the caller instead; every row in feat_path is labelled by split
    for ch in pd.read_csv(feat_path,sep='\t',chunksize=FEATURE_BATCH_SIZE):
        if ch.empty:continue
        ch['proba']=model.predict_proba(ch[FEATURE_COLS].astype(np.float32))[:,1]
        pred_parts.append(ch[['s1_id','other_id','proba']])
    pred=pd.concat(pred_parts,ignore_index=True) if pred_parts else pd.DataFrame(columns=['s1_id','other_id','proba'])
    # true dictionary is supplied already restricted to validation IDs.
    rows=[]; best=(.5,-1)
    for t in THRESHOLDS:
        hit=pred[pred.proba>=t]
        pdict={s:set(g.other_id) for s,g in hit.groupby('s1_id')}
        score=macro_f05(gt,pdict); rows.append({'threshold':float(t),'macro_f0.5':score})
        if score>best[1]:best=(float(t),float(score))
    return best[0],best[1],pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Validation / training stage
# ---------------------------------------------------------------------------

def train_and_validate(data_dir:Path,artifact_dir:Path,use_extended_blocks:bool=True)->None:
    con=connect_db(artifact_dir)
    train_ids=set(r[0] for r in con.execute("SELECT entity_id FROM work_s1 WHERE split='train'").fetchall())
    val_ids=set(r[0] for r in con.execute("SELECT entity_id FROM work_s1 WHERE split='val'").fetchall())
    con.close()
    gt_path=data_dir/'train/train_ground_truth.tsv'
    gt_train=load_gt_subset(gt_path,train_ids)
    gt_val=load_gt_subset(gt_path,val_ids)

    block_train(artifact_dir, use_extended_blocks=use_extended_blocks)
    con=connect_db(artifact_dir)
    recall=candidate_recall(con,gt_path,'val')
    log(f"VALIDATION candidate recall: S2={recall['S2']:.4%} S3={recall['S3']:.4%} overall={recall['overall']:.4%}")
    train_feat=artifact_dir/'train'/'train_features.tsv'
    val_feat=artifact_dir/'train'/'validation_features.tsv'
    model_path=artifact_dir/'models/matcher.json'
    threshold_path=artifact_dir/'validation'/'best_threshold.json'
    thresholds_path=artifact_dir/'validation'/'thresholds.tsv'
    # block_train() just regenerated candidates, so any cached features/model/threshold
    # from a previous blocking config are stale and would silently invalidate this
    # experiment (new candidates, old features) - wipe them every run.
    for stale in (train_feat, val_feat, model_path, threshold_path, thresholds_path):
        stale.unlink(missing_ok=True)
    log('building TRAIN-only candidate features')
    build_features(con,'train_candidates',train_feat,split='train')
    log('building VALIDATION-only candidate features')
    build_features(con,'train_candidates',val_feat,split='val')
    con.close()

    sample=sample_train(train_feat,gt_train,train_ids)
    if sample.empty:raise RuntimeError('No training examples survived blocking. Increase --work-s1 or relax blocking.')
    log(f"training rows={len(sample):,}; positives={int(sample.label.sum()):,}; negatives={int((sample.label==0).sum()):,}")
    model=train_model(sample); model.save_model(str(model_path))

    pred_parts=[]
    for ch in pd.read_csv(val_feat,sep='\t',chunksize=FEATURE_BATCH_SIZE):
        ch['proba']=model.predict_proba(ch[FEATURE_COLS].astype(np.float32))[:,1]
        pred_parts.append(ch[['s1_id','other_id','proba']])
    pred=pd.concat(pred_parts,ignore_index=True) if pred_parts else pd.DataFrame(columns=['s1_id','other_id','proba'])
    rows=[];best_t=.5;best_score=-1.0
    for threshold in THRESHOLDS:
        hit=pred[pred.proba>=threshold]
        pdict={s:set(g.other_id) for s,g in hit.groupby('s1_id')}
        score=macro_f05(gt_val,pdict);rows.append({'threshold':float(threshold),'macro_f0.5':score})
        if score>best_score:best_t=float(threshold);best_score=float(score)
    pd.DataFrame(rows).to_csv(thresholds_path,sep='\t',index=False)
    write_json(threshold_path,{'threshold':best_t,'macro_f0.5':best_score,'candidate_recall':recall,
        'workset_train_s1':len(train_ids),'workset_val_s1':len(val_ids),'seed':SEED,'extended_blocks':use_extended_blocks,
        'blocking_version':BLOCKING_VERSION})
    log(f"VALIDATION: candidate_recall(overall)={recall['overall']:.6f} best_threshold={best_t:.2f} macro_F0.5={best_score:.6f}")


# ---------------------------------------------------------------------------
# Test preparation / inference - explicit and separate.
# ---------------------------------------------------------------------------

def prepare_test(data_dir:Path,artifact_dir:Path)->None:
    con=connect_db(artifact_dir)
    for table,rel in [('test_s1','test/test_source1.tsv'),('test_s2','test/test_source2.tsv'),('test_s3','test/test_source3.tsv')]:
        exists=con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?",[table]).fetchone()[0]
        if exists:log(f"reusing {table}");continue
        log(f"normalizing {table}");normalize_table(con,table,data_dir/rel)
    for table in ['test_s1','test_s2','test_s3']:
        for fld,suf in [('name_norm','name_tokens'),('addr_norm','addr_tokens')]:
            t=f'{table}_{suf}'
            if not con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?",[t]).fetchone()[0]:
                log(f"building {t}");token_table(con,table,fld,t)
    write_json(artifact_dir/'test'/'prepared.json',{'status':'ready','data_dir':str(data_dir.resolve())})
    con.close();log('prepare-test complete')


def block_test(artifact_dir:Path)->None:
    con=connect_db(artifact_dir)
    for other,label in [('test_s2','S2'),('test_s3','S3')]:
        raw=f'raw_test_{label.lower()}'; final=f'test_cand_{label.lower()}'
        con.execute(f"DROP TABLE IF EXISTS {raw}")
        # Test blocking uses the same multi-pass logic, but no training ground truth.
        con.execute(f"""
        CREATE TABLE {raw} AS
        WITH name_freq AS (SELECT token FROM {other}_name_tokens GROUP BY token HAVING count(*) <= (SELECT count(*)*{MAX_TOKEN_DOC_FREQ} FROM {other})),
        addr_freq AS (SELECT token FROM {other}_addr_tokens GROUP BY token HAVING count(*) <= (SELECT count(*)*{MAX_TOKEN_DOC_FREQ} FROM {other})),
        exact_name AS (SELECT s.entity_id s1_id,o.entity_id other_id,'exact_name' block FROM test_s1 s JOIN {other} o ON s.name_norm<>'' AND s.name_norm=o.name_norm AND s.country=o.country),
        exact_addr AS (SELECT s.entity_id s1_id,o.entity_id other_id,'exact_address' block FROM test_s1 s JOIN {other} o ON s.addr_norm<>'' AND s.addr_norm=o.addr_norm AND s.country=o.country),
        postal AS (SELECT s.entity_id s1_id,o.entity_id other_id,'postal' block FROM test_s1 s JOIN {other} o ON s.postal<>'' AND s.postal=o.postal AND s.country=o.country),
        first2 AS (SELECT s.entity_id s1_id,o.entity_id other_id,'first2' block FROM test_s1 s JOIN {other} o ON s.name_first2<>'' AND s.name_first2=o.name_first2 AND s.country=o.country),
        nt AS (SELECT DISTINCT s.entity_id s1_id,ot.entity_id other_id,'name_token' block FROM test_s1 s JOIN test_s1_name_tokens st ON st.entity_id=s.entity_id JOIN name_freq f ON f.token=st.token JOIN {other}_name_tokens ot ON ot.token=st.token JOIN {other} o ON o.entity_id=ot.entity_id AND o.country=s.country),
        addr_tok AS (SELECT DISTINCT s.entity_id s1_id,ot.entity_id other_id,'address_token' block FROM test_s1 s JOIN test_s1_addr_tokens st ON st.entity_id=s.entity_id JOIN addr_freq f ON f.token=st.token JOIN {other}_addr_tokens ot ON ot.token=st.token JOIN {other} o ON o.entity_id=ot.entity_id AND o.country=s.country)
        SELECT * FROM exact_name UNION ALL SELECT * FROM exact_addr UNION ALL SELECT * FROM postal UNION ALL SELECT * FROM first2 UNION ALL SELECT * FROM nt UNION ALL SELECT * FROM addr_tok
        """)
        con.execute(f"DROP TABLE IF EXISTS {final}")
        con.execute(f"""
        CREATE TABLE {final} AS SELECT s1_id,other_id,'{label}' AS "source",exact_name,exact_address,postal,name_token,address_token,first2
        FROM (SELECT s1_id,other_id,max(block='exact_name')::INT exact_name,max(block='exact_address')::INT exact_address,max(block='postal')::INT postal,
                     max(block='name_token')::INT name_token,max(block='address_token')::INT address_token,max(block='first2')::INT first2
              FROM {raw} GROUP BY s1_id,other_id) x
        QUALIFY row_number() OVER(PARTITION BY s1_id ORDER BY exact_name DESC,exact_address DESC,postal DESC,name_token DESC,address_token DESC,first2 DESC,other_id)<={MAX_CANDIDATES_PER_S1}
        """)
        con.execute(f"DROP TABLE {raw}")
    con.execute("DROP TABLE IF EXISTS test_candidates")
    con.execute("CREATE TABLE test_candidates AS SELECT * FROM test_cand_s2 UNION ALL SELECT * FROM test_cand_s3")
    r=con.execute("SELECT count(*),count(DISTINCT s1_id) FROM test_candidates").fetchone();log(f"test candidates={int(r[0]):,} across S1={int(r[1]):,}")
    con.close()


def build_test_features(artifact_dir:Path)->Path:
    con=connect_db(artifact_dir); out=artifact_dir/'test'/'test_features.tsv'
    if out.exists():
        con.close();return out
    # Reuse build_features logic with test-specific tables.
    if out.exists():out.unlink()
    count=con.execute('SELECT count(*) FROM test_candidates').fetchone()[0]
    total=0;first=True
    for offset in range(0,count,FEATURE_BATCH_SIZE):
        q=f"""
        SELECT b.*,s.name_norm,s.addr_norm,s.postal,s.country,s.name_first2,s.addr_first2,
               CASE WHEN b."source"='S2' THEN o2.name_norm ELSE o3.name_norm END other_name_norm,
               CASE WHEN b."source"='S2' THEN o2.addr_norm ELSE o3.addr_norm END other_addr_norm,
               CASE WHEN b."source"='S2' THEN o2.postal ELSE o3.postal END other_postal,
               CASE WHEN b."source"='S2' THEN o2.country ELSE o3.country END other_country,
               CASE WHEN b."source"='S2' THEN o2.name_first2 ELSE o3.name_first2 END other_name_first2,
               CASE WHEN b."source"='S2' THEN o2.addr_first2 ELSE o3.addr_first2 END other_addr_first2
        FROM (SELECT * FROM test_candidates ORDER BY s1_id,other_id LIMIT {FEATURE_BATCH_SIZE} OFFSET {offset}) b
        JOIN test_s1 s ON s.entity_id=b.s1_id
        LEFT JOIN test_s2 o2 ON b."source"='S2' AND o2.entity_id=b.other_id
        LEFT JOIN test_s3 o3 ON b."source"='S3' AND o3.entity_id=b.other_id
        """
        df=con.execute(q).fetchdf()
        if df.empty:continue
        pair_features(df).to_csv(out,sep='\t',index=False,mode='a',header=first);first=False;total+=len(df)
        if total%100000<FEATURE_BATCH_SIZE:log(f'test features {total:,}/{count:,}')
    con.close();return out


def predict(artifact_dir:Path,output_dir:Path)->None:
    meta=read_json(artifact_dir/'validation'/'best_threshold.json'); threshold=float(meta['threshold'])
    model=XGBClassifier();model.load_model(str(artifact_dir/'models/matcher.json'))
    feat=build_test_features(artifact_dir); scored=artifact_dir/'test'/'scored.tsv'
    if scored.exists():scored.unlink()
    first=True;total=0
    for ch in pd.read_csv(feat,sep='\t',chunksize=FEATURE_BATCH_SIZE):
        ch['proba']=model.predict_proba(ch[FEATURE_COLS].astype(np.float32))[:,1]
        ch[['s1_id','other_id','source','proba']].to_csv(scored,sep='\t',index=False,mode='a',header=first);first=False;total+=len(ch)
        if total%100000<FEATURE_BATCH_SIZE:log(f'scored {total:,}')
    con=connect_db(artifact_dir);con.execute('DROP TABLE IF EXISTS test_scored');con.execute(f"CREATE TABLE test_scored AS SELECT * FROM read_csv('{qpath(scored)}',delim='\\t',header=true,all_varchar=false)")
    output_dir.mkdir(parents=True,exist_ok=True)
    m=output_dir/'matching_results.tsv';c=output_dir/'candidate_pairs.tsv'
    con.execute(f"""COPY (SELECT s.entity_id source1_entity_id,coalesce(string_agg(DISTINCT t.other_id,',' ORDER BY t.other_id),'') matched_entity_ids FROM test_s1 s LEFT JOIN (SELECT s1_id,other_id FROM test_scored WHERE try_cast(proba AS DOUBLE)>={threshold}) t ON t.s1_id=s.entity_id GROUP BY s.entity_id ORDER BY s.entity_id) TO '{qpath(m)}' (HEADER,DELIMITER '\t')""")
    con.execute(f"""COPY (SELECT s.entity_id source1_entity_id,coalesce(string_agg(DISTINCT c.other_id,',' ORDER BY c.other_id),'') candidate_entity_ids FROM test_s1 s LEFT JOIN test_candidates c ON c.s1_id=s.entity_id GROUP BY s.entity_id ORDER BY s.entity_id) TO '{qpath(c)}' (HEADER,DELIMITER '\t')""")
    con.close();log(f'wrote {m}');log(f'wrote {c}')

# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_outputs(data_dir:Path,output_dir:Path)->int:
    m=output_dir/'matching_results.tsv';c=output_dir/'candidate_pairs.tsv'
    if not m.exists() or not c.exists():log('output validation: files missing');return 1
    # Avoid loading millions of rows into pandas: DuckDB does the structural checks.
    con=duckdb.connect()
    test_s1=int(con.execute(f"SELECT count(*) FROM read_csv('{qpath(data_dir/'test/test_source1.tsv')}',delim='\\t',header=true,all_varchar=true)").fetchone()[0])
    mr=int(con.execute(f"SELECT count(*) FROM read_csv('{qpath(m)}',delim='\\t',header=true,all_varchar=true)").fetchone()[0])
    cr=int(con.execute(f"SELECT count(*) FROM read_csv('{qpath(c)}',delim='\\t',header=true,all_varchar=true)").fetchone()[0])
    dupm=int(con.execute(f"SELECT count(*)-count(DISTINCT source1_entity_id) FROM read_csv('{qpath(m)}',delim='\\t',header=true,all_varchar=true)").fetchone()[0])
    dupc=int(con.execute(f"SELECT count(*)-count(DISTINCT source1_entity_id) FROM read_csv('{qpath(c)}',delim='\\t',header=true,all_varchar=true)").fetchone()[0])
    log(f'output rows matching={mr:,} candidates={cr:,} expected_s1={test_s1:,}')
    if mr!=test_s1 or cr!=test_s1 or dupm or dupc:
        log(f'FAIL: row/duplicate checks; dup_matching={dupm}, dup_candidates={dupc}');return 1
    log('basic output validation: PASS');return 0

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default=DEFAULT_DATA_DIR)
    ap.add_argument('--artifact-dir',default=DEFAULT_ARTIFACT_DIR)
    ap.add_argument('--out-dir',default=DEFAULT_OUTPUT_DIR)
    ap.add_argument('--stage',choices=['audit','prepare-train','block-train','validate','prepare-test','predict','all'],required=True)
    ap.add_argument('--work-s1',type=int,default=DEFAULT_WORK_S1)
    ap.add_argument('--val-s1',type=int,default=DEFAULT_VAL_S1)
    ap.add_argument('--no-extended-blocks',action='store_true',
                     help='Disable rare-token and n-gram blocking, keeping only exact/composite key blocks (faster, lower recall).')
    args=ap.parse_args()
    extended = not args.no_extended_blocks
    data=Path(args.data_dir);art=Path(args.artifact_dir);out=Path(args.out_dir);ensure_dirs(art,out)
    if args.stage=='audit': audit(data,art);return
    if args.stage=='prepare-train': prepare_train(data,art,args.work_s1,args.val_s1);return
    if args.stage=='block-train': block_train(art,use_extended_blocks=extended);return
    if args.stage=='validate':
        if not (art/'train'/'prepared.json').exists(): prepare_train(data,art,args.work_s1,args.val_s1)
        train_and_validate(data,art,extended);return
    if args.stage=='prepare-test': prepare_test(data,art);block_test(art);return
    if args.stage=='predict':
        if not (art/'test'/'prepared.json').exists(): prepare_test(data,art)
        if not (art/'validation'/'best_threshold.json').exists(): raise RuntimeError('Run --stage validate before predict.')
        block_test(art);predict(art,out);validate_outputs(data,out);return
    if args.stage=='all':
        audit(data,art);prepare_train(data,art,args.work_s1,args.val_s1);train_and_validate(data,art,extended)
        log('ALL stops after validation by design. Run prepare-test and predict after reviewing F0.5.')

if __name__=='__main__':
    main()
