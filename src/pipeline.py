"""End-to-end: load -> normalize -> block -> feature -> train -> tune ->
predict -> write matching_results.tsv + candidate_pairs.tsv.

Run:  python pipeline.py --data-dir dataset --out-dir output
"""
import argparse
import os
import pandas as pd
import numpy as np

from normalize import normalize_row
from blocking import generate_candidates
from features import build_features, FEATURE_COLS
from model import train_classifier, predict_proba, tune_threshold
from metrics import macro_f_beta


def load_source(path):
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def load_ground_truth(path):
    gt = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    out = {}
    for _, row in gt.iterrows():
        ids = row["matched_entity_ids"].strip()
        out[row["source1_entity_id"]] = set(ids.split(",")) if ids else set()
    return out


def candidates_with_source(s1_df, s2_df, s3_df):
    """Generate candidates against S2 and S3 separately, tag with source,
    return one combined frame with global other-index bookkeeping."""
    frames = []
    for label, odf in (("S2", s2_df), ("S3", s3_df)):
        if odf.empty:
            continue
        c = generate_candidates(s1_df, odf)
        c["source"] = label
        frames.append(c)
    if not frames:
        return pd.DataFrame(columns=["s1_idx", "other_idx", "source"])
    return pd.concat(frames, ignore_index=True)


def norm_list(df):
    return [normalize_row(r["business_name"], r["business_address"]) for _, r in df.iterrows()]


def build_feature_table(s1_df, s2_df, s3_df, cand_all):
    s1_norm = norm_list(s1_df)
    s2_norm = norm_list(s2_df)
    s3_norm = norm_list(s3_df)
    s1_country = s1_df["country"].tolist()
    s2_country = s2_df["country"].tolist() if not s2_df.empty else []
    s3_country = s3_df["country"].tolist() if not s3_df.empty else []

    parts = []
    for source, other_df, other_norm, other_country in (
        ("S2", s2_df, s2_norm, s2_country), ("S3", s3_df, s3_norm, s3_country),
    ):
        sub = cand_all[cand_all["source"] == source]
        if sub.empty:
            continue
        feat = build_features(sub, s1_norm, other_norm, s1_country, other_country)
        if feat.empty:
            continue
        feat["source"] = source
        feat["s1_id"] = feat["s1_idx"].map(lambda i: s1_df.iloc[i]["entity_id"])
        feat["other_id"] = feat["other_idx"].map(lambda j: other_df.iloc[j]["entity_id"])
        parts.append(feat)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def label_pairs(feat_df, gt_dict):
    def is_match(row):
        return int(row["other_id"] in gt_dict.get(row["s1_id"], set()))
    feat_df["label"] = feat_df.apply(is_match, axis=1)
    return feat_df


def entity_level_split(s1_ids, val_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    ids = list(s1_ids)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    return set(ids[n_val:]), set(ids[:n_val])  # train_ids, val_ids


def write_outputs(pred_df, candidate_df, all_s1_ids, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    def _rows(df):
        agg = df.groupby("s1_id")["other_id"].apply(lambda s: ",".join(sorted(set(s))))
        rows = []
        for s1 in all_s1_ids:
            rows.append({"source1_entity_id": s1, "matched_entity_ids": agg.get(s1, "")})
        return pd.DataFrame(rows)

    match_out = _rows(pred_df)
    match_out.columns = ["source1_entity_id", "matched_entity_ids"]
    match_out.to_csv(os.path.join(out_dir, "matching_results.tsv"), sep="\t", index=False)

    cand_out = _rows(candidate_df)
    cand_out.columns = ["source1_entity_id", "candidate_entity_ids"]
    cand_out.to_csv(os.path.join(out_dir, "candidate_pairs.tsv"), sep="\t", index=False)


def run(data_dir, out_dir):
    train_s1 = load_source(f"{data_dir}/train/train_source1.tsv")
    train_s2 = load_source(f"{data_dir}/train/train_source2.tsv")
    train_s3 = load_source(f"{data_dir}/train/train_source3.tsv")
    gt = load_ground_truth(f"{data_dir}/train/train_ground_truth.tsv")

    train_ids, val_ids = entity_level_split(train_s1["entity_id"].tolist())
    s1_train = train_s1[train_s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = train_s1[train_s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    # --- candidates + features on train split ---
    cand_train = candidates_with_source(s1_train, train_s2, train_s3)
    feat_train = build_feature_table(s1_train, train_s2, train_s3, cand_train)
    feat_train = label_pairs(feat_train, gt)

    model = train_classifier(feat_train, feat_train["label"])

    # --- candidates + features on val split (recall ceiling check happens here) ---
    cand_val = candidates_with_source(s1_val, train_s2, train_s3)
    feat_val = build_feature_table(s1_val, train_s2, train_s3, cand_val)
    val_proba = predict_proba(model, feat_val)

    val_true = {s1: gt.get(s1, set()) for s1 in s1_val["entity_id"]}
    best_t, best_score, table = tune_threshold(
        feat_val, val_proba, feat_val["s1_id"].tolist(), feat_val["other_id"].tolist(), val_true,
    )
    print(f"[val] best threshold={best_t}  macro F0.5={best_score:.4f}")
    print(table.to_string(index=False))

    # --- candidate recall ceiling: how many true matches ever became candidates? ---
    all_true_pairs = {(s1, m) for s1, ms in val_true.items() for m in ms}
    cand_pairs = set(zip(feat_val["s1_id"], feat_val["other_id"]))
    if all_true_pairs:
        recall_ceiling = len(all_true_pairs & cand_pairs) / len(all_true_pairs)
        print(f"[val] candidate recall ceiling = {recall_ceiling:.4f}  "
              f"({len(cand_pairs)} candidates for {len(s1_val)} S1 entities)")

    # --- refit on full training data, predict on test ---
    full_cand = candidates_with_source(train_s1, train_s2, train_s3)
    full_feat = build_feature_table(train_s1, train_s2, train_s3, full_cand)
    full_feat = label_pairs(full_feat, gt)
    final_model = train_classifier(full_feat, full_feat["label"])

    test_s1 = load_source(f"{data_dir}/test/test_source1.tsv")
    test_s2 = load_source(f"{data_dir}/test/test_source2.tsv")
    test_s3 = load_source(f"{data_dir}/test/test_source3.tsv")

    test_cand = candidates_with_source(test_s1, test_s2, test_s3)
    test_feat = build_feature_table(test_s1, test_s2, test_s3, test_cand)
    if test_feat.empty:
        pred_df = pd.DataFrame(columns=["s1_id", "other_id"])
    else:
        test_proba = predict_proba(final_model, test_feat)
        pred_df = test_feat.loc[test_proba >= best_t, ["s1_id", "other_id"]]

    write_outputs(pred_df, test_feat[["s1_id", "other_id"]] if not test_feat.empty else pred_df,
                  test_s1["entity_id"].tolist(), out_dir)
    print(f"wrote outputs to {out_dir}/matching_results.tsv and {out_dir}/candidate_pairs.tsv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()
    run(args.data_dir, args.out_dir)
