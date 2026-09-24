"""XGBoost pair classifier + threshold search that optimizes macro F0.5
(the actual competition metric), not accuracy or pooled F1.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from features import FEATURE_COLS
from metrics import macro_f_beta

THRESHOLD_GRID = np.round(np.arange(0.30, 0.99, 0.02), 2)


def train_classifier(X: pd.DataFrame, y: pd.Series, sample_weight=None) -> XGBClassifier:
    # XGBoost is Apache-2.0 licensed; a few hundred trees stays well under
    # any parameter-count constraint aimed at neural / LLM-scale models.
    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        eval_metric="logloss", n_jobs=-1,
        scale_pos_weight=_pos_weight(y),
    )
    model.fit(X[FEATURE_COLS], y, sample_weight=sample_weight)
    return model


def _pos_weight(y: pd.Series) -> float:
    pos = max(int(y.sum()), 1)
    neg = max(len(y) - pos, 1)
    return neg / pos


def predict_proba(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[FEATURE_COLS])[:, 1]


def tune_threshold(val_feat: pd.DataFrame, val_proba: np.ndarray,
                    val_s1_ids: list, val_other_ids: list,
                    true_dict: dict) -> tuple:
    """Search THRESHOLD_GRID, return (best_threshold, best_score, table).

    val_feat rows, val_proba, val_s1_ids, val_other_ids must be aligned
    (same order / index).
    """
    df = pd.DataFrame({
        "s1_id": val_s1_ids, "other_id": val_other_ids, "proba": val_proba,
    })
    all_s1 = set(true_dict.keys())
    results = []
    best_t, best_score = 0.5, -1.0
    for t in THRESHOLD_GRID:
        pred_dict = {s1: set() for s1 in all_s1}
        matched = df[df["proba"] >= t]
        for s1_id, grp in matched.groupby("s1_id"):
            if s1_id in pred_dict:
                pred_dict[s1_id] = set(grp["other_id"])
        score = macro_f_beta(true_dict, pred_dict, beta=0.5)
        results.append({"threshold": t, "macro_f0.5": score})
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score, pd.DataFrame(results)