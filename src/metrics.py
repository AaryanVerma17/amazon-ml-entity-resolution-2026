"""F_beta computed PER S1 ENTITY then macro-averaged — matches the official
evaluation exactly, including the rule that a correctly-predicted singleton
(empty predicted, empty true) scores 1.0, and predicting anything for a true
singleton scores 0.0.
"""


def f_beta_per_entity(true_set: set, pred_set: set, beta: float = 0.5) -> float:
    if not true_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def macro_f_beta(true_dict: dict, pred_dict: dict, beta: float = 0.5) -> float:
    """true_dict / pred_dict: {s1_entity_id: set(matched_ids)}.
    Averages over every key in true_dict (every S1 entity in the eval set).
    """
    scores = []
    for s1_id, true_set in true_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        scores.append(f_beta_per_entity(true_set, pred_set, beta))
    return sum(scores) / len(scores) if scores else 0.0
