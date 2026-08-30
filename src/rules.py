"""
src/rules.py — Rule extraction from XGBoost + rule-as-feature application.

The structure-transfer regime: turn MCS-trained XGBoost trees into an interpretable
rule set, then feed those rules as engineered features to a downstream logistic
head — fitted either on MCS labels (label-free variant) or refitted on YRBS
(label-requiring variant). `transfer.rule_head` is the live consumer; it imports
`extract_rules_from_xgb` and `apply_rules` from here at call time.

How a rule is defined here:
- One root-to-leaf path in one tree, as a list of (feature, op, threshold) tuples.
- Ranked by |leaf value| × leaf cover — magnitude times how many training rows saw it.
- Deduplicated across trees on the condition set, with duplicate weights summed.
- A rule does NOT fire on a NaN input. That is deliberately conservative: XGBoost's
  own default-direction routing is not replicated, which the paper states as a design
  choice. It costs little in practice, since rules usually condition on several
  features and a missing one is rarely the only thing keeping a rule from firing.
"""
import numpy as np
import pandas as pd


# Rule extraction
def extract_rules_from_xgb(model, top_k=100):
    """Extract top-k unique root-to-leaf rules from a fitted XGBClassifier.

    Returns list of dicts: [{conditions: [(feat, op, thresh), ...], weight: float}, ...]
    Sorted by descending weight. Conditions are ordered from root to leaf.
    """
    df = model.get_booster().trees_to_dataframe()
    rules_raw = []
    for tree_id in df["Tree"].unique():
        tree = df[df["Tree"] == tree_id].set_index("ID")
        root_id = f"{tree_id}-0"
        _walk(tree, root_id, [], rules_raw)

    # Dedupe by set-of-conditions; sum weights of duplicates
    seen = {}
    for conds, weight in rules_raw:
        key = frozenset(conds)
        if key in seen:
            seen[key] = (conds, seen[key][1] + weight)
        else:
            seen[key] = (conds, weight)

    unique = sorted(seen.values(), key=lambda r: -abs(r[1]))
    return [dict(conditions=list(c), weight=float(w)) for c, w in unique[:top_k]]


def _walk(tree, node_id, conditions, out):
    """DFS from node_id, appending (conditions, weight) at each leaf."""
    node = tree.loc[node_id]
    if node["Feature"] == "Leaf":
        # Leaf value stored in 'Gain' column for leaves in trees_to_dataframe
        weight = abs(float(node["Gain"])) * float(node["Cover"])
        out.append((tuple(conditions), weight))
        return
    feat, split = node["Feature"], float(node["Split"])
    yes_id, no_id = node["Yes"], node["No"]
    _walk(tree, yes_id, conditions + [(feat, "<",  split)], out)
    _walk(tree, no_id,  conditions + [(feat, ">=", split)], out)


# Rule application
def apply_rules(rules, X):
    """Convert a rule list into a binary feature DataFrame aligned to X's index.

    Rule R_i fires (=1) on row x iff ALL its conditions are satisfied.
    NaN in any conditioned feature => rule does not fire (conservative).
    """
    n = len(X)
    features = {}
    for i, rule in enumerate(rules):
        mask = np.ones(n, dtype=bool)
        for feat, op, thresh in rule["conditions"]:
            col = X[feat].values
            valid = ~np.isnan(col) if np.issubdtype(col.dtype, np.floating) else np.ones(n, dtype=bool)
            if op == "<":
                mask &= valid & (col < thresh)
            elif op == ">=":
                mask &= valid & (col >= thresh)
            else:
                raise ValueError(f"unknown op: {op}")
        features[f"rule_{i:03d}"] = mask.astype(np.int8)
    return pd.DataFrame(features, index=X.index)


