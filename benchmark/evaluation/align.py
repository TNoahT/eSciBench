import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from Levenshtein import ratio
from fuzzywuzzy import fuzz
from scipy.optimize import linear_sum_assignment
from .metrics import compute_label_level_metrics


def lcs (s1: str, s2: str) -> str:
    m, n = len(s1), len(s2)
    dp = [[""] * (n+1) for _ in range(m + 1)]
    
    for i in range(m):
        for j in range (n):
            if s1[i] == s2[j] :
                dp[i+1][j+1] = dp[i][j] + s1[i]
            else:
                dp[i+1][j+1] = max (dp[i][j+1], dp[i+1][j], key=len)
    
    return dp[m][n]

def lcs_similarity(s1: str, s2:str) -> float:
    if not s1 or not s2:
        return 0.0
    return len(lcs(s1, s2)) / max (len(s1), len(s2))


def align_group_optimal(df_ex, df_gt, sim_threshold=0.7):
    """
    Aligns extracted entries to ground-truth entries using optimal bipartite matching.

    This method builds a similarity matrix between all extracted and ground-truth
    values using fuzzy string matching (fuzz.ratio). It then applies the Hungarian
    algorithm (via scipy's linear_sum_assignment) to compute the best global 
    one-to-one assignment.

    Unmatched extractions are considered false positives (fp_alignment).
    Unmatched ground-truth entries are false negatives (fn_alignment).
    True positives (tp_alignment) are pairs with similarity >= sim_threshold.

    Args:
        df_ex (pd.DataFrame): Extracted data. Must contain 'data_ex' and metadata 
                              columns: 'tool', 'pdf_name', 'page', 'label'.
        df_gt (pd.DataFrame): Ground truth data. Must contain 'data_gt' and the 
                              same metadata columns.
        sim_threshold (float): Minimum similarity score (between 0 and 1) for a match
                               to be considered a true positive.

    Returns:
        list[dict]: A list of alignment results. Each dict has:
            - metadata columns ('tool', 'pdf_name', etc.),
            - 'data_ex' (extracted string or None),
            - 'data_gt' (ground truth string or None),
            - 'match_type': one of {'tp_alignment', 'fp_alignment', 'fn_alignment'}
    """
    if df_ex.empty and df_gt.empty:
        return []
    elif df_ex.empty:
        return [{
            **{col: row[col] for col in {"tool", "pdf_name", "page", "label"}},
            "data_ex": None,
            "data_gt": row["data_gt"],
            "match_type": "fn_alignment"
        } for _, row in df_gt.iterrows()]
    elif df_gt.empty:
        return [{
            **row.to_dict(),
            "data_gt": None,
            "match_type": "fp_alignment"
        } for _, row in df_ex.iterrows()]
    
    ex_list = df_ex["data_ex"].tolist()
    gt_list = df_gt["data_gt"].tolist()
    
    cost_matrix = np.zeros((len(ex_list), len(gt_list)))
    for i, ex in enumerate(ex_list):
        for j, gt in enumerate(gt_list):
            #sim = fuzz.ratio(ex, gt) / 100
            sim = ratio(ex, gt)
            #sim = lcs_similarity(ex, gt)
            #sim = SequenceMatcher(None, ex, gt).ratio()
            cost_matrix[i, j] = -sim  # Hungarian algorithm minimizes

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    used_ex = set()
    used_gt = set()
    
    for i, j in zip(row_ind, col_ind):
        sim = -cost_matrix[i, j]
        ex_row = df_ex.iloc[i]
        gt_row = df_gt.iloc[j]
        if sim >= sim_threshold:
            matches.append({
                **ex_row.to_dict(),
                "data_gt": gt_row["data_gt"],
                "match_type": "tp_alignment"
            })
            used_ex.add(i)
            used_gt.add(j)

    # False Positives
    for i, ex_row in enumerate(df_ex.itertuples()):
        if i not in used_ex:
            matches.append({
                **df_ex.iloc[i].to_dict(),
                "data_gt": None,
                "match_type": "fp_alignment"
            })

    # False Negatives
    for j, gt_row in enumerate(df_gt.itertuples()):
        if j not in used_gt:
            matches.append({
                **{col: getattr(gt_row, col) for col in {"tool", "pdf_name", "page", "label"}},
                "data_ex": None,
                "data_gt": getattr(gt_row, "data_gt"),
                "match_type": "fn_alignment"
            })

    return matches


def align_group (df_ex, df_gt, sim_threshold=0.8):
    """
    Aligns extracted entries to ground-truth entries using fuzzy string matching.

    Uses fuzzywuzzy's fuzz.ratio to match `data_ex` to `data_gt`.
    Only matches one-to-one (greedy), and tracks unmatched ground-truth entries as false negatives.

    Args:
        df_ex (pd.DataFrame): DataFrame containing extracted data. Must include columns: 'data_ex', 'tool', 'pdf_name', 'page', 'label'.
        df_gt (pd.DataFrame): DataFrame containing ground truth data. Must include column 'data_gt' and metadata columns.
        sim_threshold (float): Minimum similarity score (0–1) required for a match.

    Returns:
        list[dict]: List of aligned records. Each entry has:
            - 'data_ex' from extracted,
            - 'data_gt' from matched ground truth (or None),
            - associated metadata (tool, pdf_name, etc).
    """
    matches = []
    used_gt = set()

    false_positives = []
    false_negatives = []
    true_positives = []
    
    # 1. Match extracted elements to ground truth
    for i, ex_row in df_ex.iterrows():
        best_j = None
        best_score = 0.0
        #print(df_gt)
        for gt_row_index, gt_row in df_gt.iterrows():
            #print("BBBB")
            if gt_row_index in used_gt:
                continue
            
            score = fuzz.ratio(ex_row["data_ex"], gt_row["data_gt"]) / 100
            #print(f"score: {score}, data ex : {ex_row['data_ex']}")
            if score > best_score:
                best_score = score
                best_j = gt_row_index

        # If there is a 'matching' ground truth
        if best_score >= sim_threshold and best_j is not None:
            used_gt.add(best_j)

            tp = (ex_row["data_ex"], df_gt.at[best_j, "data_gt"])
            true_positives.append(tp)

            matches.append({
                **ex_row.to_dict(),
                "data_gt": df_gt.loc[best_j, "data_gt"],
                "match_type": "tp_alignment"
            })
        # Else, no matching ground truth. It is then a false positive
        else :
            matches.append({
                **ex_row.to_dict(),
                "data_gt": None,
                "match_type": "fp_alignment"
            })
            false_positives.append(ex_row["data_ex"])

    # 2. Unmatched GT (False negatives)
    for gt_row_index, gt_row in df_gt.iterrows():
        if gt_row_index not in used_gt:
            false_negatives.append(gt_row["data_gt"])
            unmatched_entry = {
                **{col: gt_row[col] for col in {"tool", "pdf_name", "page", "label"}},
                "data_ex": None,
                "data_gt": gt_row["data_gt"],
                "match_type": "fn_alignment"
            }
            matches.append(unmatched_entry)
    #print("*************************************************")
    #print(df_ex["label"].unique() + "       " + df_ex["pdf_name"].unique())
    #print(f"True positives : {len(true_positives)}" )
    #print(f"False positives : {len(false_positives)}")
    #print(f"False negatives : {len(false_negatives)}")
    
    return matches


def align (df_ex, df_gt) :
    """
    Groups extracted and ground truth entries by metadata, and aligns each group.

    This is a wrapper around `align_group` that applies it to each group of
    (tool, pdf_name, page, label) and combines the results into a full DataFrame.

    Args:
        df_ex (pd.DataFrame): Extracted data with metadata and 'data_ex' column.
        df_gt (pd.DataFrame): Ground-truth data with metadata and 'data_gt' column.

    Returns:
        pd.DataFrame: A DataFrame containing aligned entries for all groups, including
        matched and unmatched records.
    """
    #print("HI")
    group_cols = ["tool", "pdf_name", "label"]
    aligned_all = []
    
    # Get union of all group keys in both DataFrames
    ex_keys = df_ex.groupby(group_cols).groups.keys()
    gt_keys = df_gt.groupby(group_cols).groups.keys()
    all_keys = set(ex_keys).union(set(gt_keys))

    for group_key in all_keys:
        ex_group = df_ex[
            (df_ex["tool"] == group_key[0]) &
            (df_ex["pdf_name"] == group_key[1]) &
            (df_ex["label"] == group_key[2])
        ]

        gt_group = df_gt[
            (df_gt["tool"] == group_key[0]) &
            (df_gt["pdf_name"] == group_key[1]) &
            (df_gt["label"] == group_key[2])
        ]
        #aligned = align_group(ex_group, gt_group)
        aligned = align_group_optimal(ex_group, gt_group)
        aligned_all.extend(aligned)

    aligned_df = pd.DataFrame(aligned_all)
    #print(aligned_df["match_type"].value_counts)

    aligned_metric_df = compute_label_level_metrics(aligned_df)
    #print(aligned_metric_df.to_string(index=False))
    return aligned_metric_df


def align_block_strings(df_ex: pd.DataFrame, df_gt: pd.DataFrame) -> pd.DataFrame:
    """
    For each tool, PDF and label, concatenate all extracted strings
    (across pages) and all ground-truth strings, then align them.
    Returns a DataFrame of aligned rows ready for compute_metrics().
    """
    global_rows = []

    # collect every (tool, pdf_name, label) combo seen in either df
    keys_ex = set(tuple(x) for x in df_ex[["tool","pdf_name","label"]].values)
    keys_gt = set(tuple(x) for x in df_gt[["tool","pdf_name","label"]].values)
    for tool, pdf_name, label in keys_ex | keys_gt:
        ex_texts = df_ex.loc[
            (df_ex.tool==tool)&(df_ex.pdf_name==pdf_name)&(df_ex.label==label),
            "data_ex"
        ].dropna().tolist()
        gt_texts = df_gt.loc[
            (df_gt.tool==tool)&(df_gt.pdf_name==pdf_name)&(df_gt.label==label),
            "data_gt"
        ].dropna().tolist()

        global_rows.append({
            "tool":     tool,
            "pdf_name": pdf_name,
            "page":     -1,               # marker for “global” row
            "label":    label,
            "data_ex":  " ".join(ex_texts),
            "data_gt":  " ".join(gt_texts),
        })

    if not global_rows:
        return pd.DataFrame()

    global_df = pd.DataFrame(global_rows)

    # align those big-string rows just like you do per item
    aligned_list = align_group_optimal(
        global_df[["tool","pdf_name","page","label","data_ex"]],
        global_df[["tool","pdf_name","page","label","data_gt"]],
    )

    # wrap in a DataFrame so you can call .empty, etc.
    return pd.DataFrame(aligned_list)
