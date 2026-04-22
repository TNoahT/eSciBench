import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from Levenshtein import ratio
import re
import heapq
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def normalize_text(s: str) -> str:
    """
    Normalize a string for comparison.

    - Collapses all whitespace to single spaces.
    - Converts to lowercase.
    - Strips punctuation characters.

    Args:
        s (str): Input string to normalize.

    Returns:
        str: Normalized string.
    """
    # Collapse multiple spaces/tabs/newlines into a single space
    text = " ".join(s.split())
    # Convert to lowercase for case-insensitive comparison
    text = text.lower()
    # Remove leading/trailing punctuation
    text = re.sub(r"[^\w\s]", "", text) # Remove all punctuation
    return text


def compute_sim_matrix(data_ex: np.ndarray, data_gt: np.ndarray) -> pd.DataFrame:
    """
    Compute a pairwise Levenshtein similarity matrix between all extracted and ground-truth tokens.

    Args:
        data_ex (np.ndarray): Array of extracted tokens.
        data_gt (np.ndarray): Array of ground-truth tokens.

    Returns:
        pd.DataFrame: Similarity matrix with extracted tokens as rows and GT tokens as columns.
    """
    matrix = np.zeros((len(data_ex), len(data_gt)))
    for i, ex in enumerate(data_ex):
        for j, gt in enumerate(data_gt):
            matrix[i, j] = ratio(ex, gt)
    return pd.DataFrame(matrix, index=data_ex, columns=data_gt)


def compute_tp_fp_fn(sim_matrix: pd.DataFrame, threshold: float) -> tuple[int, int, int]:
    """
    Compute true positives (TP), false positives (FP), and false negatives (FN)
    using greedy 1-to-1 matching over a similarity matrix.

    Each extracted item (row) can be matched to at most one ground-truth item (column),
    and vice versa. Matching is based on maximum similarity above a threshold.

    Args:
        sim_matrix (pd.DataFrame): Similarity matrix between extracted (rows) and GT (columns), using Lenvenshtein ratio for the values.
        threshold (float): Minimum similarity to count as a match.

    Returns:
        tuple[int, int, int]: (TP, FP, FN)
    """
    if sim_matrix.empty:
        return 0, 0, 0

    # Convert to numpy array for faster access
    sim_array = sim_matrix.values
    # Store matches above threshold as (-score, i, j) for max-heap
    matches = [
        (-sim_array[i, j], i, j)
        for i in range(sim_array.shape[0])
        for j in range(sim_array.shape[1])
        if sim_array[i, j] >= threshold
    ]
    heapq.heapify(matches)
    
    matched_rows = set()
    matched_cols = set()
    tp = 0

    while matches:
        score, i, j = heapq.heappop(matches)
        if i not in matched_rows and j not in matched_cols:
            matched_rows.add(i)
            matched_cols.add(j)
            tp += 1
            

    fp = sim_array.shape[0] - len(matched_rows)  # unmatched extracted tokens
    fn = sim_array.shape[1] - len(matched_cols)  # unmatched ground-truth tokens

    return tp, fp, fn


def compute_exact_match(data_ex: str, data_gt: str) -> int:
    """
    Compute binary exact match after normalization.

    Args:
        data_ex (str): Extracted text.
        data_gt (str): Ground-truth text.

    Returns:
        int: 1 if normalized texts match exactly, else 0.
    """
    if not data_ex or not data_gt:
        return 0
    norm_ex = normalize_text(data_ex)       # TODO
    norm_gt = normalize_text(data_gt)       # TODO
    return int(norm_ex == norm_gt)


def compute_max_similarity(sim_matrix: pd.DataFrame) -> float:
    """
    Return the maximum similarity value from the similarity matrix.

    Args:
        sim_matrix (pd.DataFrame): Token-level similarity matrix.

    Returns:
        float: Maximum similarity score, or 0.0 if matrix is empty.
    """
    return float(sim_matrix.values.max()) if not sim_matrix.empty else 0.0


def compute_scores(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """
    Compute F1 score, precision, and recall from TP/FP/FN counts.

    Args:
        tp (int): True positives.
        fp (int): False positives.
        fn (int): False negatives.

    Returns:
        tuple[float, float, float]: (F1 score, precision, recall).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def compute_char_ngram_overlap(data_ex: str, data_gt: str, n: int = 3) -> tuple[float, float]:
    """
    Compute character-level n-gram precision and recall between normalized extracted and ground-truth texts.

    Args:
        data_ex (str): Extracted text.
        data_gt (str): Ground-truth text.
        n (int, optional): Size of character n-grams. Defaults to 3.

    Returns:
        tuple[float, float]: (precision, recall) based on character n-gram overlap.
    """
    def char_ngrams(s: str, n: int) -> set:
        s = normalize_text(s)
        return set(s[i:i+n] for i in range(len(s) - n + 1)) if len(s) >= n else set()

    ex_ngrams = char_ngrams(data_ex, n)
    gt_ngrams = char_ngrams(data_gt, n)

    if not ex_ngrams or not gt_ngrams:
        return 0.0, 0.0

    intersection = ex_ngrams & gt_ngrams
    precision = len(intersection) / len(ex_ngrams)
    recall = len(intersection) / len(gt_ngrams)

    return precision, recall


def compute_label_level_metrics(merged_df: pd.DataFrame) -> pd.DataFrame:
    # 1. Compute TP/FP/FN counts per group
    counts = (
        merged_df
        .groupby(['tool', 'label', 'match_type'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    #print(merged_df)
    
    # 2. Ensure all expected match_type columns exist
    for col in ["tp_alignment", "fp_alignment", "fn_alignment"]:
        if col not in counts:
            counts[col] = 0

    # 3. Compute precision, recall, F1
    tp = counts["tp_alignment"]
    fp = counts["fp_alignment"]
    fn = counts["fn_alignment"]

    # Avoid division by zero for precision and recall
    counts["precision_alignment"] = tp / (tp + fp)
    #print(f", fp: {fp}, fn: {fn}")
    counts["recall_alignment"] = tp / (tp + fn)

    # Replace NaNs (from 0/0) with 0
    counts["precision_alignment"] = counts["precision_alignment"].fillna(0)
    counts["recall_alignment"] = counts["recall_alignment"].fillna(0)

    # Compute F1 safely: if precision + recall == 0 → f1 = 0
    denominator = counts["precision_alignment"] + counts["recall_alignment"]
    counts["f1_alignment"] = (
        2 * counts["precision_alignment"] * counts["recall_alignment"] / denominator
    ).where(denominator != 0, 0.0)
    #print(counts)
    # 4. Merge metrics back into original dataframe
    merged_with_metrics = merged_df.merge(counts, on=["tool", "label"], how="left")
    return merged_with_metrics


def compute_average_label_metrics(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average F1, precision, and recall for each (tool, label) pair,
    both overall and for rows with non-empty data_ex and data_gt.

    Args:
        merged_df (pd.DataFrame): DataFrame with per-row metrics.

    Returns:
        pd.DataFrame: Per-(tool, label) averages with and without missing values.
    """
    METRICS = [
        'f1', 'precision', 'recall',
        'rouge1_recall', 'rouge2_recall', 'rouge3_recall',
        'lev_ratio', 'bleu_score'
    ]

    # 1) overall means
    overall = (
        merged_df
        .groupby(['tool','label'])[METRICS]
        .mean()
        .reset_index()
        .rename(columns={
            'f1': 'avg_f1_all',
            'precision': 'avg_precision_alignment',
            'recall': 'avg_recall_alignment',
            'rouge1_recall': 'avg_rouge1_recall_alignment',
            'rouge2_recall': 'avg_rouge2_recall_alignment',
            'rouge3_recall': 'avg_rouge3_recall_alignment',
            'lev_ratio': 'avg_lev_ratio_alignment',
            'bleu_score': 'avg_bleu_score_alignment',
        })
    )

    # 2) averages over only the rows with lev_ratio > 0.7
    non_zero = (
        merged_df[(merged_df['lev_ratio'] > 0.7)]
        .groupby(['tool','label'])[METRICS]
        .mean()
        .reset_index()
        .rename(columns={
            'lev_ratio': 'avg_lev_ratio_non_zero'
        })
    )

    # 3) merge the two metric sets
    result = pd.merge(overall, non_zero, on=['tool', 'label'], how='left').fillna(0)

    # 4) dedupe per pdf/page so we don't double-count
    align_df = (
        merged_df
        [['tool','label','pdf_name','page','tp_alignment','fp_alignment','fn_alignment']]
        .drop_duplicates(subset=['tool','label','pdf_name','page'])
    )
    
    # 5) compute totals & averages with named aggregation
    agg = (
        align_df
        .groupby(['tool', 'label'])
        .agg(
            tp_alignment=('tp_alignment', 'mean'),
            fp_alignment=('fp_alignment', 'mean'),
            fn_alignment=('fn_alignment', 'mean'),
        )
        # Round to nearest integer and convert dtype
        .round(0)
        .astype({
            'tp_alignment': int,
            'fp_alignment': int,
            'fn_alignment': int,
        })
        .reset_index()
    )

    # 6) calculate error metrics
    agg = agg.assign(
        false_discovery_rate = lambda df: (
            df['fp_alignment'] 
            / (df['tp_alignment'] + df['fp_alignment'])
        ).replace(0, np.nan).fillna(0),
        miss_rate = lambda df: (
            df['fn_alignment'] 
            / (df['tp_alignment'] + df['fn_alignment'])
        ).replace(0, np.nan).fillna(0),
        alignment_error_rate = lambda df: (
            (df['fp_alignment'] + df['fn_alignment'])
            / (df['tp_alignment'] + df['fp_alignment'] + df['fn_alignment'])
        ).replace(0, np.nan).fillna(0),
        accuracy = lambda df: 1 - (
            (df['fp_alignment'] + df['fn_alignment'])
            / (df['tp_alignment'] + df['fp_alignment'] + df['fn_alignment'])
        ).replace(0, np.nan).fillna(0)
    )

    # 7) merge into your existing result and fillna
    result = (
        result
        .merge(agg, on=['tool', 'label'], how='left')
        .fillna(0)
    )

    cols_to_keep = [
        'tool', 'label',
        'tp_alignment', 'fp_alignment', 'fn_alignment', 'accuracy',     
        'avg_f1_all', 'avg_precision_alignment', 'avg_recall_alignment',
        'avg_rouge1_recall_alignment', 'avg_rouge2_recall_alignment', 'avg_rouge3_recall_alignment',
        'avg_bleu_score_alignment', 'avg_lev_ratio_non_zero', 'avg_lev_ratio_alignment',
        'false_discovery_rate', 'miss_rate', 'alignment_error_rate'
    ]

    result = result[cols_to_keep]

    return result


def compute_recall_rouge_n (data_ex: str, data_gt: str, n: int) -> float :
    """
    Compute ROUGE-N recall, but if either string is shorter than n tokens,
    fall back to the maximum possible n-gram size.

    Args:
        data_ex (str): system output (candidate).
        data_gt (str): reference (ground truth).
        n (int): size of the n-grams.

    Returns:
        float: ROUGE-N recall = overlap / total reference n-grams, in [0.0,1.0].
    """

    ex_toks = data_ex.split()
    gt_toks = data_gt.split()

    # figure out the largest n-gram we can actually build
    max_n = min(n, len(ex_toks), len(gt_toks))
    if max_n < 1:
        return 0.0

    # build the n-grams
    def ngrams(tokens, k):
        return Counter(zip(*[tokens[i:] for i in range(k)]))

    ex_ngrams = ngrams(ex_toks, max_n)
    gt_ngrams = ngrams(gt_toks, max_n)

    if not ex_ngrams:
        return 0.0

    overlap = sum((gt_ngrams & ex_ngrams).values())
    return overlap / sum(ex_ngrams.values())


def compute_dynamic_sentence_bleu(candidate: str, reference: str, max_n: int = 4):
    """
    Compute sentence-level BLEU but only up to N-grams that the sentence can support.
    For len(tokens)=k < max_n, assigns equal weight to the first k orders, zeros thereafter.
    """
    ref_toks = reference.split()
    hyp_toks = candidate.split()
    k = len(hyp_toks)
    if k == 0:
        return 0.0

    usable = min(k, max_n)
    # e.g. for usable=2 → weights=(0.5,0.5,0,0)
    base_weight = 1.0 / usable
    weights = tuple(
        base_weight if i < usable else 0.0
        for i in range(max_n)
    )

    return sentence_bleu(
        [ref_toks],
        hyp_toks,
        weights=weights,
        smoothing_function=SmoothingFunction().method4
    )


def compute_metrics(merged_df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Compute multiple evaluation metrics for each row of the merged results DataFrame.

    Includes:
      - F1, precision, recall
      - Token-level and character n-gram metrics
      - Max similarity
      - Exact match

    Args:
        merged_df (pd.DataFrame): Merged DataFrame containing both 'data_ex' and 'data_gt' columns.
        threshold (float, optional): Similarity threshold for TP/FP/FN computation. Defaults to 0.7.

    Returns:
        pd.DataFrame: Same DataFrame with additional metric columns.
    """
    print("[LOG] Computing metrics")
    merged_df = merged_df.copy()
    f1s, precisions, recalls = [], [], []
    token_precs, token_recs, exact_matches, max_sims, len_ex, len_gt = [], [], [], [], [], []
    lev_ratios = []
    rouge1, rouge2, rouge3 = [], [], []
    bleu_score = []

    sum_f1 = 0
    sum_prec = 0
    sum_recall = 0

    num_all_sentences = 0
    num_not_matched_sentences = 0


    for _, row in merged_df.iterrows():
        
        num_all_sentences += 1

        #data_ex = str(row.get("data_ex", "")) if pd.notna(row.get("data_ex")) else ""
        #data_gt = str(row.get("data_gt", "")) if pd.notna(row.get("data_gt")) else ""
        # grab the raw cell
        val_ex = row.get("data_ex", "")
        if isinstance(val_ex, (list, np.ndarray)):
            # if it's a list/array, flatten it into a string (or skip metrics)
            data_ex = " ".join(map(str, val_ex)) if len(val_ex) else ""
        elif pd.isna(val_ex):
             data_ex = ""
        else:
            data_ex = str(val_ex)

        val_gt = row.get("data_gt", "")
        if isinstance(val_gt, (list, np.ndarray)):
            data_gt = " ".join(map(str, val_gt)) if len(val_gt) else ""
        elif pd.isna(val_gt):
             data_gt = ""
        else:
            data_gt = str(val_gt)

        # Compute full-string Levenshtein ratio
        lr = ratio(data_ex, data_gt)
        #print(lr)
        lev_ratios.append(lr)

        # Split into tokens
        ex_tokens_arr = np.array(data_ex.split()) if data_ex else np.array([])
        gt_tokens_arr = np.array(data_gt.split()) if data_gt else np.array([])

        if ex_tokens_arr.size == 0 or gt_tokens_arr.size == 0:
            # No meaningful match possible
            tp, fp, fn = 0, ex_tokens_arr.size, gt_tokens_arr.size
            sim_max = 0.0
            tok_prec, tok_rec = 0.0, 0.0
            exact = compute_exact_match(data_ex, data_gt)
            r1 = r2 = r3 = 0.0
            bleu = 0.0
            num_not_matched_sentences += 1 # For averages
        else:
            #print("[LOG] Computing similarity matrix")
            sim_matrix = compute_sim_matrix(ex_tokens_arr, gt_tokens_arr)
            #print("[LOG] Computing tp, fp, fn")
            tp, fp, fn = compute_tp_fp_fn(sim_matrix, threshold=threshold)
            
            #print(f"[LOG] tp: {tp}, fp : {fp}, fn : {fn}")
            sim_max = compute_max_similarity(sim_matrix)
            #tok_prec, tok_rec = compute_token_overlap_metrics(data_ex, data_gt)
            tok_prec, tok_rec = compute_char_ngram_overlap(data_ex, data_gt, n=3)
            r1 = compute_recall_rouge_n(data_ex, data_gt, 1)
            r2 = compute_recall_rouge_n(data_ex, data_gt, 2)
            r3 = compute_recall_rouge_n(data_ex, data_gt, 3)
            bleu = compute_dynamic_sentence_bleu(data_ex, data_gt)
            exact = compute_exact_match(data_ex, data_gt)
            
        #print("[LOG] Computing scores")
        #print(row.get("tp_alignment", 0))
        tp = row.get("tp_alignment", 0)
        fp = row.get("fp_alignment", 0)
        fn = row.get("fn_alignment", 0)
        f1, prec, rec = compute_scores(tp, fp, fn)

        # Sums for averages
        sum_f1 += f1
        sum_prec += prec
        sum_recall += rec

        f1s.append(f1)
        precisions.append(prec)
        recalls.append(rec)
        token_precs.append(tok_prec)
        token_recs.append(tok_rec)
        exact_matches.append(exact)
        rouge1.append(r1)
        rouge2.append(r2)
        rouge3.append(r3)
        bleu_score.append(bleu)
        max_sims.append(sim_max)
        len_ex.append(len(data_ex))
        len_gt.append(len(data_gt))

    
    # Add new columns to DataFrame
    merged_df.loc[:, "lev_ratio"]       = lev_ratios
    merged_df.loc[:, "f1"]              = f1s
    merged_df.loc[:, "precision"]       = precisions
    merged_df.loc[:, "recall"]          = recalls
    merged_df.loc[:, "token_precision"] = token_precs
    merged_df.loc[:, "token_recall"]    = token_recs
    merged_df.loc[:, "exact_match"]     = exact_matches
    merged_df.loc[:, "max_similarity"]  = max_sims
    merged_df.loc[:, "len_ex"]          = len_ex
    merged_df.loc[:, "len_gt"]          = len_gt
    merged_df.loc[:, "rouge1_recall"]   = rouge1
    merged_df.loc[:, "rouge2_recall"]   = rouge2
    merged_df.loc[:, "rouge3_recall"]   = rouge3
    merged_df.loc[:, "bleu_score"]      = bleu_score

    #print(f1s)
    averaged_df = compute_average_label_metrics(merged_df)
    return merged_df, averaged_df
