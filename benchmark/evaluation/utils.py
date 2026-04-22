import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict
from ..extractors.pdf_utils import PDF


def label_as_one_row_ex(extracted:list[dict]) -> list[dict]:
    """
    Combine extracted text chunks across all the extracted data for each (pdf_name, label, tool) into one row.

    Args:
        extracted (list[dict]): List of extracted data entries.

    Returns:
        list[dict]: List of dictionaries with one entry per (pdf_name, label, tool), combining all 'data_ex'.
    """
    grouped = defaultdict(list)

    for item in extracted :
        key = (item['pdf_name'], item['label'], item['tool'])
        grouped[key].append(item['data_ex'])

    rows = []
    for (pdf_name, label, tool), texts in grouped.items():
        combined_text = ' '.join(map(str, texts))
        rows.append({
            "tool": tool,
            "pdf_name": pdf_name,
            "page": 0,
            "label": label,
            "data_ex": combined_text
        })
    return rows

def label_as_one_row_gt(extracted:list[dict]) -> list[dict]:
    """
    Combine ground-truth text chunks across all ground-truth for each (pdf_name, label, tool) into one row.

    Args:
        extracted (list[dict]): List of ground-truth data entries.

    Returns:
        list[dict]: List of dictionaries with one entry per (pdf_name, label, tool), combining all 'data_gt'.
    """
    grouped = defaultdict(list)

    for item in extracted :
        key = (item['pdf_name'], item['label'], item['tool'])
        grouped[key].append(item['data_gt'])

    rows = []
    for (pdf_name, label, tool), texts in grouped.items():
        combined_text = ' '.join(map(str, texts))
        rows.append({
            "tool": tool,
            "pdf_name": pdf_name,
            "page": 0,
            "label": label,
            "data_gt": combined_text
        })
    return rows