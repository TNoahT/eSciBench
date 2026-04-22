import os
import camelot
import numpy as np
from tqdm import tqdm
from ..pdf_utils import PDF
from ...normalisation import normalize_string

import logging
logging.getLogger('pdfminer').setLevel(logging.ERROR)  # suppress pdfminer warnings

def extract_raw(base_dir: str, label: str, pdf: PDF) -> tuple[bool, list[tuple]]:
    """
    Main Camelot extractor for a given PDF and label.

    Extracts `table` contents from the PDF (since Camelot only supports
    table extraction).

    Args:
        base_dir (str): Path to the directory containing the PDF files (unused).
        label (str): Label to extract (e.g., title, abstract, reference).
        pdf (PDF): PDF object with metadata.

    Returns:
        tuple:
            - bool: True if extraction succeeded, False on error.
            - list[tuple]: Extracted entries as (filename, page_number, label, value).
    """
    results = []

    file = pdf.filepath
    if label != "table":
        print(f"[ERROR] Unsupported label '{label}' for Camelot")
        return False, results

    try:
        #tables = camelot.read_pdf(file)
        tables = camelot.read_pdf(file, pages='all', flavor='network')
        if not tables:
            return False, results  # No tables found, but label is valid

        for table in tables:
            flat_table = " ".join([
                " ".join(cell.replace("\n", " ").strip() for cell in row)
                for row in table.df.astype(str).values.tolist()
            ])
            flat_table = normalize_string(flat_table)
            results.append((pdf.pdf_name, 0, label, flat_table))

    except Exception as e:
        print(f"[ERROR] Camelot failed for {file}: {e}")
        return False, results

    return True, results
