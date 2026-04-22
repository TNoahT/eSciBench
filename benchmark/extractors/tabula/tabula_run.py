import os
import tabula
from tqdm import tqdm
from ..pdf_utils import PDF
from ...normalisation import normalize_string

# Remove warning from PyPDF's cryptography deprication
import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)


def extract_raw(base_dir:str, label:str, pdf:PDF) -> tuple[bool, list[tuple]] :
    """
    Main Tabula extractor for a given PDF and label.

    Extracts `table` contents from the PDF (since Tabula only supports
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

    try:
        if label != 'table':
            return False, results
        tables = tabula.read_pdf(file, pages='all', multiple_tables=True, stream=True, silent=True)
        if not tables: 
            return False, results

        page_number = 0

        if label == 'table':
            for table in tables:
                # `table` is a pandas.DataFrame
                flat_table = " ".join([
                    " ".join(cell.replace("\n", " ").strip() for cell in row)
                    for row in table.astype(str).values.tolist()
                ])
                # results = [(name, page_number, label, extraction)]
                results.append((pdf.pdf_name, page_number, label, normalize_string(flat_table)))
        else:
            print(f"[ERROR] Unsupported label '{label}' for Tabula")
            return False, results

    except Exception as e:
        print(f"[ERROR] Tabula failed for {file}: {e}")
        return False, results
    
    return True, results