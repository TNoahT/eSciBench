import os
import pymupdf as fitz
from tqdm import tqdm
from ..pdf_utils import PDF
from ...normalisation import normalize_string

def extract_raw(base_dir:str, label:str, pdf:PDF) -> tuple[bool, list[tuple]]:
    """
    PyMuPDF extractor for a given PDF.

    PyMuPDF extracts the whole text from a document, page per page. So, only 
    the `paragraph` label is supported.

    Args:
        base_dir (str): Path to the directory containing the PDF files.
        label (str): Label to extract.
        pdf (PDF): PDF object with metadata.

    Returns:
        tuple:
            - bool: True if extraction succeeded, False on error.
            - list[tuple]: Extracted entries as (filename, page_number, label, value).
    """
    results = []

    file = pdf.filepath
    try:
        doc = fitz.open(file)
        if label == 'paragraph':
            for i, page in enumerate(doc):

                text = page.get_text("text").replace('\n', ' ').strip()
                # results = [(name, page_number, label, extraction)]
                results.append((pdf.pdf_name, 0, label, normalize_string(text)))

        elif label == 'table':
            for i, page in enumerate(doc):
                # detect tables on this page
                tables = page.find_tables()
                # tables is a TableFinder; you can index into it or iterate directly
                for tbl in tables:
                    # extract() returns a list of rows, each row is a list of cell-strings
                    table_data = tbl.extract()
                    flat = " ".join(
                        " ".join(cell.replace("\n", " ") for cell in row)
                        for row in table_data
                    )
                    flat = normalize_string(flat)
                    #print(flat)
                    if flat == "" : continue
                    results.append((pdf.pdf_name, 0, label, flat))

        else:
            print(f"[ERROR] Unsupported label '{label}' for PyMuPDF")
            return False, results

    except Exception as e:
        print(f"[ERROR] PyMuPDF failed for {file}: {e}")
        return False, results

    return True, results