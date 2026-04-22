import pdfplumber
import pandas as pd
from ..pdf_utils import PDF
from typing import Tuple, List
from ...normalisation import normalize_string


def df_to_clean_string(df: pd.DataFrame, sep=" ") -> str:
    lines = []

    header = [col.strip() for col in df.columns if col is not None]
    lines.append(sep.join(header))

    for _, row in df.iterrows():
        if row is not None :
            cells = [str(cell).replace("\n", " ").strip() for cell in row]
            lines.append(sep.join(cells))

    return sep.join(lines)


def extract_pdfplumber(dir: str, label: str, pdf: PDF) -> Tuple[bool, List[Tuple[str, int, str, str]]]:
    """
    Assumes that a table has at least a header and one data row.
    """
    results = []
    file = pdf.filepath
    if label != "table":
            print(f"[ERROR] Unsupported label '{label}' for PyPDF_tables")
            return False, results
    
    settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 5,
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 3,
        "min_words_vertical": 1,
        "min_words_horizontal": 1,
    }

    try:
        with pdfplumber.open(file) as doc:
            for _, page in enumerate(doc.pages):
                tbls = page.extract_tables(table_settings=settings)
                for tbl in tbls:

                    if not tbl or len(tbl) < 2:
                        continue

                    header = tbl[0]
                    data = tbl[1:]

                    # Split each multiline cell into a list
                    split_columns = []
                    clean_rows = []
                    for row in data:
                        clean_row = [
                            " ".join(cell.split()) if isinstance(cell, str) else "" 
                            for cell in row
                        ]
                        clean_rows.append(clean_row)

                    # Transpose the split columns (list of columns → list of rows)
                    rows = list(zip(*split_columns))

                    # Create a new DataFrame
                    df = pd.DataFrame(clean_rows, columns=header)
                    if not df.empty :
                        df_str = df_to_clean_string(df)

                    if df_str:
                        results.append((pdf.pdf_name, 0, label, normalize_string(df_str)))
        return True, results

    except Exception as e:
        print(f"[ERROR] pdfplumber failed for {pdf.filepath}: {e}")
        return False, results