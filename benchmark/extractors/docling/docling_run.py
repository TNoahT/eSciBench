import os
import re
import json
from tqdm import tqdm
from typing import Tuple, List

# pip install docling
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from docling_core.types.doc import DoclingDocument

from ..pdf_utils import PDF
from ...normalisation import normalize_string

def extract_docling(base_dir: str, PDFList: list[PDF]) -> None:
    """
    Run Docling's full-text processing on PDFs.
    Generates `.docling.json` files containing the structured DoclingDocument.
    """
    
    converter = DocumentConverter()
    print(converter.pipeline_options)
    print(converter.pipeline_options.ocr_options)
    print("OCR enabled:", converter.pipeline_options.do_ocr)

    for pdf in tqdm(PDFList, desc="Docling extracting"):
        output_path = pdf.filepath.replace(".pdf", ".docling.json")
        
        # Skip if already exists (handled by extraction_if_needed, but safe to have here)
        if os.path.exists(output_path):
            continue

        try:
            result = converter.convert(pdf.filepath)
            # We save as JSON to preserve the semantic labels (Heading, Table, etc.)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.document.export_to_dict(), f, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Docling failed for {pdf.pdf_name}: {e}")


def extraction_if_needed(base_dir: str, PDFList: list[PDF]) -> None:
    """
    Checks for existing .docling.json files before running extraction.
    """
    all_pdfs = [f for f in os.listdir(base_dir) if f.lower().endswith('.pdf')]
    to_process = []
    
    for pdf_file in all_pdfs:
        json_file = pdf_file.replace(".pdf", ".docling.json")
        if not os.path.exists(os.path.join(base_dir, json_file)):
            # Find the corresponding PDF object from the list
            pdf_obj = next((p for p in PDFList if p.pdf_name == pdf_file), None)
            if pdf_obj:
                to_process.append(pdf_obj)

    if to_process:
        print(f"Docling processing {len(to_process)} files...")
        extract_docling(base_dir, to_process)
        

def flatten_docling_table(table_obj: dict) -> str:
    cells = table_obj.get("data", {}).get("table_cells", [])
    if not cells:
        return ""

    # Determine table size
    max_row = max(cell["end_row_offset_idx"] for cell in cells)
    max_col = max(cell["end_col_offset_idx"] for cell in cells)

    # Initialize grid
    grid = [["" for _ in range(max_col)] for _ in range(max_row)]

    # Fill grid
    for cell in cells:
        r = cell["start_row_offset_idx"]
        c = cell["start_col_offset_idx"]
        text = cell.get("text", "").strip()

        grid[r][c] = text

    # Flatten rows
    rows_as_strings = []
    for row in grid:
        cleaned_row = " ".join(x for x in row if x)
        if cleaned_row:
            rows_as_strings.append(cleaned_row)

    return " ".join(rows_as_strings)


def extract_raw(base_dir: str, label: str, pdf: PDF):
    results = []
    json_path = pdf.filepath.replace(".pdf", ".docling.json")

    if not os.path.exists(json_path):
        return False, results

    with open(json_path, "r", encoding="utf-8") as f:
        doc_dict = json.load(f)

    texts = doc_dict.get("texts", [])
    reference_start_index = None

    # Need to find where the references start, since Docling extracts them as
    # list items
    if label in ["reference", "list"]:
        for i, item in enumerate(texts):
            node_label = item.get("label", "").lower()
            text = item.get("text", "").strip().lower()

            if node_label in ["section_header", "heading"]:
                if re.search(r"\b(reference|bibliograph)", text):
                    reference_start_index = i
                    break

    for item in texts:
        node_label = item.get("label", "").lower()
        text = item.get("text", "")
        orig = item.get("orig", "")
        page_no = item.get("prov", [{}])[0].get("page_no", 1)

        # --- SECTION ---
        if label == "section" and node_label in ["section_header", "heading"]:
            text = re.sub(r'^\s*(?:\d+(?:\.\d+)*|[A-Z]|[ivxlcdm]+)[\.\)]?\s+', '', text) # Remove section numbering
            results.append((pdf.pdf_name, page_no, "section", normalize_string(text)))

        # --- TITLE ---
        elif label == "title" and node_label == "section_header" and page_no == 1:
            # first section_header usually title
            results.append((pdf.pdf_name, page_no, "title", normalize_string(text)))
            return True, results # Only return one item

        # --- LIST ---
        elif label == "list":
            for i, item in enumerate(texts):
                if reference_start_index is not None and i >= reference_start_index:
                    break  # stop before references

                if item.get("label","").lower() == "list_item":
                    text = item.get("text","")
                    page_no = item.get("prov",[{}])[0].get("page_no",1)

                    results.append(
                        (pdf.pdf_name, page_no, "list", normalize_string(text))
                    )

            return True, results

        # --- EQUATION ---
        elif label == "equation" and node_label in ["formula", "equation"]:
            results.append((pdf.pdf_name, page_no, "equation", normalize_string(orig)))

        # --- CAPTION ---
        elif label == "caption" and node_label == "caption":
            text = re.sub(r'^(figure|table)\s*\d+\s*:\s*', '', text, flags=re.IGNORECASE)
            results.append((pdf.pdf_name, page_no, "caption", normalize_string(text)))

        # --- REFERENCE --- TODO
        elif label == "reference":
            if reference_start_index is None:
                return True, results

            for item in texts[reference_start_index:]:
                if item.get("label","").lower() == "list_item":
                    text = item.get("text","")
                    page_no = item.get("prov",[{}])[0].get("page_no",1)

                    results.append(
                        (pdf.pdf_name, page_no, "reference", normalize_string(text))
                    )

            return True, results

        # --- HEADER / FOOTER ---
        elif label == "header" and node_label == "page_header":
            results.append((pdf.pdf_name, page_no, "header", normalize_string(text)))

        elif label == "footer" and node_label in ["footnote", "page_footer"]:
            results.append((pdf.pdf_name, page_no, "footer", normalize_string(text)))

    # --- TABLE ---
    if label == "table":
        tables = doc_dict.get("tables", [])

        for tbl in tables:
            page_no = tbl.get("prov", [{}])[0].get("page_no", 1)

            flattened = flatten_docling_table(tbl)

            results.append(
                (pdf.pdf_name, page_no, "table", normalize_string(flattened))
            )

        return True, results
            
    return True, results