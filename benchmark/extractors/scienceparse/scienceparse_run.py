import os
from tqdm import tqdm
import re
from pathlib import Path
from science_parse_api.api import parse_pdf
from ..pdf_utils import PDF
from ...normalisation import normalize_string

# docker run -p 8080:8080 ucrel/ucrel-science-parse:3.0.1
host = 'http://127.0.0.1'
port = '8080'

label_to_key = {
    'abstract': 'abstractText',
    'author': 'authors',
    'affiliation': 'authors', #authors.affiliation
    'id': 'id',
    'reference': 'references',
    'section': 'sections',
    'title': 'title',
    'pub_date': 'year'
}
"""
Mapping of labels to extractable tags for ScienceParse
"""


def extraction_if_needed(base_dir: str, PDFList: list[PDF]) -> None:
    """
    Only run ScienceParse extraction if there are PDF files that have not yet been processed.

    Checks for existing `.scienceparse.json` files. If one PDF file doesn't have a
    corresponding `.scienceparse.json` file, all the files have to be re-extracted. 
    This is to have stable time metrics.

    Args:
        base_dir (str): Directory containing PDF files and expected output JSONs.
        PDFList (list[PDF]): List of PDF objects with metadata.
    """
    all_outputs_exist = True
    for pdf in PDFList:
        output_path = os.path.join(base_dir, f"{Path(pdf.pdf_name).stem}.scienceparse.json")
        if not os.path.exists(output_path):
            all_outputs_exist = False
            break

    if not all_outputs_exist:
        print("[INFO] Running ScienceParse extraction...")
        for pdf in tqdm(PDFList, desc="ScienceParse"):
            try:
                parse_pdf(host, Path(pdf.filepath), port)
            except Exception as e:
                print(f"[ERROR] ScienceParse failed for {pdf.pdf_name}: {e}")


def format_reference(entry: dict, index: int =None) -> str:
    """
    Format a single reference entry into a human-readable and ground 
    truth-comparable format.

    Args:
        entry (dict): A reference entry dictionary, expected to contain the keys:
            - 'authors' (list of str): List of author names.
            - 'title' (str): Title of the work.
            - 'venue' (str): Publication venue (e.g., journal or conference). May be empty.
            - 'year' (str or int): Publication year. May be empty.
        index (int, optional): Sequential reference number to prefix (e.g., 1 for "[1]"). 
            If provided, the formatted string will be prefixed with "[index] ". Defaults to None.

    Returns:
        str: A formatted reference string consisting of:
            "<authors> ([<year>.]). <title>. [<venue>.]"
            Optionally prefixed with "[index] ".
    """

    authors = ', '.join(entry.get('authors', []))
    title = entry.get('title', '')
    venue = entry.get('venue', '')
    year = entry.get('year', '')

    ref = ""
    if authors :
        ref += f"{authors}"
    elif year:
        ref += f" ({year})"
    ref += "."
    if title:
        ref += f" {title}."
    if venue:
        ref += f" {venue}."
    if index is not None:
        ref = f"[{index}] " + ref
    return ref.strip()


def clean_section_heading(heading: str) -> str:
    # Strip leading numbers/letters/roman numerals with or without punctuation
    return re.sub(
        r'^\s*(?:\d+(?:\.\d+)*|[A-Z]|[ivxlcdm]+)[\.\)]?\s+',
        '',
        heading,
        flags=re.IGNORECASE
    )


def extract_raw(base_dir:str, label:str, pdf: PDF) -> tuple[bool, list[tuple]] :
    """
    Main ScienceParse extraction function for a given PDF and label.

    All extractable labels are in `label_to_tag`.

    Args:
        base_dir (str): Path to directory containing the PDF files (unused).
        label (str): The target label to extract (e.g., 'author', 'title').
        pdf (PDF): PDF metadata object containing filename and path.

    Returns:
        tuple:
            - bool: Whether extraction was successful.
            - list[tuple]: List of (pdf_name, page_number, label, extracted_text) entries.
    """
    results = []
    page_number = 0
    file = pdf.filepath
    json_output_path = os.path.join(base_dir, f"{Path(pdf.pdf_name).stem}.scienceparse.json")

    try:
        if os.path.exists(json_output_path):
            with open(json_output_path, "r", encoding="utf-8") as f:
                import json
                extraction = json.load(f)
        else:
            extraction = parse_pdf(host, Path(file), port)
            if extraction is None:
                print(f"[WARNING] ScienceParse returned None for {file}")
                return False, results
            with open(json_output_path, "w", encoding="utf-8") as f:
                import json
                json.dump(extraction, f, indent=2)

        if extraction is None:
            print(f"[WARNING] Extraction is None for {file}")
            return False, results

        key = label_to_key.get(label)
        if not key or key not in extraction:
            print(f"[ERROR] Unsupported label '{label}' or missing key for ScienceParse")
            return False, results

        # === Special Cases ===
        if label == "reference":
            for ref in extraction[key]:
                formatted = format_reference(ref)
                results.append((pdf.pdf_name, page_number, label, normalize_string(formatted)))

        elif label == "author":
            for author in extraction[key]:
                name = author.get("name", "").strip()
                if name:
                    results.append((pdf.pdf_name, page_number, label, normalize_string(name)))

        elif label == "affiliation":
            for author in extraction[key]:
                aff = author.get("affiliation", "").strip()
                if aff:
                    results.append((pdf.pdf_name, page_number, label, normalize_string(aff)))

        elif label == "section":
            for section in extraction[key]:
                heading = section.get("heading", "").strip()
                if heading:
                    cleaned = clean_section_heading(heading)
                    results.append((pdf.pdf_name, page_number, label, normalize_string(cleaned)))

        else:
            value = extraction.get(key)
            if isinstance(value, list):
                for v in value:
                    norm = normalize_string(v)
                    if norm:
                        results.append((pdf.pdf_name, page_number, label, norm))
            else:
                norm = normalize_string(str(value))
                results.append((pdf.pdf_name, page_number, label, norm))

    except Exception as e:
        print(f"[ERROR] ScienceParse failed for {file}: {e}")
        return False, results

    return True, results