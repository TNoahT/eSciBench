import os
import re
import json
from tqdm import tqdm
from typing import Tuple, List

from unstructured.partition.pdf import partition_pdf
from ..pdf_utils import PDF
from ...normalisation import normalize_string


UNSTRUCTURED_LABEL_TO_TYPES = {
    "title":        ["Title"],
    #"section":      ["Heading"],
    "list":         ["ListItem"],
    #"table":        ["Table"],
    #"figure":       ["Figure"],
    "caption":      ["TableCaption", "FigureCaption"],
    #"equation":     ["Equation"],
    "footer":       ["Footer"],
    "equation":     ["Formula"],
    "email":        ["EmailAdress"],
    "table":        ["Table"],
    "header":       ["Header"],
    "code":         ["CodeSnippet"]
}

REFERENCE_TITLE_RE = re.compile(r'^\s*(References|Bibliography)\s*$', re.IGNORECASE)

# Headings that signal the end of the reference section
SECTION_BREAK_RE = re.compile(
    r'^\s*(Appendix|Annex|Supplementary|Author|Acknowledgement|Acknowledgment|About the author)',
    re.IGNORECASE
)

def extract_unstructured(PDFList) -> None:
    """
    Run Unstructured's PDF partitioner on all PDFs in the directory.
    Generates `.unstructured.json` files for each PDF.
    """
    for pdf in tqdm(PDFList):
        
        pdf_name = pdf.pdf_name
        filepath = pdf.filepath
        raw_out = pdf.filepath.replace(".pdf", ".unstructured.json")
        try:
            elements = partition_pdf(
                filename=filepath,
                skip_infer_table_types=False,
                strategy='hi_res'
            )
            with open(raw_out, 'w', encoding='utf-8') as f:
                json.dump([el.to_dict() for el in elements], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] Unstructured failed on {pdf_name}: {e}")


def extraction_if_needed(base_dir: str, PDFList) -> None:
    """
    Only run Unstructured extraction if there are PDFs without a corresponding `.unstructured.json`.
    """
    already_done = set(
        f.replace('.unstructured.json', '') for f in os.listdir(base_dir) if f.endswith('.unstructured.json')
    )
    all_pdfs = set(
        os.path.splitext(f)[0] for f in os.listdir(base_dir) if f.lower().endswith('.pdf')
    )
    to_process = all_pdfs - already_done
    print(f"To process : {len(to_process)} -> {to_process} ")
    if to_process:
        extract_unstructured(PDFList)


def classify_list_items(elements: list[dict]) -> tuple[set[str], set[str]]:
    """
    Walk the element list in document order and classify every ListItem
    element_id as either a reference or a regular list item.

    Strategy:
    - Enter "reference mode" when a Title matching References/Bibliography is seen.
    - Exit "reference mode" when a Title matching a known section-break pattern
      is seen (Appendix, Annex, Author info, etc.).
    - While in reference mode, every ListItem (regardless of parent_id) is a
      reference. This handles page-break interruptions where Unstructured drops
      the parent_id on continuation items.

    Returns:
        (reference_ids, list_ids): two sets of element_ids.
    """
    reference_ids: set[str] = set()
    list_ids: set[str] = set()
    in_ref_section = False

    for el in elements:
        el_type = el.get("type", "")
        el_id = el.get("element_id", "")
        text = el.get("text", "")

        if el_type == "Title":
            if REFERENCE_TITLE_RE.match(text):
                in_ref_section = True
            elif in_ref_section and SECTION_BREAK_RE.match(text):
                in_ref_section = False

        elif el_type == "ListItem":
            if in_ref_section:
                reference_ids.add(el_id)
            else:
                list_ids.add(el_id)

    return reference_ids, list_ids


def reference_parent_ids(elements: list[dict]) -> set[str]:
    """
    Return the set of element_ids of Title elements whose text is
    'References' or 'Bibliography'. ListItems that have one of these
    as their parent_id are bibliographic references, not regular list items.
    """
    return {
        el["element_id"]
        for el in elements
        if el.get("type") == "Title" and REFERENCE_TITLE_RE.match(el.get("text", ""))
    }


def extract_raw(base_dir: str, label: str, pdf: PDF) -> Tuple[bool, List[Tuple[str, int, str, str]]]:
    """
    Pull out all elements matching `label` from a pre-generated `.unstructured.json`.
    Returns (success, list_of_(pdf_name, page, label, text)).
    """
    results: List[Tuple[str, int, str, str]] = []
    raw_path = os.path.join(base_dir, f"{os.path.splitext(pdf.pdf_name)[0]}.unstructured.json")

    if not os.path.exists(raw_path):
        extraction_if_needed(base_dir)

    try:
        with open(raw_path, 'r', encoding='utf-8') as f:
            elements = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load unstructured JSON for {pdf.pdf_name}: {e}")
        return False, results

    # Classify all ListItems upfront in a single pass
    ref_ids, list_ids = classify_list_items(elements)

    if label in ("title", "section"):
        titles = [el for el in elements if el.get("type") == "Title"]
        candidates = titles[:1] if label == "title" else titles[1:]

    elif label == "list":
        candidates = [el for el in elements if el.get("element_id") in list_ids]

    elif label == "reference":
        candidates = [el for el in elements if el.get("element_id") in ref_ids]

    else:
        target_types = UNSTRUCTURED_LABEL_TO_TYPES.get(label, [])
        candidates = [el for el in elements if el.get("type") in target_types]

    for el in candidates:
        text = normalize_string(el.get("text", ""))
        if text:
            results.append((pdf.pdf_name, 0, label, text))

    return True, results