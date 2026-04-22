import os
import json
import subprocess
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ..pdf_utils import PDF
from ...normalisation import normalize_string

label_to_tag: dict[str,str] = {
    'abstract' : 'abstract',
    'acknowledgments': 'acknowledgement',
    'affiliation': 'affiliation',
    'appendix': 'appendix',
    'author': 'authors',
    'paragraph': 'body',
    'caption': 'caption',
    'category': 'categories',
    'figure':'figure',
    'footer': 'footnote',
    'equation': 'formula',
    'general-terms': 'general-terms',
    'section': 'heading',
    'header': 'page-header',
    'other': 'other',
    'footerr': 'footer',
    'reference': 'reference',
    'table': 'table',
    'content':'toc',
    'title': 'title'
    }
"""
Mapping of (almost) all labels to json tags possible with PDFAct.
"""


def run_pdfact_if_needed(PDFList: list[PDF], base_dir:str, labels:list[str]) -> None :
    """
    Only run PDFAct extraction if there are PDF files that have not yet been 
    processed.

    Checks for existing `.pdfact.json` files. If one PDF file doesn't have a
    corresponding `.pdfact.json` file, all the files have to be re-extracted. 
    This is to have stable time metrics.

    Args:
        base_dir (str): Directory containing PDF files and expected output XMLs.
    """
    all_outputs_exist = True
    for pdf in PDFList:
        output_path = os.path.join(base_dir, f"{Path(pdf.pdf_name).stem}.pdfact.json")
        if not os.path.exists(output_path):
            all_outputs_exist = False
            break

    # Run PDFAct if any output is missing
    lbls = "abstract,acknowledgments,affiliation,appendix,authors,body,caption,categories,figure,footnote,formula,general-terms,heading,other,page-header,footer,reference,table,toc,title"
    #lbls = "author"
    if not all_outputs_exist:
        for pdf in PDFList:
            output_path = os.path.join(base_dir, f"{Path(pdf.pdf_name).stem}.pdfact.json")
            subprocess.run([
                "java", "-jar", "./benchmark/extractors/pdfact/pdfact.jar", 
                "--include-roles", lbls,
                pdf.filepath,
                output_path,
                "--format", "json"
            ], check=True)


def extract_raw(base_dir: str, label: str, pdf: PDF) -> tuple[bool, list[tuple]]:
    """
    Main PDFAct extractor for a given PDF and label.

    Handles the information extraction from `.pdfact.json` files, 
    based on the given label.

    All extractable labels are in `label_to_tag`.

    Args :
        base_dir (str) : Path to the directory containing the PDF files.
        label (str) : Label to extract
        pdf (PDF) : PDF object with metadata

    Returns :
        tuple:
            - bool: True if extraction succeeded, False on error.
            - list[tuple]: Extracted entries as (filename, page_number, label, value).
    """
    results = []
    file_path = pdf.filepath
    output_path = os.path.join(base_dir, f"{Path(pdf.pdf_name).stem}.pdfact.json")
    
    try:
        #print(f"output_path: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f :
            data = json.load(f)

        if label not in label_to_tag.keys() :
            print(f"[ERROR] Unsupported label '{label}' for PdfAct")
            return False, results
        
        expected_role = label_to_tag[label]
        
        for para in data.get("paragraphs", []):
            block = para.get("paragraph", {})
            role = block.get("role")
            text = block.get("text", "").strip()
            page = int(block.get("positions", [{}])[0].get("page", np.nan)) - 1
            
            
            if not role or not text:
                continue
            if role == expected_role:
                
                # results = [(name, page_number, label, extraction)]
                if label == "reference" :
                    text = re.sub(r'^\s*\[\d+\]\s*', '', text)  # remove any leading "[1]", "[23]", etc.
                elif label == "section" :
                    text = re.sub(r'^\s*(?:\d+(?:\.\d+)*|[A-Z]|[ivxlcdm]+)[\.\)]?\s+', '', text) # Remove section numbering
                results.append((pdf.pdf_name, 0, label, normalize_string(text)))
            

    except Exception as e:
        print(f"[ERROR] PdfAct failed for {file_path}: {e}")
        return False, results
    
    return True, results
