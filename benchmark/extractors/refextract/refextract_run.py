import os
import json
import pandas as pd
import requests
import re
from tqdm import tqdm
from collections import OrderedDict
from .refextract.refextract.references.api import extract_references_from_file

from ..pdf_utils import PDF
from ...normalisation import normalize_string

def call_refextract_api(file_path):
    url = "http://localhost:5000/references"
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"[ERROR] Refextract API failed: {response.status_code}, {response.text}")
        return []

def extract_raw(base_dir:str, label:str, pdf:PDF) -> tuple[bool, list[tuple]] :
    """
    Refextract extractor function for a given PDF.

    Only reference extraction is supported for Refextract.

    Since the ground truth does not contain number for the references 
    ([1], [132], ...), and since refextract extracts these numbers, they are
    removed with a regular expression to better match the ground-truth.

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
    seen = OrderedDict()

    file = pdf.filepath

    json_output_path = os.path.join(
        base_dir,
        f"{os.path.splitext(pdf.pdf_name)[0]}_refextract.json"
    )

    try:
        if label == "reference":
            extracted_text = extract_references_from_file(file)

            ##############################
            try:
                with open(json_output_path, "w", encoding="utf-8") as f:
                    json.dump(extracted_text, f, ensure_ascii=False, indent=2)
                print(f"[INFO] Saved Refextract output to {json_output_path}")
            except Exception as e:
                print(f"[WARNING] Could not save JSON output: {e}")
            ########################

            for ref in extracted_text:
                raw_ref = ref.get("raw_ref", "")
                if isinstance(raw_ref, list):
                    raw_ref_text = " ".join(raw_ref)
                else:
                    raw_ref_text = str(raw_ref)

                # **************************************************************    
                # strip any leading “[1] ”, “[23] ”, etc.
                raw_ref_text = re.sub(r'^\s*\[\d+\]\s*', '', raw_ref_text)
                # **************************************************************

                if raw_ref_text not in seen:
                    seen[raw_ref_text] = raw_ref_text

            page_number = 0
            
            for ref in seen.values():
                # results = [(name, page_number, label, extraction)]
                results.append((
                    pdf.pdf_name,
                    page_number,  # Page unknown at this point
                    label,
                    normalize_string(ref)
                ))
        else:
            print(f"[ERROR] Unsupported label '{label}' for Refextract")
            return False, results

    except Exception as e:
        print(f"[ERROR] Refextract failed for {file}: {e}")
        return False, results
    
    return True, results