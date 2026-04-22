import os
import csv
import json
import pandas as pd
import pymupdf as fitz
from os import path
from glob import glob


class PDF:
    def __init__(self, pdf_name=None, filepath=None, txt_pages=None):
        self.pdf_name = pdf_name
        self.filepath = filepath
        self.txt_pages = txt_pages or {}  # key: JSON key, value: str data

def load_all_data_Docbank(dir:str, labels:list[str]) -> list[PDF] :
    """
    Loads all PDF files from the specified directory and returns them as a list of PDF objects.
    Groups all DocBank's page-level annotations under the corresponding PDF object.

    Args:
        dir (str): Path to the directory containing PDF files.
        labels (list): Unused in this function, but passed for compatibility with caller signature.

    Returns:
        list[PDF]: List of PDF objects with filenames and paths populated.
    """
    PDFMap = {}
    txt_files = glob(path.join(dir, "*.txt"))

    for txt in txt_files:
        base = path.splitext(path.basename(txt))[0]  # e.g., 2.tar_1801.00617.gz_idempotents_arxiv_4
        parts = base.split("_")
        if len(parts) < 2:
            continue
        page_number = int(parts[-1])
        keyword = "_".join(parts[:-1])
        pdf_name = f"{keyword}_black.pdf"
        pdf_path = path.join(dir, pdf_name)

        if not path.isfile(pdf_path):
            continue

        # Read txt file into DataFrame with column handling
        try:
            df_raw = pd.read_csv(txt, sep="\t", quoting=csv.QUOTE_NONE, encoding='latin1', header=None)
            num_cols = df_raw.shape[1]

            # Validate presence of required columns
            if num_cols < 5:
                print(f"[ERROR] Not enough columns in {txt}")
                continue

            # Rename columns based on how many are present
            all_names = ["token", "x0", "y0", "x1", "y1", "unk1", "unk2", "unk3", "font", "label"]
            df_raw.columns = all_names[:num_cols]

            # Only keep relevant columns if available
            wanted_cols = ["token", "x0", "y0", "x1", "y1"]
            if "font" in df_raw.columns and "label" in df_raw.columns:
                wanted_cols += ["font", "label"]

            txt_df = df_raw[wanted_cols]
        except Exception as e:
            print(f"[ERROR] Failed to load {txt}: {e}")
            continue

        if pdf_name not in PDFMap:
            PDFMap[pdf_name] = PDF(pdf_name=pdf_name, filepath=pdf_path, txt_pages={})

        PDFMap[pdf_name].txt_pages[page_number] = txt_df

    return list(PDFMap.values())

def load_data_per_page(dir:str, labels:list[str]) ->list[PDF] :

    PDFList = load_all_data(dir, labels)
    cropped_pdfs = []

    for pdf in PDFList:
        if not pdf.txt_pages : continue
        
        doc = fitz.open(pdf.filepath)

        for page_num in sorted(pdf.txt_pages):
            try:        
                cropped_doc = fitz.open() # New empty PDF
                cropped_doc.insert_pdf(doc, from_page = page_num, to_page = page_num)

                output_filename = f"{os.path.splitext(pdf.pdf_name)[0]}_page{page_num}.pdf"
                output_path = os.path.join(dir, output_filename)
                cropped_doc.save(output_path)
                cropped_doc.close()

                cropped_pdf = PDF(
                    pdf_name = output_filename,
                    filepath = output_path,
                    txt_pages= {page_num: pdf.txt_pages[page_num]}
                )
                cropped_pdfs.append(cropped_pdf)

            except Exception as e:
                print(f"[ERROR] Failed to copy page {page_num} from {pdf.pdf_name}: {e}")

        doc.close()
    return cropped_pdfs
    
def load_data_json(dir:str) -> list[PDF] :
    pdfs = []
    json_files = glob(path.join(dir, "*json"))

    for json_file in json_files :
        base = path.splitext(path.basename(json_file))[0] # Remove file extention
        pdf_name = f"{base}.pdf"
        pdf_path = path.join(dir, pdf_name)

        if not path.isfile(pdf_path):
            continue

        # Load the JSON content
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Skipping {json_file} : invalid JSON ({e})")
            continue

        # Create the PDF instance
        pdf = PDF(pdf_name=pdf_name, filepath=pdf_path)

        for tag, value in data.items():
            pdf.txt_pages[tag] = value

        pdfs.append(pdf)
    return pdfs