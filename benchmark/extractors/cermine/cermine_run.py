import os
import re
import json
import pandas as pd
import numpy as np
import subprocess
import xml.etree.ElementTree as ET
from tqdm import tqdm
from ..pdf_utils import PDF
from ...normalisation import normalize_string

label_to_tag: dict[str,str] = {
    'title': 'article-title',
    'abstract': 'abstract',
    'author': 'contrib',
    'email': 'email',
    'section': 'sec',
    'paragraph': 'p', 
    'reference': 'ref',
    'affiliation': 'aff',
    'journal': 'journal-title',
    'pub_date': 'pub-date',
    'volume': 'volume',
    'issue': 'issue',
    'fpage': 'fpage',
    'lpage': 'lpage',
    'doi': 'article-id',
    'publisher': 'publisher-name',
    'reference_label': 'reference_label',
    'citation_title': 'article-title',
    'citation_container_title': 'source',
    'citation_volume': 'volume',
    'citation_issue': 'issue',
    'citation_fpage': 'fpage',
    'citation_lpage': 'lpage',
    'citation_year': 'year',
    'citation_author': 'string-name'
}
"""
Mapping of (almost) all labels to xml tags possible with Cermine.
"""


def run_cermine_if_needed(base_dir:str) -> None:
    """
    Only run Cermine extraction if there are PDF files that have not yet been processed.

    Checks for existing `.cermxml` files. If one PDF file doesn't have a corresponding
    `.cermxml` file, all the files have to be re-extracted (since Cermine extracts per
    folder, and not per file).

    Args:
        base_dir (str): Directory containing PDF files and expected output XMLs.
    """
    exclude_pdf_names = {"Neurodata_Without_Borders"}
    already_done = set(f.replace('.cermxml', '') for f in os.listdir(base_dir) if f.endswith('.cermxml'))
    all_pdfs = set(os.path.splitext(f)[0] for f in os.listdir(base_dir) if f.endswith('.pdf'))
    to_process = all_pdfs - already_done - exclude_pdf_names
    if to_process:
        try:
            subprocess.call([
                "java", "-cp",
                "./benchmark/extractors/cermine/cermine-impl-1.13-jar-with-dependencies.jar",
                "pl.edu.icm.cermine.ContentExtractor",
                "-path", base_dir, "-outputs", "jats"
            ])
        except Exception as e:
            print(f"[ERROR] Cermine failed to generate .cerm.xml files: {e}")


def extract_article_meta(root, filename:str, label:str, page_number:int) -> list[tuple]:
    """
    Extracts high-level metadata (title, publication date, volume, issue, fpage, lpage) from article-meta.

    Args:
        root (Element): Parsed XML root from CERMINE output.
        filename (str): Name of the corresponding PDF.
        label (str): The specific metadata label to extract.
        page_number (int): Page number to associate with results.

    Returns:
        list[tuple]: Extracted metadata entries in the form (filename, page_number, label, value).
    """
    results = []
    article_meta = root.find(".//article-meta")
    if not article_meta:
        return results

    if 'title' == label:
        for elem in article_meta.iter('article-title'):
            text = re.sub(r'\s+', ' ', ''.join(elem.itertext()).strip())
            if text:
                results.append((filename, page_number, 'title', normalize_string(text)))

    meta_labels = ['pub_date', 'volume', 'issue', 'fpage', 'lpage', "affiliation", "author"]

    if label in meta_labels :
        tag = label_to_tag.get(label)
        if not tag :
            return results

        if label == 'pub_date':
            date_elem = article_meta.find("pub-date")
            if date_elem is not None:
                year = date_elem.findtext("year", "")
                month = date_elem.findtext("month", "")
                day = date_elem.findtext("day", "")
                full_date = "-".join(part for part in [year, month, day] if part)
                if full_date:
                    results.append((filename, page_number, 'pub_date', normalize_string(full_date)))
            return results
        
        elif label == "affiliation":
            for aff in article_meta.findall(".//aff"):
                # Remove label child
                for label_elem in aff.findall("label"):
                    aff.remove(label_elem)

                # Join the remaining text content
                aff_text = re.sub(r'\s+', ' ', ''.join(aff.itertext()).strip())
                if aff_text:
                    results.append((filename, page_number, 'affiliation', normalize_string(aff_text)))
            return results
        
        elif label == "author":
            contrib_group = article_meta.find("contrib-group")
            if contrib_group is not None:
                for contrib in contrib_group.findall("contrib"):
                    if contrib.attrib.get("contrib-type") == "author":
                        name_elem = contrib.find("string-name")
                        if name_elem is not None and name_elem.text:
                            name = normalize_string(name_elem.text.strip())
                            results.append((filename, page_number, 'author', name))
            return results
        
        elem = article_meta.find(f".//{tag}")
        if elem is not None and elem.text:
            results.append((filename, page_number, label, normalize_string(elem.text.strip())))
    return results


def extract_generic_labels(root, filename:str, label:str, page_number:int) -> list[tuple]:
    """
    Extract generic content elements from the XML tree for non-metadata labels.

    Args:
        root (Element): Parsed XML root from CERMINE output.
        filename (str): Name of the corresponding PDF.
        label (str): Label to extract (e.g., "abstract", "paragraph").
        page_number (int): Page number to associate with results.

    Returns:
        list[tuple]: Extracted content entries.
    """
    results = []
    if label in ['title', 'pub_date', 'volume', 'issue', 'fpage', 'lpage', 'reference', 'affiliation', 'author'] or label.startswith('citation_'):
        return False, results
    tag = label_to_tag.get(label)
    if not tag:
        print(f"[ERROR] Unsupported label '{label}' for Cermine")
        return False, results
    
    if label == 'section':
        for sec in root.iter('sec'):
            title_elem = sec.find('title')
            if title_elem is not None:
                text = normalize_string(re.sub(r'\s+', ' ', ''.join(title_elem.itertext()).strip()))
                if text:
                    results.append((filename, page_number, label, text))
    else :
        for elem in root.iter(tag):
            text = normalize_string(re.sub(r'\s+', ' ', ''.join(elem.itertext()).strip()))
            if text:
                results.append((filename, page_number, label, normalize_string(text)))
    return True, results


def extract_reference_labels(root, filename:str, page_number:int) -> list[tuple]:
    """
    Extract reference IDs from the ref-list section and convert them into citation labels.
    Extracts the full reference strings.

    Args:
        root (Element): Parsed XML root from CERMINE output.
        filename (str): PDF filename.
        page_number (int): Page number associated with references.

    Returns:
        list[tuple]: List of citation label entries in the form (filename, page_number, 'reference_label', label_text).
    """
    results = []
    for ref in root.findall(".//ref-list/ref"):
        ref_id = ref.attrib.get('id')
        if ref_id:
            label_text = normalize_string(f"[{ref_id.replace('ref', '')}]")
            results.append((filename, page_number, 'reference_label', normalize_string(label_text)))
    return results


def extract_citations(root, filename:str, label:str, page_number:int):
    """
    Extract citation information from the CERMINE XML structure.
    Extracts specific elements out of the reference string

    Args:
        root (Element): Parsed XML root from CERMINE output.
        filename (str): PDF filename.
        label (str): Label indicating what citation field to extract (e.g., citation_title, citation_author).
        page_number (int): Page number for context.

    Returns:
        list[tuple]: Extracted citation values based on the requested label.
    """
    results = []
    for ref in root.findall(".//ref-list/ref"):
        citation = ref.find(".//mixed-citation")
        if citation is None:
            continue
        raw_text = ''.join(citation.itertext())
        raw_text = clean_reference_text(raw_text)
        
        if 'reference' == label :
            results.append((filename, page_number, 'reference', normalize_string(raw_text)))

        elif label.startswith("citation_"):
            tag = label_to_tag.get(label)
            if not tag:
                continue
            for elem in citation.findall(tag):
                if tag == 'string-name':
                    name_parts = [part.text.strip() for part in elem if part.text]
                    full_name = ' '.join(name_parts)
                    if full_name:
                        results.append((filename, page_number, label, normalize_string(full_name)))
                elif elem.text:
                    results.append((filename, page_number, label, normalize_string(elem.text.strip())))
    return results


def clean_reference_text(text:str) -> str:
    """
    Clean and normalize a citation string extracted from XML.

    Args:
        text (str): Raw reference text.

    Returns:
        str: Cleaned reference with corrected spacing and punctuation.
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Fix spacing before/after punctuation
    text = re.sub(r'\s+([.,;:])', r'\1', text)  # remove space before punctuation
    text = re.sub(r'([(\[])\s+', r'\1', text)   # remove space after opening bracket
    text = re.sub(r'\s+([\])])', r'\1', text)   # remove space before closing bracket
    text = re.sub(r'^\s*\[\d+\]\s*', '', text)  # remove any leading "[1]", "[23]", etc.
    # Normalize dashes
    text = text.replace('{', '-').replace('}', '-')

    # Final tidy-up for specific common cases
    text = text.replace(' .', '.')
    text = text.replace(' ,', ',')
    text = text.replace(' - ', '-')
    return text


def extract_raw(base_dir:str, label:str, pdf:PDF) -> tuple[bool, list[tuple]]:
    """
    Main CERMINE extractor for a given PDF and label.

    Coordinates XML parsing and routing to appropriate extraction functions
    based on the label (metadata, body text, references).

    All extractable labels are in `label_to_tag`.

    Args:
        base_dir (str): Path to the directory containing the PDF files.
        label (str): Label to extract (e.g., title, abstract, reference).
        pdf (PDF): PDF object with metadata.

    Returns:
        tuple:
            - bool: True if extraction succeeded, False on error.
            - list[tuple]: Extracted entries as (filename, page_number, label, value).
    """
    results = []

    file = pdf.pdf_name
    xml_file = os.path.join(base_dir, file.replace(".pdf", ".cermxml"))
    
    if not os.path.exists(xml_file): return results
    
    page_number = 0

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = file.replace(".cerm.xml", ".pdf")
        
        if 'reference_label' == label:
            results.extend(extract_reference_labels(root, filename, page_number))

        elif 'reference' == label or label.startswith("citation_") :
            results.extend(extract_citations(root, filename, label, page_number))
        
        else:
            meta_labels = ['title', 'pub_date', 'volume', 'issue', 'fpage', 'lpage', "affiliation", 'author']
            
            if label in meta_labels :
                
                results.extend(extract_article_meta(root, filename, label, page_number))
            else :
                flag, extracted = extract_generic_labels(root, filename, label, page_number)
                results.extend(extracted)
                return flag, results
                

    except Exception as e:
        print(f"[ERROR] Failed to parse {xml_file}: {e}")
        return False, results

    return True, results