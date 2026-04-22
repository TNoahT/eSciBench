import os
import requests
import re
import xml.etree.ElementTree as ET
import pymupdf as fitz
from ..pdf_utils import PDF
from ...normalisation import normalize_string


#docker run --rm -it -p 8070:8070 sciencebeam-parser-cv
NS = {'tei': 'http://www.tei-c.org/ns/1.0'}


label_to_tag = {
    # ==== Metadata (from <teiHeader>) ====
    'title': './/tei:teiHeader//tei:titleStmt/tei:title',
    'abstract': './/tei:teiHeader//tei:profileDesc//tei:abstract//tei:p',
    'author': './/tei:teiHeader//tei:author/tei:persName',
    'email': './/tei:teiHeader//tei:email',
    'affiliation': ".//tei:affiliation/tei:note[@type='raw_affiliation']",
    'publisher': './/tei:teiHeader//tei:publisher',
    'journal': './/tei:teiHeader//tei:monogr//tei:title',
    'doi': './/tei:teiHeader//tei:idno[@type="DOI"]',
    'pub_date': './/tei:teiHeader//tei:note[@type="<date>"]',
    'keyword':     './/tei:teiHeader//tei:note[@type="<keyword>"]',

    # ==== Body content (from <text><body>) ====
    'section': './/tei:text//tei:body//tei:head',
    'paragraph': './/tei:text//tei:body//tei:p',
    'footer' : './/tei:text//tei:body//tei:note//tei:p',
    'equation' : './/tei:text//tei:body//tei:formula',
    'figure' : './/tei:text//tei:body//tei:figure',

    # ==== Bibliography content (from <text><back><biblStruct>) ====
    'reference': './/tei:text/tei:back/tei:div[@type="references"]/tei:listBibl/tei:biblStruct/tei:note[@type="raw_reference"]',
    'citation_title': './/tei:text//tei:biblStruct/tei:analytic/tei:title[@type="main"]',
    'citation_author': './/tei:text//tei:biblStruct/tei:analytic/tei:author/tei:persName',
    'citation_year': './/tei:text//tei:biblStruct/tei:monogr/tei:imprint/tei:date',
    'citation_container_title': './/tei:text//tei:biblStruct/tei:monogr/tei:title',                      # i.e. The whole book if the citation is from a chapter
    'citation_publisher': './/tei:text//tei:biblStruct/tei:monogr/tei:imprint/tei:publisher',
    'citation_pub_place': './/tei:text//tei:biblStruct/tei:monogr/tei:imprint/tei:pubPlace',
    'citation_volume': './/tei:text//tei:biblStruct/tei:monogr/tei:imprint/tei:biblScope[@unit="volume"]',
    'citation_issue': './/tei:text//tei:biblStruct/tei:monogr/tei:imprint/tei:biblScope[@unit="issue"]',
    'citation_fpage': './/tei:text//tei:biblStruct//tei:biblScope[@unit="page"]',
    'citation_lpage': './/tei:text//tei:biblStruct//tei:biblScope[@unit="page"]',
    'reference_label': './/tei:text//tei:ref'
}
"""
Mapping of (almost) all labels to xml tags possible with Science Beam
"""


def extract_sciencebeam(PDFList: list[PDF]):
    """
    Run ScienceBeam's full-text processing on PDFs in the given PDF list.

    Generates the .sciencebeam.tei.xml files in the same directory as the
    PDF files.

    Args:
        PDFList (list[PDF]): List of PDF files to be processed
    """
    url = "http://localhost:8070/api/processFulltextDocument"
    for pdf in PDFList:
        pdf_path = pdf.filepath

        try :
            # Get the number of pages
            doc = fitz.open(pdf_path)
            num_pages = doc.page_count # from 1 to x

            params = {
                "first_page": 1,
                "last_page": num_pages,
                "output_format": "raw_data"
            }

            # Send the API request
            with open(pdf_path, "rb") as f:
                response = requests.post(
                    url,
                    files={"file": (pdf_path, f, "application/pdf")},
                    params=params
                )
            response.raise_for_status()
            print(pdf.filepath)
            xml_name = pdf.filepath.replace('.pdf', '.sciencebeam.tei.xml')
            with open(xml_name, "wb") as out_file:
                out_file.write(response.content)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 500:
                print(f"[ERROR] ScienceBeam failed (500) on {pdf.pdf_name}")
                return False, []


def extract_sciencebeam_if_needed(PDFList: list[PDF], base_dir: str) :
    """
    Only run ScienceBeam extraction if there are PDF files that have not yet 
    been processed.

    Checks for existing `.sciencebeam.tei.xml` files. If one PDF file doesn't 
    have a corresponding `.sciencebeam.tei.xml` file, all the files will be
    re-extracted. This is to have stable time metrics.

    Args:
        base_dir (str): Directory containing PDF files and expected output XMLs.
    """

    exclude_pdf_names = {"sigma_term_lattice_v2"} # Sciencebeam fails to extract this file

    already_done = set(
        f.replace('.sciencebeam.tei.xml', '') for f in os.listdir(base_dir) if f.endswith('.sciencebeam.tei.xml')
    )
    all_pdfs = set(
        os.path.splitext(f)[0] for f in os.listdir(base_dir) if f.lower().endswith('.pdf')
    )
    to_process = all_pdfs - already_done - exclude_pdf_names
    print(f"Number to process : {len(to_process)} -> {to_process}")
    if to_process :
        extract_sciencebeam(PDFList)


def extract_raw(base_dir:str, label:str, pdf:PDF) -> tuple[bool, list[tuple]]:
    """
    Main ScienceBeam extractor function for a given pdf and label.

    Handles the information extraction from the `.sciencebeam.tei.xml` file
    corresponding to the given pdf, based on `label`.

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

    filepath = pdf.filepath.replace('.pdf', '.sciencebeam.tei.xml')
    filename = pdf.pdf_name
    page_number = 0
    
    try :
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Special treatment for author names
        if label == 'author':
            for author in root.findall('.//tei:teiHeader//tei:author/tei:persName', namespaces=NS):
                # Collect all forenames (first, middle, etc.)
                forenames = []
                for f in author.findall('tei:forename', namespaces=NS):
                    if f.text:
                        text = f.text.strip()
                        """
                        # Add a period if the forename is a single letter
                        if len(text) == 1:
                            text += '.'
                        """
                        forenames.append(text)

                # Get the surname
                surname_elem = author.find('tei:surname', namespaces=NS)
                surname = surname_elem.text.strip() if surname_elem is not None and surname_elem.text else ''

                # Combine full name
                name_parts = forenames + ([surname] if surname else [])
                full_name = ' '.join(name_parts)
                if full_name:
                    results.append((filename, page_number, label, normalize_string(full_name)))
                    
            return True, results

        tag = label_to_tag[label]
        
        # Special treatment for page numbers in bibliographical references
        if label in ['citation_fpage', 'citation_lpage']:
            for bibl in root.findall('.//tei:biblStruct//tei:biblScope[@unit="page"]', NS):
                fpage = bibl.attrib.get('from')
                lpage = bibl.attrib.get('to')

                if label == 'citation_fpage' and fpage:
                    results.append((filename, page_number, label, normalize_string(fpage)))
                elif label == 'citation_lpage' and lpage:
                    results.append((filename, page_number, label, normalize_string(lpage)))
            return True, results
        
        elif label == "keyword":
            kw_notes = root.findall(tag, NS)
            for kw_note in kw_notes:
                if kw_note.text :
                    for key in kw_note.text.replace(';', ',').split(',') :
                        results.append((filename, page_number, label, normalize_string(key)))
            return True, results
        
        # Treatment of the regular tags
        for elem in root.findall(tag, namespaces=NS):
            text = re.sub(r'\s+', ' ', ''.join(elem.itertext()).strip())
            #if text == "" : continue
            if text :
                results.append((filename, page_number, label, normalize_string(text)))

    except Exception as e:
        print(f"[ERROR] Sciencebeam's tei.xml file generation failed: {e}")
        return False, results
    
    return True, results