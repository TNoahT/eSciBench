import os
import subprocess
import re
from tqdm import tqdm
from typing import Tuple, List
from pylatexenc.latex2text import LatexNodes2Text
from ..pdf_utils import PDF
from ...normalisation import normalize_string

 # pip install nougat-ocr
 # https://github.com/facebookresearch/nougat

def extract_nougat(base_dir:str, PDFList:list[PDF]) -> None:
    """
    Run Nougat's full-text processing on PDFs in the given directory.

    This generates `.nougat.mmd` files in the same directory, using the
    Nougat CLI.

    Args:
        base_dir (str): Directory containing PDF files to be processed.
    """

    for pdf in tqdm(PDFList):
        pdf_path = pdf.filepath

        try:
            subprocess.run(
                [
                    "nougat",
                    pdf_path,
                    "-o", base_dir,
                    "--no-skipping"
                ],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Nougat failed for {pdf.pdf_name}: {e}")



def extraction_if_needed(base_dir:str, PDFList:list[PDF]) -> None:
    """
    Only run Nougat extraction if there are PDF files that have not yet been processed.

    Checks for existing `.nougat.mmd` files. If one PDF file doesn't have a corresponding
    `.nougat.mmd` file, all the files have to be re-extracted (since Nougat extracts per
    folder, and not per file).

    Args:
        base_dir (str): Directory containing PDF files and expected output XMLs.
    """

    already_done = set(
        f.replace('.mmd', '') for f in os.listdir(base_dir) if f.endswith('.mmd')
    )
    all_pdfs = set(
        os.path.splitext(f)[0] for f in os.listdir(base_dir) if f.lower().endswith('.pdf')
    )
    to_process = all_pdfs - already_done
    if to_process :
        print(f"To process: {len(to_process)}, {to_process}")
        extract_nougat(base_dir, PDFList)


def extract_environment_blocks(latex: str, env: str) -> list[str]:
    """
    Extracts content between \begin{env}...\end{env} or \begin{env*}...\end{env*},
    non-greedy, with optional [args].
    """
    # re.escape(env) so that any literal * in env is treated as “\*”
    # \*? on both begin and end to allow the starred versions
    pattern = re.compile(
        rf'\\begin\{{{re.escape(env)}\*?\}}'      # \begin{env} or \begin{env*}
        r'(?:\[[^\]]*\])?'                       # optional [..]
        r'(.*?)'                                 # capture everything (non-greedy)
        rf'\\end\{{{re.escape(env)}\*?\}}',      # \end{env} or \end{env*}
        re.DOTALL
    )
    return [m.group(1).strip() for m in pattern.finditer(latex)]


def clean_latex_tabular(latex_tabular: str) -> str:
    """
    Converts LaTeX tabular contents into a plain-text representation.
    Removes LaTeX math and formatting, flattens rows.
    """
    txt = latex_tabular.strip()

    # Unwrap markdown bold
    txt = re.sub(r'\*\*(.*?)\*\*', r'\1', txt)
    
    leading_spec_re = re.compile(
        r'^(?:'                                    # from start of string
          r'\|?\s*'                                # optional leading pipe + spaces
          r'(?:'                                   # one token:
             r'[lcr](?=[\s|])|'                    #   single letter l/c/r only if next is space or pipe
             r'[pmbx]\d+(?:\.\d+)?pt(?=[\s|])|'    #   width spec: p###.##pt, m###pt, etc., if next is space or pipe
             r'\d+(?=[\s|])'                       #   or standalone digits if next is space or pipe
          r')\s*'                                  # then optional spaces
          r'\|?\s*'                                # optional trailing pipe + spaces
        r')+',                                    # repeat as long as it keeps matching
        flags=re.MULTILINE
    )

    # Drop any leading specs
    txt = leading_spec_re.sub('', txt)

    # Remove math wrappers: \( ... \) and $...$
    latex_tabular = re.sub(r'\\\(|\\\)', '', txt)
    latex_tabular = re.sub(r'\$([^$]+)\$', r'\1', latex_tabular)

    # Replace LaTeX commands like \text{...}, \delta, etc.
    latex_tabular = re.sub(r'\\text\{([^}]*)\}', r'\1', latex_tabular)
    latex_tabular = re.sub(r'\\[a-zA-Z]+\s*', '', latex_tabular)  # strip other commands

    # Remove curly braces around simple tokens
    latex_tabular = re.sub(r'\{([^}]*)\}', r'\1', latex_tabular)

    # Replace LaTeX row/column separators
    latex_tabular = latex_tabular.replace(r'\\hline', '')
    latex_tabular = latex_tabular.replace(r'\\\\', '\n')  # row separator
    latex_tabular = latex_tabular.replace('&', ' ')  # column separator

    # Normalize whitespace
    lines = latex_tabular.strip().split('\n')
    cleaned_txt = ' '.join(lines)

    cleaned_txt = re.sub(r',(?=\S)', ', ', cleaned_txt)
    return ' '.join(cleaned_txt.split())


def extract_tables(latex: str) -> list[str]:
    tables = extract_environment_blocks(latex, 'table')
    sideways_tables = extract_environment_blocks(latex, 'sidewaystable')
    all_tables = tables + sideways_tables

    tabular_blocks = []
    for tbl in all_tables:
        raw_tabulars = extract_environment_blocks(tbl, 'tabular')
        raw_tabulars += extract_environment_blocks(tbl, 'tabularx')

        # Clean each extracted tabular block
        for raw in raw_tabulars:
            cleaned = clean_latex_tabular(raw)
            tabular_blocks.append(cleaned)

    return tabular_blocks


def split_into_sections(md_text: str) -> list[tuple[str, str]]:
    """
    Split a markdown document into (heading, body) pairs.
    The heading is the full heading line (e.g. '## References'); body is
    the text that follows until the next heading of the same or higher level.

    A synthetic ('', ...) section is prepended for any content before the
    first heading.
    """
    heading_re = re.compile(r'^(#{1,6})\s+(.*)', re.MULTILINE)
    sections: list[tuple[str, str]] = []
    prev_end = 0
    prev_heading = ''

    for m in heading_re.finditer(md_text):
        body = md_text[prev_end:m.start()]
        sections.append((prev_heading, body))
        prev_heading = m.group(0)
        prev_end = m.end()

    # Append the last section
    sections.append((prev_heading, md_text[prev_end:]))
    return sections


_REFERENCE_HEADING_RE = re.compile(
    r'^#{1,6}\s+(References|Bibliography)\s*$',
    re.IGNORECASE | re.MULTILINE,
)

_LIST_ITEM_RE = re.compile(
    r'^\s*[-*+]\s+(.*)',
    re.MULTILINE,
)


def extract_non_reference_lists(md_text: str) -> list[str]:
    """
    Return all markdown list items that do NOT appear under a
    'References' or 'Bibliography' heading.
    """
    sections = split_into_sections(md_text)
    items: list[str] = []

    for heading, body in sections:
        if heading and _REFERENCE_HEADING_RE.match(heading):
            # Skip list items that belong to the reference section
            continue
        for m in _LIST_ITEM_RE.finditer(body):
            item = m.group(1).strip()
            if item:
                items.append(item)

    return items


def extract_references(md_text):
    # Find the section starting with "## References" or "## Bibliography"
    section_match = re.search(r"^#{1,6}\s+(References|Bibliography)\s*\n(.*)", md_text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
    if not section_match:
        print("No references section found.")
        return []

    # Extract everything after the heading
    refs_block = section_match.group(2).strip()

    # Split into individual references using bullet markers or newlines
    raw_refs = re.split(r'^\s*[\*\-+]\s*', refs_block, flags=re.MULTILINE)
    cleaned_refs = []

    for ref in raw_refs:
        ref = ref.strip()
        if not ref:
            continue
        # Remove leading reference numbers: (1), [1], 1.
        ref = re.sub(r'^(\(\d+\)|\[\d+\]|\d+\.)\s*', '', ref)
        # Remove [<AuthorName>]
        ref = re.sub(r'^\[[^\]]+\]\s*', '', ref)
        cleaned_refs.append(ref)
    return cleaned_refs


def extract_equations(latex: str) -> list[str]:
    """
    Extract various equation-like blocks from the Nougat `.mmd` LaTeX output,
    using pylatexenc to handle symbols, accents, and nested macros.
    """
    eq_pattern = re.compile(
        r'\\begin\{(equation\*?|align\*?|aligned\*?|eqnarray\*?)\}(.*?)\\end\{\1\}'
        r'|\\beqa(.*?)\\eeqa'
        r'|\\bge(.*?)\\ede'
        r'|\$\$(.*?)\$\$'
        r'|\\\[(.*?)\\\]'
        r'|\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}',
        re.DOTALL
    )

    tex2text = LatexNodes2Text()
    blocks = []
    for m in eq_pattern.finditer(latex):
        groups = m.groups()
        for body in groups[1:]:
            if body is not None:
                # Split gather blocks on line breaks
                if '\\begin{gather' in m.group(0):
                    lines = re.split(r'(?<!\\)\\\\(?!\[)', body)
                    for line in lines:
                        cleaned = tex2text.latex_to_text(line.strip())
                        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                        if cleaned:
                            blocks.append(cleaned)
                else:
                    cleaned = tex2text.latex_to_text(body.strip())
                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                    if cleaned:
                        blocks.append(cleaned)
                break
    return blocks


def extract_raw(base_dir: str, label: str, pdf: PDF) -> Tuple[bool, List[Tuple[str, int, str, str]]]:
    results: List[Tuple[str, int, str, str]] = []
    md_name = os.path.splitext(pdf.pdf_name)[0] + ".mmd"
    output_md = os.path.join(base_dir, md_name)

    if not os.path.exists(output_md):
        print(f"[ERROR] No Markdown output found for {pdf.pdf_name}. Run extraction first.")
        return False, results

    try:
        with open(output_md, "r", encoding="utf-8") as f:
            content = f.read()

        # Simple label mapping: you can expand this for more fine-grained tasks
        # For now: split by markdown syntax types
        label_extractors = {
            "title": lambda text: [
                m.group(1).strip()
                for m in (re.match(r'^#\s+(.*)', l) for l in text.splitlines())
                if m
             ][:1],
            "section": lambda text: [
                # Remove any leading numbering like "3", "3.1", "3.6.4.", etc.
                re.sub(r'^\d+(?:\.\d+)*\.?\s*', '', line.lstrip("# ").strip())
                for line in text.splitlines()
                if line.startswith("#")
            ],
            "list": lambda text: extract_non_reference_lists(text),
            "table": lambda text: extract_tables(text),
            'equation': lambda text: extract_equations(text),
            'reference': lambda text: extract_references(text),
            "footer": lambda text: [
                m.group(1).strip()
                for m in re.finditer(
                    r'^Footnote [^:]+:\s*(.*)$',
                    text,
                    re.MULTILINE
                )
            ],
        }

        if label not in label_extractors:
            print(f"[ERROR] Unsupported label for Nougat: {label}")
            return False, results

        # prepare a single Latex→text converter
        tex2text = LatexNodes2Text()

        # run the extractor for this label
        extracted_texts = label_extractors[label](content)

        for entry in extracted_texts:
            # 1) pull out any inline equations and emit them first
            for eq in extract_equations(entry):
                unicode_eq = tex2text.latex_to_text(eq.strip())
                cleaned_eq = normalize_string(unicode_eq)
                if cleaned_eq:
                    results.append((pdf.pdf_name, 0, "equation", cleaned_eq))

            # 2) convert the full entry (with inline math) to unicode, then normalize
            unicode_entry = tex2text.latex_to_text(entry)
            normalized = normalize_string(unicode_entry)
            if normalized:
                results.append((pdf.pdf_name, 0, label, normalized))

        return True, results

    except Exception as e:
        print(f"[ERROR] Failed to parse Nougat output for {pdf.pdf_name}: {e}")
        return False, results