#!/usr/bin/env python3
import re
import json
import sys
from pylatexenc.latex2text import LatexNodes2Text, MacroTextSpec
import pylatexenc

def strip_comments(latex: str) -> str:
    lines = latex.splitlines()
    clean = []
    for line in lines:
        clean.append(re.split(r'(?<!\\)%', line)[0])
    return "\n".join(clean)

def extract_all_macro_args(latex: str, macro: str) -> list[str]:
    """
    Extracts the braced argument(s) of \macro{...}, including nested braces.
    """
    pat = re.compile(fr'\\{macro}\*?(?:\[[^\]]*\])?\{{')
    results = []
    for m in pat.finditer(latex):
        i = m.end()
        depth = 1
        buf = []
        while i < len(latex) and depth > 0:
            c = latex[i]
            if c == '{':
                depth += 1
                buf.append(c)
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
                buf.append(c)
            else:
                buf.append(c)
            i += 1
        results.append(''.join(buf).strip())
    return results

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

def clean_and_flatten(raw_list: list[str], tex2text: LatexNodes2Text) -> list[str]:
    out = []
    for raw in raw_list:
        try:
            txt = tex2text.latex_to_text(raw)
        except Exception as e:
            print("\n[⚠️ ERROR converting LaTeX to text]")
            print("Raw input:")
            print(raw[:500])  # print first 500 characters for context
            print("Error:", e)
            continue
        txt = re.sub(r'\s+', ' ', txt).strip()
        if txt:
            out.append(txt)
    return out


def extract_keywords(latex: str, tex2text: LatexNodes2Text) -> list[str]:
    """
    Grab every \\keywords{…} or \\keyword{…} invocation, split on any
    comma or \\[…]\ break, convert to text, collapse whitespace,
    and return a flat list of keywords.
    """
    raws = (
        extract_all_macro_args(latex, 'keywords') +
        extract_all_macro_args(latex, 'keyword')
    )
    if not raws:
        return []

    keywords: list[str] = []
    # pattern that your old code used to split on \\[…]
    linebreak_split = re.compile(r'\\\\\[[^\]]*\]')

    for raw in raws:
        # remove any \hfill
        raw = raw.replace(r'\hfill', ' ')
        # first split on your linebreak pattern
        segments = [seg.strip() for seg in linebreak_split.split(raw) if seg.strip()]
        for seg in segments:
            # convert LaTeX -> plain text
            txt = tex2text.latex_to_text(seg)
            # collapse whitespace
            txt = re.sub(r'\s+', ' ', txt).strip()
            if not txt:
                continue
            # now split on commas and add each
            for part in txt.split(','):
                part = part.strip()
                if part:
                    keywords.append(part)
    return keywords


def extract_date(latex: str) -> str:
    """
    Grabs the argument to \\date{…}, strips any \\textcolor or other macros,
    and returns a cleaned-up string.
    """
    raw = extract_all_macro_args(latex, 'date')
    if not raw:
        return ''
    txt = LatexNodes2Text().latex_to_text(raw[0])
    return re.sub(r'\s+', ' ', txt).strip()


def extract_sections(latex: str):
    """
    Returns a list of dicts with keys: level, number, title
    """
    sec_pat = re.compile(
        r'\\(section|subsection|subsubsection)\*?(?:\[[^\]]*\])?\{',
        re.MULTILINE
    )

    raw_matches = []
    for m in sec_pat.finditer(latex):
        level = m.group(1)
        i = m.end()
        depth = 1
        buf = []
        while i < len(latex) and depth > 0:
            c = latex[i]
            if c == '{':
                depth += 1; buf.append(c)
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
                buf.append(c)
            else:
                buf.append(c)
            i += 1
        raw = ''.join(buf).strip()
        raw_matches.append((m.start(), level, raw))

    raw_matches.sort(key=lambda x: x[0])
    tex2text = LatexNodes2Text()
    cleaned = [
        (level, re.sub(r'\s+', ' ', tex2text.latex_to_text(raw)).strip())
        for (_, level, raw) in raw_matches
    ]

    sec_ctr = sub_ctr = subsub_ctr = 0
    numbered = []
    for level, title in cleaned:
        if level == 'section':
            sec_ctr += 1; sub_ctr = subsub_ctr = 0
            num = f"{sec_ctr}"
        elif level == 'subsection':
            sub_ctr += 1; subsub_ctr = 0
            num = f"{sec_ctr}.{sub_ctr}"
        else:
            subsub_ctr += 1
            num = f"{sec_ctr}.{sub_ctr}.{subsub_ctr}"

        numbered.append({
            "number": num,
            "title":  title
        })
    return numbered

def extract_paragraphs(latex: str, tex2text: LatexNodes2Text) -> list[str]:
    """
    Extract \paragraph{} and custom <p> environments, exclude equations.
    """
    raw_paras = (
        extract_all_macro_args(latex, 'paragraph') +
        extract_environment_blocks(latex, 'p')
    )
    eq_pattern = re.compile(
        r'\\begin\{(equation\*?|align\*?|aligned)\}.*?\\end\{\1\}'
        r'|\\beqa.*?\\eeqa'
        r'|\\bge.*?\\ede'
        r'|\$\$.*?\$\$'
        , re.DOTALL
    )
    paras_no_eq = [eq_pattern.sub('', rp) for rp in raw_paras]
    return clean_and_flatten(paras_no_eq, tex2text)


def extract_equations(latex: str, tex2text: LatexNodes2Text) -> list[str]:
    """
    Extract various equation-like blocks from the document, including
    splitting gather environments into individual equations.
    """
    # Match block environments
    eq_pattern = re.compile(
        r'\\begin\{(equation\*?|align\*?|aligned\*?|eqnarray\*?)\}(.*?)\\end\{\1\}'
        r'|\\beqa(.*?)\\eeqa'
        r'|\\bge(.*?)\\ede'
        r'|\$\$(.*?)\$\$'
        r'|\\\[(.*?)\\\]'
        r'|\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}'
        , re.DOTALL
    )

    blocks = []
    for m in eq_pattern.finditer(latex):
        # Groupdict mapping for the named groups (many will be None)
        groups = m.groups()

        for body in groups[1:]:
            if body is not None:
                if '\\begin{gather' in m.group(0):
                    # Split gather block on \\ not followed by [
                    split_lines = re.split(r'(?<!\\)\\\\(?!\[)', body)
                    for line in split_lines:
                        line = line.strip()
                        if line:
                            blocks.append(line)
                else:
                    blocks.append(body.strip())
                break

    return clean_and_flatten(blocks, tex2text)


def number_citations(latex: str, bib_entries: list[dict]) -> str:
    """
    Given the raw LaTeX and the list of bibliography entries
    (each with a 'key' in the order they appear),
    replace every \cite{key} (and comma-separated lists) with “[n]”.
    """
    # build a mapping from key -> its numeric index (1-based)
    cite_map = { entry['key']: i+1
                 for i, entry in enumerate(bib_entries) }

    # match \cite{key1,key2,…}  (you can easily extend this regex
    # to handle \citep, \citet, optional args, etc.)
    cite_re = re.compile(r'\\cite\{([^}]+)\}')

    def repl(m):
        keys = [k.strip() for k in m.group(1).split(',')]
        nums = [ str(cite_map.get(k, k)) for k in keys ]
        return "[" + ",".join(nums) + "]"

    return cite_re.sub(repl, latex)


def extract_bibliography(tex2text, latex_content: str) -> list[dict]:
    m = re.search(
        r'\\begin\{thebibliography\}(.*?)\\end\{thebibliography\}',
        latex_content,
        re.DOTALL
    )
    if not m:
        return []

    # Remove latex comments
    bib_body = m.group(1)
    bib_body = "\n".join(
        re.split(r'(?<!\\)%', line)[0] for line in bib_body.splitlines()
    )

    # Match \bibitem[optional]{key}, allowing optional arg with spaces/newlines
    pattern = re.compile(
        r'\\(?:bibitem|harvarditem)\s*'
        r'(?:\[[^\]]*?\])?'
        r'\s*\{(?P<key>[^\}]+)\}'
        r'(?P<entry>.*?)(?=\\(?:bibitem|harvarditem)|\Z)',
        re.DOTALL
    )

    entries = []
    for em in pattern.finditer(bib_body):
        key = em.group('key').strip()
        raw = em.group('entry').strip()

        # --- NEW: unwrap all \bibinfo{field}{value} into just 'value' ---
        raw = re.sub(
            r'\\bibinfo\{[^\}]+\}\{([^\}]+)\}',
            r'\1',
            raw,
            flags=re.DOTALL
        )
        # ----------------------------------------------------------------

        # turn \foo{bar} → bar
        raw = re.sub(
            r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])*\{([^\}]*)\}',
            r'\1',
            raw,
            flags=re.DOTALL
        )
        # drop any leftover \macroname
        raw = re.sub(r'\\[a-zA-Z]+\*?', '', raw)

        # Normalize whitespace and strip remaining LaTeX
        text = tex2text.latex_to_text(raw)
        text = re.sub(r'\s+', ' ', text).strip()

        entries.append({'key': key, 'text': text})

    return entries


def extract_tables_with_tabulars(latex: str) -> list[str]:
    tables = extract_environment_blocks(latex, 'table')
    sideways_tables = extract_environment_blocks(latex, 'sidewaystable')
    all_tables = tables + sideways_tables
    
    tabular_blocks = []
    for tbl in all_tables:
        tabular_blocks.extend(extract_environment_blocks(tbl, 'tabular'))
        tabular_blocks.extend(extract_environment_blocks(tbl, 'tabularx'))  # also catch tabularx if used
    
    return tabular_blocks


def extract_metadata(latex: str) -> dict:
    clean = strip_comments(latex)

    tex2text = LatexNodes2Text(macros=[
        MacroTextSpec('href', simplify_repl=lambda node, **kwargs: (
            node.nodeargd.argnlist[1].nodelist_to_latex() if len(node.nodeargd.argnlist) > 1 else '')
        )
    ])
    
    bib = extract_bibliography(tex2text, clean)

    clean = number_citations(clean, bib)

    author_names = clean_and_flatten(
        extract_all_macro_args(clean, 'author'),
        tex2text
    )

    affiliations = clean_and_flatten(
        extract_all_macro_args(clean, 'affiliation'), tex2text
    )

    # catch both \emailAdd{…} (jcappub style) and plain \email{…}
    emails = clean_and_flatten(
        extract_all_macro_args(clean, 'emailAdd')
      + extract_all_macro_args(clean, 'email'),
        tex2text
    )

    # Build structured author objects
    authors = []
    for i, name in enumerate(author_names):
        authors.append({
            'author_name': name,
            'affiliation': affiliations,
            'email':      emails.pop(0) if emails else ""
        })

    return {
        'title': clean_and_flatten(extract_all_macro_args(clean, 'title'), tex2text)[:1],
        'author': authors,
        'date': extract_date(clean),
        'abstract': clean_and_flatten(
            extract_all_macro_args(clean, 'abstract') +
            extract_environment_blocks(clean, 'abstract'),
            tex2text
        )[:1],
        'keyword': extract_keywords(clean, tex2text),
        'copyright' : [],
        'funding': [],
        'section': extract_sections(clean),
        'caption': clean_and_flatten(extract_all_macro_args(clean, 'caption'), tex2text),
        'paragraph': extract_paragraphs(clean, tex2text),
        'equation': extract_equations(clean, tex2text),
        'header': clean_and_flatten(
            sum((extract_all_macro_args(clean, m) for m in ('lhead','chead','rhead','header')), []),
            tex2text),
        'footer': clean_and_flatten(
            sum((extract_all_macro_args(clean, m) for m in ('lfoot','cfoot','rfoot','footer','footnote')), []),
            tex2text),
        'figure': [],
        'table': clean_and_flatten(
            extract_tables_with_tabulars(clean),
            tex2text
        ),
        'list': clean_and_flatten(
            sum((extract_environment_blocks(clean, ev) for ev in ('itemize','enumerate')), []),
            tex2text),
        'bibliography': bib
    }


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: extract_latex_labels.py <input.tex> <output.json>")
        sys.exit(1)

    tex_path, json_path = sys.argv[1], sys.argv[2]
    with open(tex_path, encoding='utf-8') as f:
        tex = f.read()

    metadata = extract_metadata(tex)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Done. Extracted fields:")
    for k, v in metadata.items():
        print(f"  {k:12s}: {len(v) if isinstance(v, list) else '1'} item(s)")
