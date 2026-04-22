<h1 align="center"> <strong><em>eSciBench</em></strong></br> An Extensible Scientific PDF Extraction Benchmark</h1>

## Overview
eSciBench is a benchmark and a framework made to evaluate PDF axtraction tools on scientific publications. Inspired by [Meuschke et al.](https://link.springer.com/chapter/10.1007/978-3-031-28032-0_31#citeas)'s work ([GitHub repo](https://github.com/gipplab/pdf-benchmark/tree/main)), eSciBench proposes an improved benchmark with a brand new dataset.

### The dataset
The dataset is composed of 100 scientific documents, for a total of 1 949 pages, all sourced from
arXiv over the 2009-2025 period and encompassing all arXiv categories, ensuring a variety of layouts, contents and bibliographical styles. Only articles with available LATEX source files were selected,
allowing accurate ground truth and PDF generation. 

The following table shows the arXiv's categories and the volumetry of the dataset.

| Category               | # PDFs | # Pages |
|------------------------|--------|---------|
| Physics                | 17     | 428     |
| Mathematics            | 16     | 347     |
| Computer Science       | 17     | 213     |
| Quantitative Biology   | 16     | 218     |
| Quatitative Finance    | 7      | 230     |
| Statistics             | 2      | 30      |
| EESS                   | 18     | 174     |
| Economics              | 8      | 309     |
| **Total**              | **100**| **1 949** |

The [dataset](https://huggingface.co/datasets/TNoahT/eSciBench) can be found on HuggingFace.

### Current extraction tools benchmarked



Add the following documents manually:
- Download the [cermine.jar](https://github.com/CeON/CERMINE) in /benchmark/extractors/cermine/
- Download the [pdfact.jar](https://github.com/ad-freiburg/pdfact) in /benchmark/extractors/pdfact/
- Clone [refextract](https://github.com/inspirehep/refextract) in /benchmark/extractors/refectract/
- Clone [unstructured](https://github.com/Unstructured-IO/unstructured) in /benchmark/extractors/unstructured/

## Installation

## Results

## Adding more extraction tools


