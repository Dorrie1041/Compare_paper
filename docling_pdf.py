import os
import re
import yaml
from urllib.parse import urlparse
from collections import OrderedDict
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pdf_link_file = "pdf_link.txt"
output_yaml = "papers.yaml"
artifacts_path = "/Users/dongruishen/.cache/docling/models"

# === extract ID from URL (for arXiv style) ===
def extract_id_from_url(url):
    parsed = urlparse(url)
    if "arxiv.org" in parsed.netloc:
        return os.path.splitext(os.path.basename(parsed.path))[0]
    return "unknown"

# === extract metadata from markdown ===
def extract_metadata(sections):
    title = ""
    abstract = ""
    keywords = ""

    for key in sections:
        lowered = key.lower()
        if not title and ('title' in lowered or 'the_' in lowered or 'towards' in lowered):
            title = re.sub(r'_+', ' ', key).strip()
        if not abstract and 'abstract' in lowered:
            abstract = sections[key].strip().replace('\n', ' ')
        if not keywords and ('keywords' in lowered):
            keywords = re.sub(r'[_\n]+', ' ', sections[key]).strip()

    return title, abstract, keywords

# === split markdown into sections ===
def split_sections(markdown_text):
    sections = OrderedDict()
    current_section = None

    for line in markdown_text.splitlines():
        line = line.strip()
        if not line:
            continue

        heading_match = re.match(r"^(#+)\s+(.*)", line)
        if heading_match:
            current_section = heading_match.group(2).strip().lower().replace(" ", "_")
            sections[current_section] = ""
        elif current_section:
            sections[current_section] += line + "\n"

    return sections

# === cut off everything before the first heading after abstract ===
def trim_document(markdown_text):
    abstract_match = re.search(r"^##\s*abstract\s*$", markdown_text, re.IGNORECASE | re.MULTILINE)
    if abstract_match:
        remaining = markdown_text[abstract_match.end():]
        next_heading = re.search(r"^##\s+", remaining, re.MULTILINE)
        if next_heading:
            return remaining[next_heading.start():].strip()
    return markdown_text.strip()

# === docling markdown convert ===
pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

papers = []
with open(pdf_link_file, "r", encoding="utf-8") as f:
    pdfs = [line.strip() for line in f if line.strip()]

for pdf in pdfs:
    try:
        paper_id = extract_id_from_url(pdf)
        print(f"[->] Converting {paper_id}")

        result = doc_converter.convert(pdf)
        markdown_text = result.document.export_to_markdown()
        trimmed_markdown = trim_document(markdown_text)
        sections = split_sections(markdown_text)

        title, abstract, keywords = extract_metadata(sections)

        document_content = trim_document(markdown_text) if title and abstract else markdown_text

        paper_entry = {
            "title": title or paper_id,
            "abstract": abstract,
            "url": pdf,
            "keywords": keywords,
            "document": document_content.strip() if title and abstract else ""
        }

        papers.append(paper_entry)
        print(f"[✓] Added: {paper_id}")
    except Exception as e:
        print(f"[✗] Failed to convert {pdf}: {e}")

# === Save all to YAML ===
with open(output_yaml, "w", encoding="utf-8") as f:
    yaml.dump({"papers": papers}, f, allow_unicode=True, sort_keys=False)

print(f"\n✅ All done! YAML saved to {output_yaml}")
