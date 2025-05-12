# Usage: keep all link that you want the pdf link in pdf_link.txt, then python or python3 the marker_runner.py
import os
import re
import yaml
import urllib.request
from collections import OrderedDict
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
import requests
import feedparser
import argparse


# === Configuration ===
pdf_link_file = "pdf_link.txt"
output_yaml = "papers.yaml"
download_dir = "downloads"


# === Marker configuration ===
config = {"output_format": "markdown"}
config_parser = ConfigParser(config)
converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service()
)

# === Ensure directories exist ===
os.makedirs(download_dir, exist_ok=True)

# === Utility functions ===
def extract_id_from_url(url):
    match = re.search(r'arxiv\.org/pdf/(\d{4}\.\d+)(v\d+)?', url)
    if match:
        base = match.group(1)
        version = match.group(2) or ''
        return base + version 
    return None

def download_pdf(url, save_dir):
    paper_id = extract_id_from_url(url)
    local_path = os.path.join(save_dir, f"{paper_id}.pdf")
    if not os.path.exists(local_path):
        print(f"  [↓] Downloading {url} ...")
        urllib.request.urlretrieve(url, local_path)
    return local_path

def split_sections(markdown_text):
    sections = OrderedDict()
    current_section = None
    for line in markdown_text.splitlines():
        line = line.strip()
        if not line:
            continue
        heading_match = (
            re.match(r"^(#+)\s+(.*)", line) or                     
            re.match(r"^(?:[IVXLCDM]+)\.\s+(.*)", line, re.IGNORECASE) or  
            re.match(r"^\d+(\.\d+)*\.\s+(.*)", line)                
        )

        if heading_match:
            heading_text = heading_match.groups()[-1] 
            current_section = heading_text.strip().lower().replace(" ", "_")
            sections[current_section] = ""
        elif current_section:
            sections[current_section] += line + "\n"

    return sections


def extract_metadata(sections, markdown_text=None):
    keywords = ""

    # 1. First, try normal section headings
    for key in sections:
        lowered = key.lower().replace('*', '').replace('_', ' ').strip()
        if not keywords and any(kw in lowered for kw in ["keywords", "key words", "index terms", "index term"]):
            keywords = re.sub(r'[_\n]+', ' ', sections[key]).strip()

    # 2. If still not found, scan full markdown for "*Index Terms*" or "*Keywords*"
    if not keywords and markdown_text:
        pattern = re.compile(
            r'\*(?:index terms|keywords|key words|index term)\*\s*[—\-–]?\s*(.+?)(?:\n|\#|\Z)', 
            re.IGNORECASE | re.DOTALL
        )
        match = pattern.search(markdown_text)
        if match:
            keywords = match.group(1).strip()
            # Post-clean: stop if there’s a heading or big paragraph start
            keywords = keywords.split('\n')[0].strip()
            keywords = re.sub(r'\s+', ' ', keywords)  # Fix multiple spaces

    return keywords

def trim_document(markdown_text):
    # Match various Introduction styles
    intro_pattern = re.compile(
        r"(?m)^(?:#+\s*)?(?:\d+\.\d*\s*)?(?:[IVXLCDM]+\.\s*)?(?:Introduction|INTRODUCTION)\b"
    )
    
    match = intro_pattern.search(markdown_text)
    if match:
        return markdown_text[match.start():].strip()
    
    return markdown_text.strip()

# === Detect is EN ===
def is_english(text):
    ascii_letters = sum(c.isascii() and c.isalpha() for c in text)
    total_letters = sum(c.isalpha() for c in text)

    # Avoid division by zero
    if total_letters == 0:
        return False

    ratio = ascii_letters / total_letters

    # If more than 80% of letters are ASCII, treat as English
    return ratio >= 0.8

def extract_title_abstract(markdown_text, sections, paper_id):
    # Title: First heading or fallback to filename
    lines = markdown_text.splitlines()
    first_heading = next((line.strip("# ").strip() for line in lines if line.startswith("#")), None)
    title = first_heading if first_heading else paper_id

    # Abstract: Look for section named "abstract"
    abstract = ""
    for key in sections:
        if "abstract" in key.lower():
            abstract = sections[key].strip()
            break

    return title, abstract

def main(input_path, output_yaml):
    papers = []

    if input_path.endswith(".txt"):
        mode = "url"
        pdf_link_file = input_path
        download_dir = "./downloaded_pdfs"
        os.makedirs(download_dir, exist_ok=True)

        with open(pdf_link_file, "r", encoding="utf-8") as f:
            pdf_urls = [line.strip() for line in f if line.strip()]

        paper_ids = [extract_id_from_url(url) for url in pdf_urls]
        id_query = "+OR+".join([f"id:{pid.split('v')[0]}" for pid in paper_ids if pid]) 

        feed = feedparser.parse(requests.get(
            f"http://export.arxiv.org/api/query?search_query={id_query}&start=0&max_results=100"
        ).text)

        id_to_metadata = {}
        for entry in feed.entries:
            paper_id = entry.id.split('/')[-1]
            paper_id_base = paper_id.split('v')[0]
            id_to_metadata[paper_id_base] = {
                "title": entry.title.strip(),
                "abstract": entry.summary.strip(),
                "url": entry.id
            }

        seen_ids = set()
        for url in pdf_urls:
            try:
                paper_id = extract_id_from_url(url)
                if paper_id in seen_ids:
                    print(f"[!] Skipping already-processed ID: {paper_id}")
                    continue
                paper_id_base = paper_id.split('v')[0]

                seen_ids.add(paper_id)
                print(f"[->] Converting {paper_id}")

                local_pdf_path = download_pdf(url, download_dir)
                rendered = converter(local_pdf_path)
                markdown_text, _, _ = text_from_rendered(rendered)

                if not is_english(markdown_text):
                    print(f"[SKIP] {paper_id} → Not English")
                    continue

                keywords = extract_metadata({}, markdown_text)
                document = trim_document(markdown_text)

                metadata = id_to_metadata.get(paper_id_base, {})
                title = metadata.get("title", paper_id)
                abstract = metadata.get("abstract", "")
                arxiv_url = metadata.get("url", url)

                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "url": arxiv_url,
                    "keywords": keywords,
                    "document": document
                })

                print(f"[✓] Added: {paper_id}")

            except Exception as e:
                print(f"[✗] Failed to convert {url}: {e}")

    else:
        mode = "pdf"
        download_dir = input_path

        pdf_files = [f for f in os.listdir(download_dir) if f.lower().endswith(".pdf")]
        for pdf_filename in pdf_files:
            try:
                paper_id = os.path.splitext(pdf_filename)[0]
                print(f"[->] Converting {pdf_filename}")

                local_pdf_path = os.path.join(download_dir, pdf_filename)
                rendered = converter(local_pdf_path)
                markdown_text, _, _ = text_from_rendered(rendered)

                if not is_english(markdown_text):
                    print(f"[SKIP] {paper_id} → Not English")
                    continue

                sections = split_sections(markdown_text)
                keywords = extract_metadata(sections, markdown_text)
                document = trim_document(markdown_text)
                title, abstract = extract_title_abstract(markdown_text, sections, paper_id)

                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "keywords": keywords,
                    "document": document
                })

                print(f"[✓] Added: {paper_id}")

            except Exception as e:
                print(f"[✗] Failed to convert {pdf_filename}: {e}")

    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"papers": papers}, f, allow_unicode=True, sort_keys=False)

    print(f"\n✅ All done! YAML saved to {output_yaml}")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    parser = argparse.ArgumentParser(description="Extract and convert arXiv papers to YAML.")
    parser.add_argument("-f", "--file", required=True, help="Path to folder (PDFs) or file (URLs)")
    parser.add_argument("-o", "--output", required=True, help="Output YAML file")

    args = parser.parse_args()
    main(args.file, args.output)