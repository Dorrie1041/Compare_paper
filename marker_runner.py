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
from dotenv import load_dotenv, find_dotenv
from litellm import completion



# === Configuration ===
pdf_link_file = "pdf_link.txt"
output_yaml = "papers.yaml"
download_dir = "downloads"

# litellm key input
_ = load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = "sk-"

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
        return base + version  # returns '2504.11380' or '2504.11380v2'
    return None

def download_pdf(url, save_dir):
    paper_id = extract_id_from_url(url)
    local_path = os.path.join(save_dir, f"{paper_id}.pdf")
    if not os.path.exists(local_path):
        print(f"  [↓] Downloading {url} ...")
        urllib.request.urlretrieve(url, local_path)
    return local_path

def extract_metadata(sections, markdown_text=None):
    title = ""
    abstract = ""
    keywords = ""

    # Try to find abstract and keywords from section keys
    for key in sections:
        lowered = key.lower()
        if not abstract and 'abstract' in lowered:
            abstract = sections[key].strip().replace('\n', ' ')
        if not keywords and 'keywords' in lowered:
            keywords = re.sub(r'[_\n]+', ' ', sections[key]).strip()

    # Updated title logic: find the first level-1 heading if markdown is provided
    if markdown_text:
        title_match = re.search(r"^#\s+(.*)", markdown_text, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()

    return title, abstract, keywords

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

def trim_document(markdown_text):
    intro_pattern = re.compile(
        r"^(#{1,6})\s*(\d+[\.\d]*\s+)?Introduction\b", re.IGNORECASE | re.MULTILINE
    )
    match = intro_pattern.search(markdown_text)
    if match:
        return markdown_text[match.start():].strip()
    return ""

def is_disqualified(markdown_text):
    prompt = f"""You are reviewing academic papers. Disqualify any paper that meets any of the following criteria:

1. No evaluation – there is no experimental, empirical, or quantitative analysis.
   A paper is considered to have evaluation **only if** it includes at least one of the following:  
        • A section labeled “Experiments”, “Evaluation”, or similar  
        • A table showing accuracy, performance, benchmarks, timing, etc.  
        • Charts or graphs comparing results  
        • Qualitative or quantitative analysis of outcomes  
        • Phrases like:  
            – “We evaluate our method on…”  
            – “Table 1 shows…”  
            – “Our approach achieves better results than…”

2. No related work – the paper does not engage with prior research in a meaningful way. This includes cases where:
        •	It cites few or no research papers,
        •	It only cites tools, frameworks, or datasets (e.g., PyTorch, COCO),
        •	It does not explain or compare to the contributions of cited work,
        •	It lacks a section labeled “Related Work” or any such discussion embedded in the introduction.

3. Not in English – the paper is written in a language other than English.

4. No novelty – the work does not offer any new idea, method, result, or insight.
   A paper is considered to have novelty **only if it introduces something not previously done**. Signs of novelty include:  
   • A new method, algorithm, framework, dataset, or result  
   • A new application of existing techniques to a previously unstudied domain  
   • Claims like:  
     – “We propose a new...”  
     – “Our main contribution is...”  
     – “To the best of our knowledge, this is the first...”  
     – “Unlike previous approaches, our method...”  
   If the paper merely reuses existing techniques without proposing any improvement, innovation, or new perspective, it should be disqualified.

5. Survey/review paper – the paper summarizes existing work without proposing a new approach.
        • If the **title contains the word "survey" or "review"**, disqualify it immediately without analyzing the full content.  
        • Otherwise, disqualify only if the main body primarily summarizes existing work without introducing any new contribution.


Here is the paper content (in Markdown):

\"\"\"{markdown_text[:8000]}\"\"\"  # Trim for token limits

Answer with either:
- "Disqualified: <reason>"
- "Qualified"
"""

    response = completion(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200
    )

    return response["choices"][0]["message"]["content"].strip()

# === Main process ===
papers = []
with open(pdf_link_file, "r", encoding="utf-8") as f:
    pdf_urls = [line.strip() for line in f if line.strip()]

seen_ids = set()
for url in pdf_urls:
    try:
        paper_id = extract_id_from_url(url)
        if paper_id in seen_ids:
            print(f"[!] Skipping already-processed ID: {paper_id}")
            continue
        seen_ids.add(paper_id)
        print(f"[->] Converting {paper_id}")

        local_pdf_path = download_pdf(url, download_dir)

        # Use Marker Python API to convert
        rendered = converter(local_pdf_path)
        markdown_text, _, _ = text_from_rendered(rendered)

        # avoid all disqualify papers save in the yaml
        decision = is_disqualified(markdown_text)
        print(f"[FILTER] {paper_id} → {decision}")

        cleaned = decision.strip().lower().strip('"').strip("'")

        if cleaned.startswith("qualified"):

            sections = split_sections(markdown_text)
            title, abstract, keywords = extract_metadata(sections, markdown_text)
            document = trim_document(markdown_text) if title and abstract else markdown_text

            papers.append({
                "title": title or paper_id,
                "abstract": abstract,
                "url": url,
                "keywords": keywords,
                "document": document
            })
        else:
            print(f"[✓] Added: {paper_id}")

    except Exception as e:
        print(f"[✗] Failed to convert {url}: {e}")

# === Save all results to YAML ===
with open(output_yaml, "w", encoding="utf-8") as f:
    yaml.dump({"papers": papers}, f, allow_unicode=True, sort_keys=False)

print(f"\n✅ All done! YAML saved to {output_yaml}")