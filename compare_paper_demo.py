import openai
from dotenv import load_dotenv, find_dotenv
from pdfminer.high_level import extract_text
from markdownify import markdownify as md
import sys
import os
import re

_ = load_dotenv(find_dotenv())
openai.api_key = "---"
client = openai.OpenAI(api_key=openai.api_key)

# STEP 1: Convert PDF to Markdown
def convert_pdf_to_markdown(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"[+] Extracting text from: {pdf_path}")
    text = extract_text(pdf_path)
    print(f"[+] Converting to Markdown...")
    markdown = md(text)
    return markdown

# STEP 2: Extract Sections from Markdown
def extract_sections(markdown_text):
    section_pattern = r'(?m)^(?P<header>([A-Z][A-Z\s\-]+|[0-9]+[\.\d]*\s+[A-Z][A-Z\s\-]+))$'
    matches = list(re.finditer(section_pattern, markdown_text))
    sections = {}
    for i, match in enumerate(matches):
        title = match.group('header').strip().lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        content = markdown_text[start:end].strip()
        sections[title] = content
    return sections

# STEP 3: Compare Sections
def compare_sections(sections_a, sections_b):
    common_sections = set(sections_a.keys()).intersection(sections_b.keys())

    if not common_sections:
        print("No matching top-level sections found between papers.")
        sys.exit(1)

    for section in sorted(common_sections):
        prompt = f"""
Compare the following **{section}** sections from two papers:

### {section} of Paper A:
{sections_a[section]}

### {section} of Paper B:
{sections_b[section]}

Evaluate them based on:
1. Impact – How significant is the described contribution?
2. Novelty – How original is the approach or idea?
3. Clarity – How clearly is the section written?

Give 1 point to the better paper per category. Do not allow ties — always choose one paper as better for each criterion.
"""
        print(f"\n===== Comparing Section: {section} =====\n")
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        print(response.choices[0].message.content)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 compare_papers_from_pdf.py paper_a.pdf paper_b.pdf")
        sys.exit(1)

    pdf_a = sys.argv[1]
    pdf_b = sys.argv[2]

    # Step 1: Convert PDFs to markdown
    paper_a_md = convert_pdf_to_markdown(pdf_a)
    paper_b_md = convert_pdf_to_markdown(pdf_b)

    # Step 2: Extract sections
    sections_a = extract_sections(paper_a_md)
    sections_b = extract_sections(paper_b_md)

    # Step 3: Compare them
    compare_sections(sections_a, sections_b)