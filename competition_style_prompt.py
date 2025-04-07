import sys
import os
from pdfminer.high_level import extract_text
from markdownify import markdownify as md
import re
from collections import defaultdict
import itertools
import openai
from dotenv import load_dotenv, find_dotenv


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

if len(sys.argv) != 2:
    print("Usage: python3 compare_papers_from_pdf.py papers.txt")
    sys.exit(1)

with open(sys.argv[1], "r") as f:
    pdf_files = [line.strip() for line in f if line.strip()]


papers = {}
for pdf_path in pdf_files:
    print(f"\n[+] Processing: {pdf_path}")
    md = convert_pdf_to_markdown(pdf_path)
    sections = extract_sections(md)
    papers[pdf_path] = sections


scores = defaultdict(int)
paper_names = list(papers.keys())

for paper_a, paper_b in itertools.combinations(paper_names, 2):
    print(f"\n===== Comparing: {paper_a} vs {paper_b} =====\n")

    sections_a = papers[paper_a]
    sections_b = papers[paper_b]
    common_sections = set(sections_a.keys()).intersection(sections_b.keys())

    if not common_sections:
        print("No common sections found. Skipping.")
        continue

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
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )

        result = response.choices[0].message.content

        # Simple vote counter
        result_lower = result.lower()
        if "paper a" in result_lower:
            scores[paper_a] += 1
        elif "paper b" in result_lower:
            scores[paper_b] += 1


print("\n===== FINAL RANKING =====")
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for i, (paper, score) in enumerate(ranked, 1):
    print(f"{i}. {paper} — {score} points")