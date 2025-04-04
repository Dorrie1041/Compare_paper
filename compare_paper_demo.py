import openai
from dotenv import load_dotenv, find_dotenv
import sys
import os
import re

if len(sys.argv) != 3:
    print("Usage: python3 f.py paper_a.md paper_b.md")
    sys.exit(1)
file_a = sys.argv[1]
file_b = sys.argv[2]


_ = load_dotenv(find_dotenv())
openai.api_key = "---"
client = openai.OpenAI(api_key=openai.api_key)

with open(file_a, "r", encoding="utf-8") as f:
    paper_a = f.read()

with open(file_b, "r", encoding="utf-8") as f:
    paper_b = f.read()

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


sections_a = extract_sections(paper_a)
sections_b = extract_sections(paper_b)

# compare matching sections, like abstract compare with abstract
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