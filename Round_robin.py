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
openai.api_key = "sk---"
client = openai.OpenAI(api_key=openai.api_key)

# Convert PDF to Markdown
def convert_pdf_to_markdown(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)
    print(f"[+] Extracting text from: {pdf_path}")
    text = extract_text(pdf_path)
    print(f"[+] Converting to Markdown...")
    markdown = md(text)
    return markdown

# Extract top-level sections
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

# Main execution
if len(sys.argv) != 2:
    print("Usage: python3 comp_paper.py papers.txt")
    sys.exit(1)

with open(sys.argv[1], "r") as f:
    pdf_files = [line.strip() for line in f if line.strip()]

papers = {}
for pdf_path in pdf_files:
    print(f"\n[+] Processing: {pdf_path}")
    markdown_text = convert_pdf_to_markdown(pdf_path)
    sections = extract_sections(markdown_text)
    papers[pdf_path] = sections

match_results = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "points": 0})
paper_names = list(papers.keys())

# each paper compare with others paper except itself
for i, (paper_a, paper_b) in enumerate(itertools.combinations(paper_names, 2), 1):
    print(f"\n===== Match #{i}: {os.path.basename(paper_a)} vs {os.path.basename(paper_b)} =====\n")
    sections_a = papers[paper_a]
    sections_b = papers[paper_b]
    common_sections = set(sections_a.keys()).intersection(sections_b.keys())

    if not common_sections:
        print("No common sections found. Skipping.")
        continue

    wins_a = 0
    wins_b = 0

    for section in sorted(common_sections):
        prompt = f"""
You are acting as a NeurIPS reviewer. Compare the following **{section}** sections from two papers:

### {section} of Paper A:
{sections_a[section]}

### {section} of Paper B:
{sections_b[section]}

---

For each paper, write a review including:
1. A short **summary** of what the section covers.
2. The section's **strengths and weaknesses**.
3. Assign a score for:
   - **Novelty** (1–10)
   - **Significance** (1–10)
   - **Clarity** (1–10)
4. Provide a **confidence level** in your evaluation (1–5).

Finally, choose which paper is stronger **for this section only**, and explain why.

Respond only with the final decision in this format: `Winner: Paper A` or `Winner: Paper B`.
"""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            result = response.choices[0].message.content.strip()
            print(f"[Section: {section.title()}] → {result}")
            if "Winner: Paper A" in result:
                wins_a += 1
            elif "Winner: Paper B" in result:
                wins_b += 1
        except Exception as e:
            print(f"Error comparing section '{section}': {e}")

    if wins_a > wins_b:
        match_results[paper_a]["wins"] += 1
        match_results[paper_b]["losses"] += 1
        match_results[paper_a]["points"] += 3
    elif wins_b > wins_a:
        match_results[paper_b]["wins"] += 1
        match_results[paper_a]["losses"] += 1
        match_results[paper_b]["points"] += 3
    else:
        match_results[paper_a]["draws"] += 1
        match_results[paper_b]["draws"] += 1
        match_results[paper_a]["points"] += 1
        match_results[paper_b]["points"] += 1

print("\n===== LEAGUE TABLE =====")
print(f"{'Team':<40} {'W':>3} {'D':>3} {'L':>3} {'Pts':>5}")
print("-" * 60)
ranked = sorted(match_results.items(), key=lambda x: x[1]["points"], reverse=True)
for team, stats in ranked:
    print(f"{os.path.basename(team):<40} {stats['wins']:>3} {stats['draws']:>3} {stats['losses']:>3} {stats['points']:>5}")
