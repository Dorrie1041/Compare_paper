import sys
import openai
from dotenv import load_dotenv, find_dotenv
import re
import yaml

# Load .env and OpenAI API key
_ = load_dotenv(find_dotenv())
openai.api_key = "sk-"
client = openai.OpenAI(api_key=openai.api_key)


# === Load YAML Papers ===
def load_papers_from_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("papers", [])

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

def process_yaml(yaml_path):
    all_papers = load_papers_from_yaml(yaml_path)
    qualified_papers = {}
    for paper in all_papers:
        title = paper.get("title", "Untitled")
        document = paper.get("document", "")
        print(f"\n[Checking] {title[:60]}...")
        result = is_disqualified(document)
        print(f"→ {result}")
        if result.lower().startswith("qualified"):
            qualified_papers[title] = document
    return qualified_papers

# Ask OpenAI to determine if the paper should be disqualified
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

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    result = response.choices[0].message.content.strip()
    return result

# Example usage
if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: python3 paperRanking.py [knock|round] papers.yaml")
        sys.exit(1)

    run_mode = None
    if len(sys.argv) == 3:
        mode = sys.argv[1].lower()
        yaml_path = sys.argv[2]
        if mode in ["knock", "round"]:
            run_mode = mode
        else:
            print("Invalid mode. Use 'knock' or 'round'.")
            sys.exit(1)
    else:
        yaml_path = sys.argv[1]

    qualified = process_yaml(yaml_path)
    print(f"\n✅ {len(qualified)} papers qualified out of {len(load_papers_from_yaml(yaml_path))}")

    if run_mode:
        qualified_sections = {title: extract_sections(md) for title, md in qualified.items()}
        if run_mode == "knock":
            from rankingprompt import knock_out
            knock_out(qualified_sections)
        elif run_mode == "round":
            from rankingprompt import round_robin
            round_robin(qualified_sections)  
