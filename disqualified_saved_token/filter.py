import yaml
import litellm
import os
import re
import argparse

# === Load disqualification prompts from YAML ===
with open("prompt.yaml", "r", encoding="utf-8") as f:
    full_prompts = yaml.safe_load(f)

prompts = {k: v for k, v in full_prompts.items() if k != "verification_prompt"}

PROMPT_ORDER = [
    ("evaluation_prompt", "disqualify_result_eval.yaml"),
    ("related_work_prompt", "disqualify_result_related.yaml"),
    ("novelty_prompt", "disqualify_result_novelty.yaml"),
    ("review_only_prompt", "disqualify_result_review.yaml"),
]

def extract_introduction(document_text):
    intro_match = re.search(
        r"""
        ^\s*                                              # Optional leading whitespace
        (?:<[^>]+>\s*)*                                   # Optional inline HTML like <span>
        [#]+\s*                                           # Markdown heading ##, ### etc.
        (?:<[^>]+>\s*)*                                   # More inline HTML if needed
        (?:[\*\_]*\s*)*                                   # Optional markdown styling
        (?:[IVXLCDM0-9]+[\.\-\)]?\s+)?                    # Optional section number (1, I., etc.)
        INTRODUCTION                                      # The word 'INTRODUCTION'
        (?:\s*[\*\_]*)*                                   # Optional styling after
        [^\n\S]*[\r\n]+                                   # Line break after heading
        (.*?)                                             # Capture everything after intro heading
        (?=                                               
            ^\s*                                          # Start of line
            (?:<[^>]+>\s*)*                               # Optional HTML span
            [#]+                                          # Next heading
            |
            ^\s*\*\*?[A-Z]                                # Or something like "**A"
            |
            \Z                                            # Or end of file
        )
        """,
        document_text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL | re.VERBOSE
    )
    return intro_match.group(1).strip() if intro_match else ""

def is_disqualified(paper, prompt_text, prompt_key=None):
    abstract_text = paper.get("abstract", "").strip()
    document_text = paper.get("document", "")

    if not abstract_text:
        abstract_match = re.search(
            r"(?i)^#{1,3}\s*abstract\s*\n+(.*?)(?=^#{1,3}\s|\Z)", 
            document_text, 
            flags=re.DOTALL | re.MULTILINE
        )
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
        else:
            fallback_match = re.search(
                r"(?<=\n\n)abstract\.\s+(.*?)(?=\n\n|#{1,3}\s|\Z)", 
                document_text, 
                flags=re.IGNORECASE | re.DOTALL
            )
            if fallback_match:
                abstract_text = fallback_match.group(1).strip()
                
    intro_text = extract_introduction(paper.get("document", ""))

    markdown_text = ""
    if abstract_text:
        markdown_text += f"## Abstract\n{abstract_text}\n\n"
    if intro_text:
        markdown_text += f"## Introduction\n{intro_text}"

    if not markdown_text:
        return "Disqualified: No abstract or introduction found."
    
    paper["llm_input_used"] = markdown_text
    
    prompt = f"""{prompt_text}

Here is the paper content (Abstract + Introduction only):

\"\"\"{markdown_text}\"\"\"

Answer using one of:
- Qualified. Reason: <brief explanation>
- Disqualified: <reason>. Reason: <brief explanation>
"""
    response = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200
    )
    return response["choices"][0]["message"]["content"].strip().strip('"').strip("'")

def run_all_checks(papers):
    for paper in papers:
        if "decisions" not in paper:
            paper["decisions"] = {}

        for prompt_key, _ in PROMPT_ORDER:
            prompt_text = prompts[prompt_key]
            try:
                decision = is_disqualified(paper, prompt_text)
                print(f"[CHECK] {paper['title']} → {prompt_key}: {decision}")
                paper["decisions"][prompt_key] = decision
            except Exception as e:
                error_msg = f"ERROR: {str(e)}"
                print(f"[ERROR] {paper['title']} → {prompt_key}: {error_msg}")
                paper["decisions"][prompt_key] = error_msg

    return papers

def is_fully_qualified(paper):
    for prompt_key, _ in PROMPT_ORDER:
        decision = paper.get("decisions", {}).get(prompt_key, "")
        cleaned = decision.lower().lstrip("-• ").strip()
        if not cleaned.startswith("qualified"):
            return False
    return True

def main(input_yaml):
    qualified_output_yaml = "qualified_papers.yaml"
    disqualified_output_yaml = "disqualified_papers.yaml"
    full_log_yaml = "all_papers_with_reasons.yaml"

    with open(input_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    papers = data["papers"]
    papers = run_all_checks(papers)

    qualified = [p for p in papers if is_fully_qualified(p)]
    disqualified = [p for p in papers if not is_fully_qualified(p)]

    with open(qualified_output_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"papers": qualified}, f, allow_unicode=True, sort_keys=False)

    with open(disqualified_output_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"papers": disqualified}, f, allow_unicode=True, sort_keys=False)

    with open(full_log_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"papers": papers}, f, allow_unicode=True, sort_keys=False)

    print(f"\n🎉 Qualified papers saved to {qualified_output_yaml}")
    print(f"❌ Disqualified papers (with reasons) saved to {disqualified_output_yaml}")
    print(f"📋 All decision logs saved to {full_log_yaml}")

    for _, filename in PROMPT_ORDER:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"🗑 Deleted: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter papers by qualification status.")
    parser.add_argument("input_yaml", help="YAML file containing papers to filter")
    args = parser.parse_args()

    main(args.input_yaml)