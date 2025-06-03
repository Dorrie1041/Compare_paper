import yaml
import litellm
import os
import re
import argparse
import time

# === Load disqualification prompts from YAML ===
with open("merge_prompts.yaml", "r", encoding="utf-8") as f:
    full_prompts = yaml.safe_load(f)

prompts = full_prompts

PROMPT_ORDER = [
    (("evaluation_prompt", "evaluation_prompt_full"), "disqualify_result_eval.yaml"),
    (("related_work_prompt", "related_work_prompt_full"), "disqualify_result_related.yaml"),
    (("novelty_prompt", "novelty_prompt_full"), "disqualify_result_novelty.yaml"),
    (("review_only_prompt", "review_only_prompt_full"), "disqualify_result_review.yaml"),
]

def extract_introduction(document_text, abstract_text=""):
    intro_match = re.search(
        r"""
        ^\s*
        (?:<[^>]+>\s*)*
        [#]+\s*
        (?:<[^>]+>\s*)*
        (?:[\*\_]*\s*)*
        (?:[IVXLCDM0-9]+[\.\-\)]?\s+)?
        INTRODUCTION
        (?:\s*[\*\_]*)*
        [^\n\S]*[\r\n]+
        (.*?)
        (?=^\s*(?:<[^>]+>\s*)*[#]+|^\s*\*\*?[A-Z]|\Z)
        """,
        document_text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL | re.VERBOSE
    )

    if intro_match:
        return intro_match.group(1).strip()

    abstract_text = abstract_text.strip()
    if abstract_text:
        norm_doc = re.sub(r'\s+', ' ', document_text)
        norm_abstract = re.sub(r'\s+', ' ', abstract_text)
        abstract_pos = norm_doc.find(norm_abstract)
        if abstract_pos != -1:
            raw_pos = document_text.lower().find(abstract_text[:30].lower())
            content_after = document_text[raw_pos + len(abstract_text):]
            section_match = re.search(
                r"^#{1,6}.*?\n+(.*?)(?=^#{1,6}|\Z)", 
                content_after,
                flags=re.DOTALL | re.MULTILINE
            )
            if section_match:
                return section_match.group(1).strip()

    return ""

def is_disqualified(paper, prompt_text, prompt_key=None, use_full_text=False):
    abstract_text = paper.get("abstract", "").strip()
    document_text = paper.get("document", "")

    if use_full_text:
        markdown_text = document_text
        scope_desc = "Full paper"
    else:
        if not abstract_text:
            abstract_match = re.search(r"(?i)^#{1,3}\s*abstract\s*\n+(.*?)(?=^#{1,3}\s|\Z)", document_text, flags=re.DOTALL | re.MULTILINE)
            if abstract_match:
                abstract_text = abstract_match.group(1).strip()
            else:
                fallback_match = re.search(r"(?<=\n\n)abstract\.\s+(.*?)(?=\n\n|#{1,3}\s|\Z)", document_text, flags=re.IGNORECASE | re.DOTALL)
                if fallback_match:
                    abstract_text = fallback_match.group(1).strip()

        intro_text = extract_introduction(document_text, abstract_text)

        markdown_text = ""
        if abstract_text:
            markdown_text += f"## Abstract\n{abstract_text}\n\n"
        if intro_text:
            markdown_text += f"## Introduction\n{intro_text}"

        if not markdown_text:
            return "Disqualified: No abstract or introduction found."

        scope_desc = "Abstract + Introduction"

    prompt = f"""{prompt_text}

Here is the paper content ({scope_desc}):

\"\"\"{markdown_text}\"\"\"

Answer using one of:
- Qualified. Reason: <brief explanation>
- Disqualified: <reason>. Reason: <brief explanation>
"""
    start_time = time.time()
    response = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200
    )
    elapsed = time.time() - start_time
    tokens_used = response.get("usage", {}).get("total_tokens", 0)

    paper.setdefault("token_usage", 0)
    paper["token_usage"] += tokens_used if isinstance(tokens_used, int) else 0

    paper.setdefault("time_usage", 0.0)
    paper["time_usage"] += elapsed

    return response["choices"][0]["message"]["content"].strip().strip('"').strip("'")

# === Detect is EN ===
def is_english(text):
    ascii_letters = sum(c.isascii() and c.isalpha() for c in text)
    total_letters = sum(c.isalpha() for c in text)

    if total_letters == 0:
        return False

    ratio = ascii_letters / total_letters

    # If more than 80% of letters are ASCII, treat as English
    return ratio >= 0.8

def run_all_checks(papers):
    for paper in papers:
        if "decisions" not in paper:
            paper["decisions"] = {}

        document_text = paper.get("document", "")
        if not is_english(document_text):
            paper["decisions"]["language"] = "Disqualified: Not English. Reason: Paper is not primarily written in English."
            print(f"[LANGUAGE] {paper['title']} → Disqualified (not English)")
            continue
        else:
            paper["decisions"]["language"] = "- Qualified. Reason: English Paper"

        for (short_key, full_key), _ in PROMPT_ORDER:
            short_prompt = prompts[short_key]
            full_prompt = prompts[full_key]

            try:
                result = is_disqualified(paper, short_prompt, short_key, use_full_text=False)
                print(f"[CHECK] {paper['title']} → {short_key}: {result}")
                normalized = result.lower().lstrip("-: ").strip()
                if not normalized.startswith("qualified"):
                    result = is_disqualified(paper, full_prompt, full_key, use_full_text=True)
                    print(f"[FALLBACK] {paper['title']} → {full_key}: {result}")
                paper["decisions"][short_key] = result
            except Exception as e:
                error_msg = f"ERROR: {str(e)}"
                print(f"[ERROR] {paper['title']} → {short_key}: {error_msg}")
                paper["decisions"][short_key] = error_msg

    return papers

def is_fully_qualified(paper):
    for (short_key, _), _ in PROMPT_ORDER:
        decision = paper.get("decisions", {}).get(short_key, "")
        cleaned = decision.lower().lstrip("-• ").strip()
        if not cleaned.startswith("qualified"):
            return False
    return True

def main(input_yaml):
    qualified_output_yaml = "qualified_papers.yaml"
    disqualified_output_yaml = "disqualified_papers.yaml"

    with open(input_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    papers = data["papers"]
    papers = run_all_checks(papers)

    qualified = [p for p in papers if is_fully_qualified(p)]
    disqualified = [p for p in papers if not is_fully_qualified(p)]

    def minimal(p):
        return {
            "title": p.get("title", ""),
            "id": p.get("id", ""),
            "decisions": p.get("decisions", {}),
            "token_usage": p.get("token_usage", 0),
            "time_usage": round(p.get("time_usage", 0.0), 2)
        }

    with open(qualified_output_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"papers": [minimal(p) for p in qualified]}, f, allow_unicode=True, sort_keys=False)

    with open(disqualified_output_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"papers": [minimal(p) for p in disqualified]}, f, allow_unicode=True, sort_keys=False)

    print(f"\n🎉 Qualified papers saved to {qualified_output_yaml}")
    print(f"❌ Disqualified papers (with reasons) saved to {disqualified_output_yaml}")

    token_vals = [p.get("token_usage", 0) for p in papers]
    time_vals = [p.get("time_usage", 0.0) for p in papers]

    if token_vals:
        print(f"\n📊 Avg Total Tokens per Paper: {sum(token_vals) / len(token_vals):.1f}")
    if time_vals:
        print(f"⏱  Avg Total Time per Paper: {sum(time_vals) / len(time_vals):.2f}s")

    for _, filename in PROMPT_ORDER:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"🗑 Deleted: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter papers by qualification status.")
    parser.add_argument("input_yaml", help="YAML file containing papers to filter")
    args = parser.parse_args()
    main(args.input_yaml)
