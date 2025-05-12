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


def split_by_section(markdown_text):
    section_headers = list(re.finditer(r"^#{2,3} .*", markdown_text, flags=re.MULTILINE))
    if not section_headers:
        return [markdown_text]

    sections = []
    for i in range(len(section_headers)):
        start = section_headers[i].start()
        end = section_headers[i + 1].start() if i + 1 < len(section_headers) else len(markdown_text)
        sections.append(markdown_text[start:end])
    return sections

def is_disqualified(paper, prompt_text, prompt_key=None):
    markdown_text = paper["document"]

    if len(markdown_text) <= 100_000:
        # Full context if size allows
        content_to_use = markdown_text
        prompt = f"""{prompt_text}

Here is the paper content (in Markdown):

\"\"\"{content_to_use}\"\"\"

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

    # Too long: split by section and evaluate ALL chunks
    sections = split_by_section(markdown_text)
    all_replies = []
    disqualified_chunks = []

    for idx, section in enumerate(sections):
        section_prompt = f"""{prompt_text}

Here is a section from the paper (chunk {idx + 1}):

\"\"\"{section[:8000]}\"\"\"

Answer using one of:
- Qualified. Reason: <brief explanation>
- Disqualified: <reason>. Reason: <brief explanation>
"""
        response = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": section_prompt}],
            temperature=0.2,
            max_tokens=200
        )
        reply = response["choices"][0]["message"]["content"].strip().strip('"').strip("'")
        all_replies.append(f"Chunk {idx + 1}: {reply}")
        if reply.lower().startswith("disqualified"):
            disqualified_chunks.append(f"Chunk {idx + 1}: {reply}")

    if disqualified_chunks:
        return "Disqualified: " + " | ".join(disqualified_chunks)
    return "Qualified. Reason: All relevant sections passed."

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