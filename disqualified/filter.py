import yaml
import litellm
import os

# === Load disqualification prompts from YAML ===
with open("prompt.yaml", "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

PROMPT_ORDER = [
    ("evaluation_prompt", "disqualify_result_eval.yaml"),
    ("related_work_prompt", "disqualify_result_related.yaml"),
    ("novelty_prompt", "disqualify_result_novelty.yaml"),
    ("review_only_prompt", "disqualify_result_review.yaml"),
]

def is_disqualified(paper, prompt_text):
    markdown_text = paper["document"]

    prompt = f"""{prompt_text}

Here is the paper content (in Markdown):

\"\"\"{markdown_text[:8000]}\"\"\"

Answer with either:
- Disqualified: <reason>
- Qualified

Do not explain your answer. Reply with only one of the two options.
"""

    response = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200
    )

    reply = response["choices"][0]["message"]["content"].strip()
    reply = reply.strip('"').strip("'").strip()
    return reply

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
        decision = paper.get("decisions", {}).get(prompt_key, "").lower()
        if not decision.startswith("qualified"):
            return False
    return True

def main():
    input_yaml = "papers.yaml"
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

    # --- Clean up temporary disqualification YAML files ---
    for _, filename in PROMPT_ORDER:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"🗑 Deleted: {filename}")

if __name__ == "__main__":
    main()