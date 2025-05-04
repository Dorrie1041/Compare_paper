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

\"\"\"{markdown_text[:8000]}\"\"\"  # Trim for token limits

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

def filter_papers(papers, prompt_key, result_yaml):
    qualified_papers = []
    decision_log = []

    prompt_text = prompts[prompt_key]

    for paper in papers:
        try:
            decision = is_disqualified(paper, prompt_text)
            print(f"[CHECK] {paper['title']} → {decision}")

            clean_decision = decision.strip('"').strip("'").strip().lower()

            # === Record decision into the paper data itself
            if "decisions" not in paper:
                paper["decisions"] = {}
            paper["decisions"][prompt_key] = decision

            decision_log.append({
                "title": paper['title'],
                "decision": decision
            })

            if clean_decision.startswith("qualified"):
                qualified_papers.append(paper)
                print(f"[✓] Passed: {paper['title']}")
            else:
                print(f"[SKIP] Disqualified: {decision}")

        except Exception as e:
            print(f"[ERROR] Failed to check paper '{paper['title']}': {e}")
            decision = f"ERROR: {str(e)}"

            if "decisions" not in paper:
                paper["decisions"] = {}
            paper["decisions"][prompt_key] = decision

            decision_log.append({
                "title": paper['title'],
                "decision": decision
            })

    with open(result_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"results": decision_log}, f, allow_unicode=True, sort_keys=False)

    return qualified_papers

def main():
    input_yaml = "papers.yaml"
    final_output_yaml = "qualified_papers.yaml"

    with open(input_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    current_papers = data["papers"]

    for prompt_key, result_yaml in PROMPT_ORDER:
        print(f"\n🔎 Running check: {prompt_key}")
        current_papers = filter_papers(current_papers, prompt_key, result_yaml)
        print(f"✅ Remaining papers after {prompt_key}: {len(current_papers)}")

    with open(final_output_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"papers": current_papers}, f, allow_unicode=True, sort_keys=False)

    print(f"\n🎉 Final qualified papers saved to {final_output_yaml}")

    # === Clean up intermediate decision files ===
    for _, result_yaml in PROMPT_ORDER:
        if os.path.exists(result_yaml):
            os.remove(result_yaml)
            print(f"🗑 Deleted temporary file: {result_yaml}")

if __name__ == "__main__":
    main()