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

def filter_papers(papers, prompt_key):
    qualified_papers = []

    prompt_text = prompts[prompt_key]

    for paper in papers:
        try:
            decision = is_disqualified(paper, prompt_text)
            print(f"[CHECK] {paper['title']} → {decision}")

            clean_decision = decision.strip('"').strip("'").strip().lower()

            # Save the decision into the paper data
            if "decisions" not in paper:
                paper["decisions"] = {}
            paper["decisions"][prompt_key] = decision  # Save exact reply

            if clean_decision.startswith("qualified"):
                qualified_papers.append(paper)
                print(f"[✓] Passed: {paper['title']}")
            else:
                print(f"[SKIP] Disqualified at {prompt_key}: {decision}")

        except Exception as e:
            print(f"[ERROR] Failed to check paper '{paper['title']}': {e}")
            decision = f"ERROR: {str(e)}"

            if "decisions" not in paper:
                paper["decisions"] = {}
            paper["decisions"][prompt_key] = decision

    return qualified_papers

def main():
    input_yaml = "papers.yaml"
    qualified_output_yaml = "qualified_papers.yaml"
    disqualified_output_yaml = "disqualified_papers.yaml"

    with open(input_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    current_papers = data["papers"]

    for prompt_key, _ in PROMPT_ORDER:
        print(f"\n🔎 Running check: {prompt_key}")
        current_papers = filter_papers(current_papers, prompt_key)
        print(f"✅ Remaining papers after {prompt_key}: {len(current_papers)}")

    # --- Split into qualified and disqualified papers ---
    qualified_papers = current_papers
    disqualified_papers = []

    for paper in data["papers"]:
        if paper in qualified_papers:
            continue  # Already qualified
        disqualified_papers.append(paper)

    # --- Save final outputs ---
    with open(qualified_output_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"papers": qualified_papers}, f, allow_unicode=True, sort_keys=False)

    with open(disqualified_output_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"papers": disqualified_papers}, f, allow_unicode=True, sort_keys=False)

    print(f"\n🎉 Qualified papers saved to {qualified_output_yaml}")
    print(f"❌ Disqualified papers (with reasons) saved to {disqualified_output_yaml}")

if __name__ == "__main__":
    main()