import yaml
import litellm

# === Load disqualification prompt from YAML ===
with open("prompt.yaml", "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

DISQUALIFICATION_PROMPT = prompts["review_only_prompt"]

def is_disqualified(paper):
    markdown_text = paper["document"]

    prompt = f"""{DISQUALIFICATION_PROMPT}

Here is the paper content (in Markdown):

\"\"\"{markdown_text[:8000]}\"\"\"  # Trim for token limits

Answer with either:
- "Disqualified: <reason>"
- "Qualified"
"""

    response = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200
    )

    return response["choices"][0]["message"]["content"].strip()

def main():
    input_yaml = "papers.yaml"
    output_yaml = "qualified_papers.yaml"

    with open(input_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    qualified_papers = []

    for paper in data["papers"]:
        try:
            decision = is_disqualified(paper)
            print(f"[CHECK] {paper['title']} → {decision}")

            if decision.lower().startswith("qualified"):
                qualified_papers.append(paper)
            else:
                print(f"[SKIP] Disqualified: {decision}")

        except Exception as e:
            print(f"[ERROR] Failed to check paper '{paper['title']}': {e}")

    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"papers": qualified_papers}, f, allow_unicode=True, sort_keys=False)

    print(f"\n✅ Done! Qualified papers saved to {output_yaml}")

if __name__ == "__main__":
    main()