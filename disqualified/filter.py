import yaml
import litellm

# === Load disqualification prompt from YAML ===
with open("prompt.yaml", "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

DISQUALIFICATION_PROMPT = prompts["disqualification_prompt"]

def is_disqualified(paper):
    markdown_text = paper["document"]

    prompt = f"""{DISQUALIFICATION_PROMPT}

Here is the paper content (in Markdown):

\"\"\"{markdown_text[:8000]}\"\"\"  # Trim for token limits

Answer with either:
- Disqualified: <reason>
- Qualified
"""

    response = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200
    )

    reply = response["choices"][0]["message"]["content"].strip()

    # <<< FIX: remove extra quotation marks if the model adds them
    reply = reply.strip('"').strip("'").strip()

    return reply

def main():
    input_yaml = "papers.yaml"
    output_yaml = "qualified_papers.yaml"
    result_yaml = "disqualify_result.yaml"

    with open(input_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    qualified_papers = []
    decision_log = []

    for paper in data["papers"]:
        try:
            decision = is_disqualified(paper)
            print(f"[CHECK] {paper['title']} → {decision}")

            clean_decision = decision.strip('"').strip("'").strip().lower()

            if clean_decision.startswith("qualified"):
                qualified_papers.append(paper)
                print(f"[✓] Added: {paper['title']}")
            else:
                print(f"[SKIP] Disqualified: {decision}")

        except Exception as e:
            print(f"[ERROR] Failed to check paper '{paper['title']}': {e}")
            decision_log.append({
                "title": paper['title'],
                "decision": f"ERROR: {str(e)}"
            })

    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"papers": qualified_papers}, f, allow_unicode=True, sort_keys=False)
    
    with open(result_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"results": decision_log}, f, allow_unicode=True, sort_keys=False)

    print(f"\n✅ Done! Qualified papers saved to {output_yaml}")
    print(f"📄 Decisions log saved to {result_yaml}")

if __name__ == "__main__":
    main()