import yaml
import litellm

# Load verification prompt
with open("prompt.yaml", "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

verification_prompt = prompts["verification_prompt"]

def verify_paper(paper):
    title = paper.get("title", "")
    decisions = paper.get("decisions", {})
    document = paper.get("document", "")

    prior = "\n".join([f"- {k}: {v}" for k, v in decisions.items()])

    prompt = f"""{verification_prompt}

Title: {title}

Prior decisions:
{prior}

Markdown content (truncated to 8000 chars):
\"\"\"{document[:8000]}\"\"\"

Respond only with:
- Correct: <comment>
- Incorrect: <reason>
"""

    try:
        response = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200
        )
        reply = response["choices"][0]["message"]["content"].strip()
        return reply
    except Exception as e:
        return f"ERROR: {e}"

def main():
    input_yaml = "qualified_papers.yaml"
    disqualified_yaml = "disqualified_papers.yaml"
    output_yaml = "verified_decision_log.yaml"

    papers = []

    for fname in [input_yaml, disqualified_yaml]:
        with open(fname, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            papers.extend(data.get("papers", []))

    verified_results = []

    for paper in papers:
        result = verify_paper(paper)
        print(f"[✓] {paper['title']} → {result}")
        verified_results.append({
            "title": paper.get("title", ""),
            "verification": result,
            "decisions": paper.get("decisions", {})
        })

    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"verified": verified_results}, f, allow_unicode=True, sort_keys=False)

    print(f"\n✅ Verification results saved to {output_yaml}")

if __name__ == "__main__":
    main()