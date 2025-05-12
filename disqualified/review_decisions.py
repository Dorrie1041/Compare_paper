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

    rule_block = f"""
Decision rules the prior judgments should have followed:

[EVALUATION RULE]
{prompts["evaluation_prompt"]}

[RELATED WORK RULE]
{prompts["related_work_prompt"]}

[NOVELTY RULE]
{prompts["novelty_prompt"]}

[REVIEW-ONLY RULE]
{prompts["review_only_prompt"]}
"""

    def build_prompt(content, chunk_info="full document"):
        return f"""{prompts["verification_prompt"]}

{rule_block}

Title: {title}

Prior decisions:
{prior}

Markdown content ({chunk_info}):
\"\"\"{content}\"\"\"

Respond only with:
- Correct: <short comment>
- Incorrect: <short reason>
"""

    try:
        prompt = build_prompt(document)
        response = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        return response["choices"][0]["message"]["content"].strip()
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