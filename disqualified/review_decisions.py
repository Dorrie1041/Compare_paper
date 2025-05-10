import yaml
import litellm

REVIEW_PROMPT_TEMPLATE = """
A paper was evaluated under the category: "{category}".

Here is the paper content (truncated to fit limits):
\"\"\"{document}\"\"\"

The decision made was:
> {decision}

Now, assess whether the decision is reasonable. Only respond with one of the two formats:
- Correct: <short justification>
- Incorrect: <short justification>

Be concise. No extra explanation.
"""

PROMPT_ORDER = [
    ("evaluation_prompt", "Evaluation of general quality"),
    ("related_work_prompt", "Assessment of related work and context"),
    ("novelty_prompt", "Assessment of novelty and originality"),
    ("review_only_prompt", "Evaluation based only on review information"),
]

def reverse_check(paper, prompt_key, category_name):
    decision = paper.get("decisions", {}).get(prompt_key, "No decision recorded")
    truncated_doc = paper.get("document", "")[:8000]

    prompt = REVIEW_PROMPT_TEMPLATE.format(
        category=category_name,
        document=truncated_doc,
        decision=decision
    )

    response = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150
    )

    reply = response["choices"][0]["message"]["content"].strip()
    return reply

def main():
    input_yaml = "all_papers_with_reasons.yaml"
    output_yaml = "review_of_decisions.yaml"

    with open(input_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    papers = data["papers"]

    for paper in papers:
        review_results = {}
        for prompt_key, category_name in PROMPT_ORDER:
            try:
                review = reverse_check(paper, prompt_key, category_name)
                print(f"[✓] {paper['title']} – {prompt_key}: {review}")
                review_results[prompt_key] = review
            except Exception as e:
                print(f"[ERROR] Failed on {paper['title']} – {prompt_key}: {e}")
                review_results[prompt_key] = f"ERROR: {e}"

        paper["reverse_review"] = review_results

    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"papers": papers}, f, allow_unicode=True, sort_keys=False)

    print(f"\n📄 Review results saved to {output_yaml}")

if __name__ == "__main__":
    main()