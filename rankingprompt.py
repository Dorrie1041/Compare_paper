import os
from collections import defaultdict
import itertools
import openai
from dotenv import load_dotenv, find_dotenv
import random

_ = load_dotenv(find_dotenv())
openai.api_key = "sk-"
client = openai.OpenAI(api_key=openai.api_key)

def knock_out(paper_dict):
    match_results = defaultdict(lambda: {"wins": 0, "losses": 0, "round_eliminated": 0})
    paper_names = list(paper_dict.keys())

    for paper in paper_names:
        match_results[paper]["round_eliminated"] = 0

    remaining_papers = paper_names.copy()
    random.shuffle(remaining_papers)

    round = 1

    while len(remaining_papers) > 1:
        print(f"\n===== ROUND {round}: {len(remaining_papers)} papers =====")
        next_round = []
        round_matches = list(itertools.zip_longest(*[iter(remaining_papers)] * 2))

        for pair in round_matches:
            paper_a, paper_b = pair

            if paper_b is None:
                next_round.append(paper_a)
                continue

            print(f"\n== Match: {os.path.basename(paper_a)} vs {os.path.basename(paper_b)} ==")
            sections_a = paper_dict[paper_a]
            sections_b = paper_dict[paper_b]
            common_sections = set(sections_a.keys()).intersection(sections_b.keys())

            if not common_sections:
                print("No common sections found. Skipping.")
                continue

            wins_a = 0
            wins_b = 0

            for section in sorted(common_sections):
                prompt = f"""
You are acting as a NeurIPS reviewer. Compare the following **{section}** sections from two papers:

### {section} of Paper A:
{sections_a[section]}

### {section} of Paper B:
{sections_b[section]}

---

For each paper, write a review including:
1. A short **summary** of what the section covers.
2. The section's **strengths and weaknesses**.
3. Assign a score for:
   - **Novelty** (1–10)
   - **Significance** (1–10)
   - **Clarity** (1–10)
4. Provide a **confidence level** in your evaluation (1–5).

Finally, choose which paper is stronger **for this section only**, and explain why.

Respond only with the final decision in this format: `Winner: Paper A` or `Winner: Paper B`.
"""
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=1024
                    )
                    result = response.choices[0].message.content.strip()
                    print(f"[Section: {section.title()}] → {result}")
                    if "Winner: Paper A" in result:
                        wins_a += 1
                    elif "Winner: Paper B" in result:
                        wins_b += 1
                except Exception as e:
                    print(f"Error comparing section '{section}': {e}")

            if wins_a > wins_b:
                print(f"{os.path.basename(paper_a)} wins and advances.")
                match_results[paper_a]["wins"] += 1
                match_results[paper_b]["losses"] += 1
                match_results[paper_b]["round_eliminated"] = round
                next_round.append(paper_a)
            else:
                print(f"{os.path.basename(paper_b)} wins and advances.")
                match_results[paper_b]["wins"] += 1
                match_results[paper_a]["losses"] += 1
                match_results[paper_a]["round_eliminated"] = round
                next_round.append(paper_b)

        remaining_papers = next_round
        round += 1

    champion = remaining_papers[0]
    match_results[champion]["round_eliminated"] = round
    print(f"\n FINAL WINNER: {os.path.basename(champion)}")

    print("\n===== FINAL RANKINGS =====")
    print(f"{'Rank':<5} {'Paper':<40} {'Eliminated In':<15} {'Wins':<5} {'Losses':<6}")
    print("-" * 75)

    ranked = sorted(
        match_results.items(),
        key=lambda x: (-x[1]["round_eliminated"], -x[1]["wins"])
    )

    for rank, (paper, stats) in enumerate(ranked, 1):
        elim_round = stats["round_eliminated"]
        print(f"{rank:<5} {os.path.basename(paper):<40} Round {elim_round:<10} {stats['wins']:<5} {stats['losses']:<6}")






def round_robin(paper_dict):
    match_results = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "points": 0})
    paper_names = list(paper_dict.keys())

    # each paper compare with others paper except itself
    for i, (paper_a, paper_b) in enumerate(itertools.combinations(paper_names, 2), 1):
        print(f"\n===== Match #{i}: {os.path.basename(paper_a)} vs {os.path.basename(paper_b)} =====\n")
        sections_a = paper_dict[paper_a]
        sections_b = paper_dict[paper_b]
        common_sections = set(sections_a.keys()).intersection(sections_b.keys())

        if not common_sections:
            print("No common sections found. Skipping.")
            continue

        wins_a = 0
        wins_b = 0

        for section in sorted(common_sections):
            prompt = f"""
    You are acting as a NeurIPS reviewer. Compare the following **{section}** sections from two papers:

    ### {section} of Paper A:
    {sections_a[section]}

    ### {section} of Paper B:
    {sections_b[section]}

    ---

    For each paper, write a review including:
    1. A short **summary** of what the section covers.
    2. The section's **strengths and weaknesses**.
    3. Assign a score for:
    - **Novelty** (1–10)
    - **Significance** (1–10)
    - **Clarity** (1–10)
    4. Provide a **confidence level** in your evaluation (1–5).

    Finally, choose which paper is stronger **for this section only**, and explain why.

    Respond only with the final decision in this format: `Winner: Paper A` or `Winner: Paper B`.
    """
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1024
                )
                result = response.choices[0].message.content.strip()
                print(f"[Section: {section.title()}] → {result}")
                if "Winner: Paper A" in result:
                    wins_a += 1
                elif "Winner: Paper B" in result:
                    wins_b += 1
            except Exception as e:
                print(f"Error comparing section '{section}': {e}")

        if wins_a > wins_b:
            match_results[paper_a]["wins"] += 1
            match_results[paper_b]["losses"] += 1
            match_results[paper_a]["points"] += 3
        elif wins_b > wins_a:
            match_results[paper_b]["wins"] += 1
            match_results[paper_a]["losses"] += 1
            match_results[paper_b]["points"] += 3
        else:
            match_results[paper_a]["draws"] += 1
            match_results[paper_b]["draws"] += 1
            match_results[paper_a]["points"] += 1
            match_results[paper_b]["points"] += 1

    print("\n===== LEAGUE TABLE =====")
    print(f"{'Team':<40} {'W':>3} {'D':>3} {'L':>3} {'Pts':>5}")
    print("-" * 60)
    ranked = sorted(match_results.items(), key=lambda x: x[1]["points"], reverse=True)
    for team, stats in ranked:
        print(f"{os.path.basename(team):<40} {stats['wins']:>3} {stats['draws']:>3} {stats['losses']:>3} {stats['points']:>5}")
