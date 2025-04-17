import sys
import os
import time
import random
from pdfminer.high_level import extract_text
from markdownify import markdownify as md
import re
from collections import defaultdict
import itertools
from litellm import completion, completion_cost
import matplotlib.pyplot as plt

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = ""
output_dir = "C:/Users/chloe/Downloads/Compare_paper"
result_txt_path = os.path.join(output_dir, "test_result_cache.txt")
chart_path = os.path.join(output_dir, "test_result_chart_cache.png")
LOG_PIPELINE = os.path.join(output_dir, "test_pipeline_log.txt")
os.makedirs(output_dir, exist_ok=True)

# === UPDATED SYSTEM PROMPT ===
SYSTEM_PROMPT = (
    "You are an expert academic reviewer comparing research paper sections.\n"
    "For each pair of sections, evaluate them solely on the following criteria:\n"
    "   1. Novelty (score: +3 if innovative, -1 if not innovative)\n"
    "   2. Technical rigor (score: +2 if very rigorous, -1 if lacking rigor)\n"
    "   3. Clarity (score: +1 if clear, -1 if unclear)\n\n"
    "For each section comparison, please provide a detailed explanation covering the following:\n"
    "   - List the strengths and weaknesses for each paper's section on each criterion.\n"
    "   - Provide a breakdown of the scores for each criterion for each paper.\n"
    "   - Sum the scores to decide which paper wins that section.\n\n"
    "At the very end of your response, output EXACTLY one of the following strings on a new line (with no extra text):\n"
    "   Winner: Paper A\n"
    "   Winner: Paper B\n"
    "   Draw\n"
)

# === PDF -> Markdown & Section Extraction ===
def convert_pdf_to_markdown(pdf_path):
    try:
        print(f"[DEBUG] Extracting text from: {pdf_path}")
        text = extract_text(pdf_path)
        return md(text)
    except Exception as e:
        print(f"[ERROR] Failed processing PDF {pdf_path}: {e}")
        return None


def extract_sections(markdown_text):
    if not markdown_text:
        return {}
    pattern = r'(?m)^(?P<header>([A-Z][A-Z\s\-]+|[0-9]+[\.\d]*\s+[A-Z][A-Z\s\-]+))$'
    matches = list(re.finditer(pattern, markdown_text))
    secs = {}
    for i, m in enumerate(matches):
        title = m.group('header').strip().lower()
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(markdown_text)
        content = markdown_text[start:end].strip()
        if content:
            secs[title] = content
    return secs

# === LLM Comparison Helpers ===
def standardize_result(resp):
    if not resp:
        return None
    low = resp.lower()
    if 'winner: paper a' in low:
        return 'Winner: Paper A'
    if 'winner: paper b' in low:
        return 'Winner: Paper B'
    if 'draw' in low:
        return 'Draw'
    return None


def compare_with_caching(a, b, sec, limit=2000):
    """
    Compare two sections using LiteLLM API with correct OpenAI‐style messages and increased max_tokens.
    """
    for lim in (limit, limit // 2):
        try:
            print(f"[DEBUG] (limit={lim}) Comparing section '{sec}'...")
            # OpenAI-style chat messages
            msg_sys = {'role': 'system', 'content': SYSTEM_PROMPT}
            prompt_text = f"""Compare these {sec} sections:

### Paper A:
{a[:lim]}

### Paper B:
{b[:lim]}"""
            msg_usr = {'role': 'user', 'content': prompt_text}

            # Call completion without custom cache, with higher max_tokens
            response = completion(
                model='gpt-4o',
                messages=[msg_sys, msg_usr],
                temperature=0.0,
                max_tokens=1000
            )

            # Validate response
            if not hasattr(response, 'choices') or not response.choices:
                print(f"[ERROR] No choices received for section '{sec}'")
                continue
            result = response.choices[0].message.content.strip()
            print(f"[DEBUG] API returned result: {result[:100]}...")

            # Log token usage if available
            if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens'):
                print(f"[DEBUG] Tokens used: {response.usage.prompt_tokens}")

            return result
        except Exception as e:
            print(f"[ERROR] (limit={lim}) API error for section '{sec}': {e}")
    print(f"[ERROR] All attempts failed for section '{sec}'")
    return None

# === TOURNAMENT PIPELINE UTILITIES ===
def classification_round(papers, cmp, iters=4):
    scores = {p: 0 for p in papers}
    details = []
    for _ in range(iters):
        random.shuffle(papers)
        for a, b in zip(papers[0::2], papers[1::2]):
            r = cmp(a, b)
            scores[a] += 3 if r == 'A' else 1 if r == 'Draw' else 0
            scores[b] += 3 if r == 'B' else 1 if r == 'Draw' else 0
            details.append(f"Classify {os.path.basename(a)} vs {os.path.basename(b)} -> {r}")
    return sorted(papers, key=lambda x: scores[x], reverse=True), scores, details


def group_stage(sorted_papers, cmp, size=4):
    quals = []
    details = []
    for i in range(0, len(sorted_papers), size):
        grp = sorted_papers[i:i+size]
        gs = {p: 0 for p in grp}
        for x in range(len(grp)):
            for y in range(x+1, len(grp)):
                a, b = grp[x], grp[y]
                r = cmp(a, b)
                gs[a] += 3 if r == 'A' else 1 if r == 'Draw' else 0
                gs[b] += 3 if r == 'B' else 1 if r == 'Draw' else 0
                details.append(f"Group {os.path.basename(a)} vs {os.path.basename(b)} -> {r}")
        top2 = sorted(grp, key=lambda p: gs[p], reverse=True)[:2]
        quals.extend(top2)
    return quals, details


def knockout_stage(quals, cmp):
    bracket = quals[:]
    details = []
    random.shuffle(bracket)
    while len(bracket) > 1:
        next_round = []
        for a, b in zip(bracket[0::2], bracket[1::2]):
            r = cmp(a, b)
            winner = a if r == 'A' else b if r == 'B' else random.choice([a, b])
            details.append(f"Knockout {os.path.basename(a)} vs {os.path.basename(b)} -> {os.path.basename(winner)}")
            next_round.append(winner)
        bracket = next_round
    return bracket[0], details

# === MAIN EXECUTION ===
if __name__ == '__main__':
    with open(result_txt_path, 'w') as rf, open(LOG_PIPELINE, 'w') as lf:
        rf.write("===== PAPER COMPARISON RESULTS =====\n\n")

    if len(sys.argv) != 2:
        print("Usage: python comp_paper.py papers.txt")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        pdf_files = [l.strip() for l in f if l.strip()]

    papers = {}
    for pdf in pdf_files:
        md_text = convert_pdf_to_markdown(pdf)
        if md_text:
            secs = extract_sections(md_text)
            if secs:
                papers[pdf] = secs

    if not papers:
        print("[ERROR] No papers processed successfully.")
        sys.exit(1)

    def cmp_p(a, b):
        wa = wb = 0
        common = set(papers[a]) & set(papers[b])
        for sec in common:
            raw = compare_with_caching(papers[a][sec], papers[b][sec], sec)
            res = standardize_result(raw)
            if res == 'Winner: Paper A': wa += 1
            elif res == 'Winner: Paper B': wb += 1
        return 'A' if wa > wb else 'B' if wb > wa else 'Draw'

    paper_list = list(papers.keys())
    sorted_p, class_scores, class_details = classification_round(paper_list, cmp_p)
    print("Classification ranking:", sorted_p)
    for d in class_details: print(d)
    with open(LOG_PIPELINE, 'a') as lf:
        lf.write("CLASSIFICATION\n" + "\n".join(class_details) + "\n")

    qualifiers, group_details = group_stage(sorted_p, cmp_p)
    print("Qualified for knockout:", qualifiers)
    for d in group_details: print(d)
    with open(LOG_PIPELINE, 'a') as lf:
        lf.write("GROUP STAGE\n" + "\n".join(group_details) + "\n")

    champion, knock_details = knockout_stage(qualifiers, cmp_p)
    print("Tournament champion:", champion)
    for d in knock_details: print(d)
    with open(LOG_PIPELINE, 'a') as lf:
        lf.write("KNOCKOUT\n" + "\n".join(knock_details) + "\n")

    results = defaultdict(lambda: {'wins':0,'draws':0,'losses':0,'points':0})
    detailed_log = []
    for i, (pa, pb) in enumerate(itertools.combinations(papers.keys(), 2), 1):
        header = f"===== Match #{i}: {os.path.basename(pa)} vs {os.path.basename(pb)} ====="
        print(header)
        detailed_log.append(header)

        common_secs = set(papers[pa]) & set(papers[pb])
        if not common_secs:
            warning = "[WARNING] No common sections found between these papers."
            print(warning)
            detailed_log.append(warning)
            continue

        wins_a = wins_b = 0
        for sec in common_secs:
            print(f"[DEBUG] Comparing section: '{sec}'")
            raw_res = compare_with_caching(papers[pa][sec], papers[pb][sec], sec)
            std = standardize_result(raw_res) or "Error"
            detailed_log.append(f"[RESULT] [{sec}] -> {std}")
            if std == 'Winner: Paper A': wins_a += 1
            elif std == 'Winner: Paper B': wins_b += 1
            time.sleep(1)

        if wins_a > wins_b:
            match_res = f"[INFO] Winner for this match: {os.path.basename(pa)}"
            results[pa]['wins'] += 1
            results[pb]['losses'] += 1
            results[pa]['points'] += 3
        elif wins_b > wins_a:
            match_res = f"[INFO] Winner for this match: {os.path.basename(pb)}"
            results[pb]['wins'] += 1
            results[pa]['losses'] += 1
            results[pb]['points'] += 3
        else:
            match_res = "[INFO] This match resulted in a draw."
            results[pa]['draws'] += 1
            results[pb]['draws'] += 1
            results[pa]['points'] += 1
            results[pb]['points'] += 1

        print(match_res)
        detailed_log.append(match_res)

    with open(result_txt_path, 'a') as rf:
        rf.write("\n===== DETAILED MATCH RESPONSES =====\n")
        rf.write("\n".join(detailed_log) + "\n\n")
        rf.write("===== FINAL RESULTS =====\n")
        ranked = sorted(results.items(), key=lambda x: x[1]['points'], reverse=True)
        for paper, stats in ranked:
            rf.write(f"{os.path.basename(paper):<40} {stats['wins']:>3} {stats['draws']:>3} {stats['losses']:>3} {stats['points']:>5} ")

    # Generate and save chart
    try:
        plt.figure(figsize=(10, 6))
        names = [os.path.basename(p) for p, _ in ranked]
        points = [s['points'] for _, s in ranked]
        plt.barh(names, points)
        plt.gca().invert_yaxis()
        plt.xlabel("Points")
        plt.title("Paper Comparison Results")
        plt.tight_layout()
        plt.savefig(chart_path)
        print(f"[DEBUG] Chart saved to {chart_path}")
    except Exception as e:
        print(f"[ERROR] Failed to generate chart: {e}")

    print(" [INFO] Processing complete.")
