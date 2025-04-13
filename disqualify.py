import sys
import os
from pdfminer.high_level import extract_text
from markdownify import markdownify as md
import openai
from dotenv import load_dotenv, find_dotenv

# Load .env and OpenAI API key
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")  # safer than hardcoding
client = openai.OpenAI(api_key=openai.api_key)

# Convert PDF to Markdown
def convert_pdf_to_markdown(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)
    print(f"[+] Extracting text from: {pdf_path}")
    text = extract_text(pdf_path)
    print(f"[+] Converting to Markdown...")
    markdown = md(text)
    return markdown

# Ask OpenAI to determine if the paper should be disqualified
def is_disqualified(markdown_text):
    prompt = f"""You are reviewing academic papers. Disqualify any paper that meets any of the following criteria:

1. No evaluation – there is no experimental, empirical, or quantitative analysis.
   A paper is considered to have evaluation **only if** it includes at least one of the following:  
        • A section labeled “Experiments”, “Evaluation”, or similar  
        • A table showing accuracy, performance, benchmarks, timing, etc.  
        • Charts or graphs comparing results  
        • Qualitative or quantitative analysis of outcomes  
        • Phrases like:  
            – “We evaluate our method on…”  
            – “Table 1 shows…”  
            – “Our approach achieves better results than…”

2. No related work – the paper does not engage with prior research in a meaningful way. This includes cases where:
        •	It cites few or no research papers,
        •	It only cites tools, frameworks, or datasets (e.g., PyTorch, COCO),
        •	It does not explain or compare to the contributions of cited work,
        •	It lacks a section labeled “Related Work” or any such discussion embedded in the introduction.

3. Not in English – the paper is written in a language other than English.

4. No novelty – the work does not offer any new idea, method, result, or insight.
   A paper is considered to have novelty **only if it introduces something not previously done**. Signs of novelty include:  
   • A new method, algorithm, framework, dataset, or result  
   • A new application of existing techniques to a previously unstudied domain  
   • Claims like:  
     – “We propose a new...”  
     – “Our main contribution is...”  
     – “To the best of our knowledge, this is the first...”  
     – “Unlike previous approaches, our method...”  
   If the paper merely reuses existing techniques without proposing any improvement, innovation, or new perspective, it should be disqualified.

5. Survey/review paper – the paper summarizes existing work without proposing a new approach.
        • If the **title contains the word "survey" or "review"**, disqualify it immediately without analyzing the full content.  
        • Otherwise, disqualify only if the main body primarily summarizes existing work without introducing any new contribution.


Here is the paper content (in Markdown):

\"\"\"{markdown_text[:8000]}\"\"\"  # Trim for token limits

Answer with either:
- "Disqualified: <reason>"
- "Qualified"
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    result = response.choices[0].message.content.strip()
    return result

# Process all PDFs in a folder and filter qualified ones
def process_papers(pdf_folder):
    qualified_papers = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            markdown = convert_pdf_to_markdown(pdf_path)
            decision = is_disqualified(markdown)
            print(f"[{filename}] → {decision}")
            if decision.lower().startswith("qualified"):
                qualified_papers.append((filename, markdown))
    return qualified_papers

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python filter_papers.py /path/to/pdf_folder")
        sys.exit(1)

    folder_path = sys.argv[1]
    qualified = process_papers(folder_path)
    print(f"\n✅ {len(qualified)} papers qualified out of {len(os.listdir(folder_path))}")