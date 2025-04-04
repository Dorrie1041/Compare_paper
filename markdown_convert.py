from pdfminer.high_level import extract_text
from markdownify import markdownify as md
import os
import sys

def convert_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print("File not found.")
        return
    print(f"Extracting from: {pdf_path}")
    text = extract_text(pdf_path)
    
    print("Converting to Markdown...")
    markdown = md(text)

    output_path = os.path.splitext(pdf_path)[0] + ".md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Saved Markdown to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 convert_pdf_to_md.py <path_to_pdf>")
    else:
        pdf_path = sys.argv[1]
        convert_pdf(pdf_path)    
