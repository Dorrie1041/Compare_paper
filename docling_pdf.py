from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
import os

output_dir = "markdown_files"
os.makedirs(output_dir, exist_ok=True)

artifacts_path = "/Users/dongruishen/.cache/docling/models"


with open("pdf_link.txt", "r", encoding="utf-8") as f:
    pdfs = [line.strip() for line in f if line.strip()]

pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
for pdf in pdfs:
    try:
        result = doc_converter.convert(pdf)
        filename = os.path.basename(pdf)
        if filename.endswith(".pdf"):
            filename = filename.replace(".pdf", ".md")
        else:
            filename += ".md"    
        output_path = os.path.join(output_dir, filename)  

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.document.export_to_markdown())

        print(f"[✓] Converted: {pdf} → {output_path}")
    except Exception as e:
        print(f"[✗] Failed to convert {pdf}: {e}")