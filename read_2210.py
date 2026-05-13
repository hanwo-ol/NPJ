import fitz
import sys

sys.stdout.reconfigure(encoding='utf-8')

pdf_path = r"c:\Users\user\Documents\NPJ2\2210.12759v4.pdf"
try:
    doc = fitz.open(pdf_path)
    with open(r"c:\Users\user\Documents\NPJ2\2210_output.txt", "w", encoding="utf-8") as f:
        for page in doc:
            f.write(page.get_text())
    print("PDF extraction successful. Saved to 2210_output.txt")
except Exception as e:
    print(f"Error: {e}")
