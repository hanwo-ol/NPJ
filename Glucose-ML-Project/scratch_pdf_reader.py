import sys
import io

# Force utf-8 encoding for standard output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

pdf_path = "C:/Users/user/Documents/NPJ2/2507.14077v1.pdf"

try:
    import PyPDF2
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page_num in range(len(reader.pages)):
            text = reader.pages[page_num].extract_text()
            if not text: continue
            if "sampl" in text.lower() or "5 " in text.lower() or "5-min" in text.lower() or "align" in text.lower():
                print(f"--- Page {page_num+1} (PyPDF2) ---")
                lines = text.split('\n')
                for i, line in enumerate(lines):
                    if "sampl" in line.lower() or "resampl" in line.lower() or "align" in line.lower():
                        start = max(0, i-3)
                        end = min(len(lines), i+4)
                        print('\n'.join(lines[start:end]))
                        print("="*40)
except Exception as e2:
    print(f"PyPDF2 failed: {e2}")
