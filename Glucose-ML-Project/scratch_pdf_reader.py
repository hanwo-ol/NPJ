import sys
import io

# sys.stdout.reconfigure는 python 3.7+ 에서 가장 안정적인 인코딩 재설정 방식입니다.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

pdf_path = "c:/Users/user/Documents/NPJ2/Glucose-ML-Project/020_Tier_9_Temporal_Models/paper_raw/2605.21088.pdf"

try:
    import fitz # PyMuPDF
    doc = fitz.open(pdf_path)
    print(f"Total pages: {len(doc)}")
    # 처음 4페이지의 텍스트를 추출하여 출력합니다.
    for i in range(min(4, len(doc))):
        print(f"\n================ PAGE {i+1} ================")
        print(doc[i].get_text())
except Exception as e:
    print(f"Extraction failed: {e}")
