import PyPDF2, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open(r'C:\Users\user\Documents\NPJ2\2507.14077v1.pdf', 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    print(f'Total pages: {len(reader.pages)}')
    for i in range(min(7, len(reader.pages))):
        text = reader.pages[i].extract_text() or ''
        low = text.lower()
        if 'figure 1' in low or 'fig. 1' in low or 'fig 1' in low:
            print(f'\n=== PAGE {i+1} — Figure 1 found ===')
            print(text[:4000])
        else:
            print(f'\n--- Page {i+1} (first 300 chars) ---')
            print(text[:300])
