import sys, fitz
sys.stdout.reconfigure(encoding='utf-8')
doc = fitz.open(r'c:\Users\user\Documents\NPJ2\2204.12044v1.pdf')
with open(r'c:\Users\user\Documents\NPJ2\istrboost_output.txt', 'w', encoding='utf-8') as f:
    f.write(f'Total pages: {len(doc)}\n')
    for i in range(len(doc)):
        txt = doc[i].get_text()
        f.write(f'\n===PAGE {i}===\n')
        f.write(txt)
print('Done')
