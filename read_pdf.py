import sys

def try_read_pdf(filepath):
    text = ""
    try:
        import PyPDF2
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        with open('pdf_output_utf8.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        pass

    try:
        import pypdf
        with open(filepath, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        with open('pdf_output_utf8.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        pass

    try:
        import fitz
        doc = fitz.open(filepath)
        for page in doc:
            text += page.get_text()
        with open('pdf_output_utf8.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        pass
        
    print("Could not read PDF. No suitable library found.")
    return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try_read_pdf(sys.argv[1])
