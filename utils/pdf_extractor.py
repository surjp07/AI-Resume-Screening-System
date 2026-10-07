def extract_text_from_pdf(uploaded_file):
    from pypdf import PdfReader
    reader=PdfReader(uploaded_file)
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()
