from PyPDF2 import PdfReader
from .ollama_gateway import generate_ollama
import io

def answer_question_from_pdf(pdf_bytes, question, model="mistral"):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    context = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

    prompt = f"""Tu es un assistant intelligent. Voici un document :\n
{context}\n
En te basant uniquement sur ce document, réponds en français précisément à cette question :\n
{question}
"""
    response, model_used = generate_ollama(prompt, num_predict=300, models=[model])
    return response
