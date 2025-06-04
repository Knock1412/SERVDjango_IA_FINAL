import os
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams

# ------------ Découpage dynamique optimisé ------------

def determine_dynamic_chunk_size(total_pages):
    """
    Découpage hybride optimisé industriel :
    - ≤ 80 pages : page-par-page
    - > 80 pages : blocs de 5 pages
    """
    if total_pages <= 80:
        return 1
    else:
        return 5

def extract_blocks_from_pdf(pdf_path, chunk_size=None, return_pages_only=False):
    """
    Découpe un PDF en blocs de pages avec découpage adaptatif.
    """
    with open(pdf_path, "rb") as f:
        pages = list(PDFPage.get_pages(f))
        total_pages = len(pages)
        if return_pages_only:
            return total_pages

    if chunk_size is None:
        chunk_size = determine_dynamic_chunk_size(total_pages)

    blocks = [list(range(i, min(i + chunk_size, total_pages))) for i in range(0, total_pages, chunk_size)]
    return blocks

# ------------ Extraction texte PDF brute --------------

def extract_text_from_block(block_indices, pdf_path):
    """
    Extrait tout le texte d’un bloc (liste d’index de pages).
    """
    laparams = LAParams()
    output = ""

    for i in sorted(block_indices):
        text = extract_text(pdf_path, page_numbers=[i], laparams=laparams)
        if text:
            output += text + "\n"

    return output.strip()

def extract_full_text(pdf_path):
    """
    Extrait tout le texte brut du PDF complet.
    """
    laparams = LAParams()
    text = extract_text(pdf_path, laparams=laparams)
    return text.strip()

# ------------ Détection avancée d'annexes V4.2 ------------

ANNEXE_KEYWORDS = [
    "annexe", "appendice", "bibliographie", "références", 
    "remerciements", "glossaire", "notes", "table des matières", 
    "index", "abstract", "abréviations", "sigles, Valorisation"
]

def detect_annex_start_page(pdf_path, last_pages=7, y_threshold_high=150, y_threshold_low=400):
    """
    Analyse structurée des dernières pages pour identifier le début réel d'annexe.
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    for page_num in range(max(0, total_pages - last_pages), total_pages):
        page = doc[page_num]
        blocks = page.get_text("blocks", sort=True)  # On garde l’ordre vertical des blocs

        # On parcourt chaque bloc de haut en bas
        for b in blocks:
            x0, y0, x1, y1, text, *_ = b
            text_lower = text.lower()
            for kw in ANNEXE_KEYWORDS:
                if kw in text_lower:
                    if y0 < y_threshold_high:
                        print(f"🔎 Annexe détectée haut à page {page_num+1} — y={y0}")
                        return page_num  # coupe immédiatement à partir de cette page
                    elif y0 >= y_threshold_low:
                        print(f"🔎 Annexe détectée bas à page {page_num+1} — y={y0}")
                        return page_num + 1  # coupe à partir de la page suivante
                    else:
                        print(f"🔎 Annexe détectée zone intermédiaire page {page_num+1} — y={y0}")
                        return page_num  # on reste conservateur : on coupe quand même sur cette page

    return None

def is_likely_annex(text, block_indices, total_pages, annex_start_page=None):
    """
    Vérifie si un bloc est à ignorer selon la position d'annexe détectée.
    """
    if annex_start_page is None:
        return False

    for page in block_indices:
        if page >= annex_start_page:
            return True
    return False
