import os
from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams

def extract_blocks_from_pdf(pdf_path, chunk_size=3, return_pages_only=False):
    """
    Découpe un PDF en blocs de pages (par chemin uniquement).
    - chunk_size : nombre de pages par bloc
    - return_pages_only : si True, retourne juste le nombre total de pages
    """
    with open(pdf_path, "rb") as f:
        pages = list(PDFPage.get_pages(f))
        if return_pages_only:
            return len(pages)

    blocks = [list(range(i, min(i + chunk_size, len(pages)))) for i in range(0, len(pages), chunk_size)]
    return blocks

def extract_text_from_block(block_indices, pdf_path):
    """
    Extrait tout le texte d’un bloc (liste d’index de pages) depuis un PDF local.
    """
    laparams = LAParams()
    output = ""

    # On force l'ordre croissant pour sécuriser l'ordre des pages
    for i in sorted(block_indices):
        text = extract_text(pdf_path, page_numbers=[i], laparams=laparams)
        if text:
            output += text + "\n"

    return output.strip()

def extract_full_text(pdf_path):
    """
    Extrait tout le texte brut d’un PDF complet en mémoire.
    """
    laparams = LAParams()
    text = extract_text(pdf_path, laparams=laparams)
    return text.strip()
