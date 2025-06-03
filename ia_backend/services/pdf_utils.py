import os
from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams

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
    - chunk_size : permet de forcer manuellement la taille des blocs (sinon dynamique)
    - return_pages_only : si True, retourne juste le nombre total de pages
    """
    with open(pdf_path, "rb") as f:
        pages = list(PDFPage.get_pages(f))
        total_pages = len(pages)
        if return_pages_only:
            return total_pages

    # Calcul dynamique si chunk_size non précisé
    if chunk_size is None:
        chunk_size = determine_dynamic_chunk_size(total_pages)

    blocks = [list(range(i, min(i + chunk_size, total_pages))) for i in range(0, total_pages, chunk_size)]
    return blocks

def extract_text_from_block(block_indices, pdf_path):
    """
    Extrait tout le texte d’un bloc (liste d’index de pages) depuis un PDF local.
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
    Extrait tout le texte brut d’un PDF complet en mémoire.
    """
    laparams = LAParams()
    text = extract_text(pdf_path, laparams=laparams)
    return text.strip()