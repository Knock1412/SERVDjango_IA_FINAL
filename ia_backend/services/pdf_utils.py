import os
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from functools import lru_cache
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union, Optional
import logging

# ---------- Logging centralisé ----------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ---------- Configuration centralisée ----------
@dataclass
class PDFConfig:
    PAGES_PER_CHUNK: int = 1  # par défaut 1 page
    CACHE_SIZE: int = 100
    MIN_TEXT_LENGTH: int = 10
    DYNAMIC_THRESHOLD: int = 80  # seuil pour passer en découpage dynamique

# ---------- Découpage dynamique ----------
def determine_dynamic_chunk_size(total_pages: int) -> int:
    return 1 if total_pages <= PDFConfig.DYNAMIC_THRESHOLD else 5

def extract_blocks_from_pdf(pdf_path: str, chunk_size: Optional[int] = None, return_pages_only: bool = False) -> List[List[int]]:
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = doc.page_count
            if return_pages_only:
                return total_pages

            if chunk_size is None:
                chunk_size = determine_dynamic_chunk_size(total_pages)

            blocks = [list(range(i, min(i + chunk_size, total_pages))) for i in range(0, total_pages, chunk_size)]
            return blocks

    except Exception as e:
        logger.error(f"Erreur lors du découpage PDF : {e}")
        return []

# ---------- Extraction texte hybride ----------
@lru_cache(maxsize=PDFConfig.CACHE_SIZE)
def extract_text_pymupdf(pdf_path: str, page_num: int) -> str:
    try:
        with fitz.open(pdf_path) as doc:
            text = doc.load_page(page_num).get_text("text")
            return text.strip() if text else ""
    except Exception as e:
        logger.error(f"Erreur extraction texte page {page_num}: {e}")
        return ""

def extract_text_from_block(block_indices: List[int], pdf_path: str) -> str:
    laparams = LAParams()
    output = ""
    for i in sorted(block_indices):
        try:
            text = extract_text(pdf_path, page_numbers=[i], laparams=laparams)
            if text:
                output += text + "\n"
        except Exception as e:
            logger.warning(f"Erreur extraction PDFMiner page {i}: {e}")
    return output.strip()

def extract_full_text(pdf_path: str) -> str:
    laparams = LAParams()
    try:
        text = extract_text(pdf_path, laparams=laparams)
        return text.strip() if text else ""
    except Exception as e:
        logger.error(f"Erreur extraction full text: {e}")
        return ""
