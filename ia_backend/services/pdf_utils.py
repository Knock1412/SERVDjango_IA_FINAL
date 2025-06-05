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
    formatter = logging.Formatter('%(message)s')  # ✅ Suppression timestamp
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ---------- Configuration centralisée ----------
@dataclass
class PDFConfig:
    PAGES_PER_CHUNK: int = 1  # par défaut 1 page
    ANNEX_LAST_N: int = 10    # nombre de pages de fin à analyser pour les annexes
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

# ---------- Détection avancée des annexes ----------
ANNEXE_KEYWORDS = [
    "annexe", "appendice", "bibliographie", "références", 
    "remerciements", "glossaire", "notes", "table des matières", 
    "index", "abstract", "abréviations", "sigles", "valorisation"
]

def detect_annex_start_page(pdf_path: str, last_pages: int = 7, y_threshold_high: int = 150, y_threshold_low: int = 400) -> Optional[int]:
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = doc.page_count

            for page_num in range(max(0, total_pages - last_pages), total_pages):
                page = doc[page_num]
                blocks = page.get_text("blocks", sort=True)

                for b in blocks:
                    x0, y0, x1, y1, text, *_ = b
                    text_lower = text.lower()
                    for kw in ANNEXE_KEYWORDS:
                        if kw in text_lower:
                            if y0 < y_threshold_high:
                                logger.info(f"🔎 Annexe détectée haut p{page_num+1} (y={y0})")
                                return page_num
                            elif y0 >= y_threshold_low:
                                logger.info(f"🔎 Annexe détectée bas p{page_num+1} (y={y0})")
                                return page_num + 1
                            else:
                                logger.info(f"🔎 Annexe zone intermédiaire p{page_num+1} (y={y0})")
                                return page_num
        return None
    except Exception as e:
        logger.error(f"Erreur détection annexes : {e}")
        return None

def is_likely_annex(block_indices: List[int], total_pages: int, annex_start_page: Optional[int]) -> bool:
    if annex_start_page is None:
        return False
    return any(page >= annex_start_page for page in block_indices)
