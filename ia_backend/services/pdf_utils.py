import os
import re
import time
import fitz  # PyMuPDF
from functools import lru_cache
from typing import List, Dict, Union, Optional, Tuple
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from dataclasses import dataclass

# ------------------ Configuration -------------------
@dataclass
class PDFConfig:
    PAGES_PER_CHUNK: int = 1  # Nombre de pages par chunk (1 = page par page)
    ANNEX_LAST_N: int = 10    # Pages à scanner pour annexes
    CACHE_SIZE: int = 100     # Taille du cache LRU
    MIN_TEXT_LENGTH: int = 10 # Nombre min de caractères pour considérer une page valide

# ------------------ Core Classes -------------------
class PDFMetrics:
    """Suivi des métriques de traitement"""
    def __init__(self):
        self.metrics = {
            'total_pages': 0,
            'processed_pages': 0,
            'annex_pages': 0,
            'empty_pages': 0
        }

    def log_page(self, is_annex: bool, is_empty: bool):
        self.metrics['total_pages'] += 1
        if is_annex: self.metrics['annex_pages'] += 1
        if is_empty: self.metrics['empty_pages'] += 1
        if not is_annex and not is_empty: 
            self.metrics['processed_pages'] += 1

    def get_report(self) -> Dict[str, int]:
        return self.metrics

class AnnexDetector:
    """Détection avancée des annexes"""
    def __init__(self):
        self.keywords = {
            'fr': ["annexe", "appendice", "bibliographie", "références"],
            'en': ["appendix", "references", "bibliography"]
        }

    def detect(self, pdf_path: str) -> List[int]:
        """Détecte les pages d'annexes"""
        doc = fitz.open(pdf_path)
        annex_pages = []
        
        for page_num in range(max(0, len(doc)-PDFConfig.ANNEX_LAST_N), len(doc)):
            text = get_page_text(pdf_path, page_num).lower()
            if any(kw in text for lang_kws in self.keywords.values() for kw in lang_kws):
                annex_pages.append(page_num)
                
        return annex_pages

# ------------------ Extraction de texte -------------------
@lru_cache(maxsize=PDFConfig.CACHE_SIZE)
def get_page_text(pdf_path: str, page_num: int) -> str:
    """Extraction avec cache d'une page"""
    doc = fitz.open(pdf_path)
    return doc.load_page(page_num).get_text("text").strip()

def extract_text_hybrid(pdf_path: str, page_numbers: List[int]) -> str:
    """Extraction hybride pour un ensemble de pages"""
    return "\n".join(
        get_page_text(pdf_path, p) 
        for p in page_numbers
    )

# ------------------ Découpage -------------------
def chunk_pdf(pdf_path: str) -> List[List[int]]:
    """
    Découpage garanti page par page
    Retourne une liste de chunks avec les numéros de page
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    return [
        list(range(i, min(i + PDFConfig.PAGES_PER_CHUNK, total_pages)))
        for i in range(0, total_pages, PDFConfig.PAGES_PER_CHUNK)
    ]

# ------------------ Pipeline complet -------------------
def process_pdf(pdf_path: str) -> Dict[str, Union[List[Tuple[int, str]], List[int]]]:
    """
    Traitement complet d'un PDF :
    1. Ouvre le PDF et compte les pages
    2. Détecte les annexes
    3. Découpe en chunks
    4. Extrait le texte
    """
    metrics = PDFMetrics()
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # Détection des annexes
    annex_detector = AnnexDetector()
    annex_pages = annex_detector.detect(pdf_path)
    
    # Découpage et extraction
    chunks = chunk_pdf(pdf_path)
    results = []
    
    for chunk in chunks:
        page_num = chunk[0]  # Un seul numéro de page par chunk
        is_annex = page_num in annex_pages
        text = get_page_text(pdf_path, page_num)
        is_empty = len(text) < PDFConfig.MIN_TEXT_LENGTH
        
        metrics.log_page(is_annex, is_empty)
        
        if not is_annex and not is_empty:
            results.append((page_num, text))
    
    return {
        'pages': results,
        'annex_pages': annex_pages,
        'metrics': metrics.get_report()
    }

# ------------------ Utilitaires -------------------
def print_processing_report(pdf_path: str):
    """Affiche un rapport de traitement"""
    result = process_pdf(pdf_path)
    metrics = result['metrics']
    
    print(f"\n📊 Rapport de traitement pour {os.path.basename(pdf_path)}")
    print(f"Pages totales: {metrics['total_pages']}")
    print(f"Pages traitées: {metrics['processed_pages']}")
    print(f"Pages annexes: {metrics['annex_pages']}")
    print(f"Pages vides: {metrics['empty_pages']}")
    print(f"Texte extrait: {sum(len(t[1]) for t in result['pages'])} caractères")