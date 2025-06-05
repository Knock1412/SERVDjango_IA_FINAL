# language_detection_and_translation.py
import logging
from langdetect import detect, DetectorFactory, LangDetectException
import argostranslate.package
import argostranslate.translate
from typing import Tuple
import hashlib

# --------- Configuration ---------
DetectorFactory.seed = 0  # Reproductibilité
SUPPORTED_LANGS = {'fr', 'en'}  # Langues gérées
MIN_TEXT_LENGTH = 20  # Longueur minimale pour traitement
CACHE_SIZE = 1000  # Taille du cache de traduction

# --------- Logging ---------
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --------- Cache de traduction ---------
_translation_cache = {}

def _get_cache_key(text: str, target_lang: str) -> str:
    """Génère une clé de cache unique"""
    return hashlib.md5(f"{text}_{target_lang}".encode()).hexdigest()

# --------- Détection de langue améliorée ---------
def detect_language(text: str) -> str:
    """Détecte la langue avec gestion des cas limites"""
    if not text or len(text.strip().split()) < 3:
        return 'unknown'
    
    try:
        lang = detect(text)
        logger.debug(f"Langue détectée: {lang}")
        return lang if lang in SUPPORTED_LANGS else 'unknown'
    except LangDetectException as e:
        logger.warning(f"Erreur détection langue: {str(e)}")
        return 'unknown'

# --------- Gestion des modèles ---------
def _check_argos_models(from_code: str = 'en', to_code: str = 'fr') -> bool:
    """Vérifie la présence des modèles nécessaires"""
    installed_languages = argostranslate.translate.get_installed_languages()
    available_codes = {lang.code for lang in installed_languages}
    
    if not {from_code, to_code}.issubset(available_codes):
        logger.error(f"Modèles manquants. Disponibles: {available_codes}")
        raise ValueError(
            f"Modèles {from_code}->{to_code} non installés. "
            f"Télécharger via: argospm install {from_code} {to_code}"
        )
    return True

# --------- Traduction optimisée ---------
def translate_text(text: str, from_lang: str = 'en', to_lang: str = 'fr') -> str:
    """Traduction avec cache et vérification préalable"""
    if len(text) < MIN_TEXT_LENGTH:
        return text
    
    cache_key = _get_cache_key(text, to_lang)
    if cache_key in _translation_cache:
        logger.debug("Utilisation du cache de traduction")
        return _translation_cache[cache_key]
    
    _check_argos_models(from_lang, to_lang)
    
    installed_langs = argostranslate.translate.get_installed_languages()
    source_lang = next(lang for lang in installed_langs if lang.code == from_lang)
    target_lang = next(lang for lang in installed_langs if lang.code == to_lang)
    
    translation = source_lang.get_translation(target_lang)
    result = translation.translate(text)
    
    _translation_cache[cache_key] = result
    if len(_translation_cache) > CACHE_SIZE:
        _translation_cache.pop(next(iter(_translation_cache)))
    
    logger.info(f"Traduction {from_lang}->{to_lang} réussie")
    return result

# --------- Pipeline principal ---------
def process_text_block(text: str) -> Tuple[str, bool]:
    """
    Traite un bloc de texte:
    - Détecte la langue
    - Traduit si nécessaire (en->fr)
    Retourne: (texte_processed, was_translated)
    """
    if not text or not isinstance(text, str):
        return text, False
    
    try:
        lang = detect_language(text)
        
        if lang == 'en':
            logger.info("Traduction EN->FR en cours...")
            translated = translate_text(text)
            return translated, True
        
        return text, False
    
    except Exception as e:
        logger.error(f"Erreur traitement texte: {str(e)}", exc_info=True)
        return text, False

# --------- Initialisation ---------
def initialize_translation():
    """Vérifie les dépendances au démarrage"""
    try:
        _check_argos_models()
        logger.info("Modules de traduction prêts")
    except Exception as e:
        logger.critical(f"Échec initialisation: {str(e)}")
        raise

# Vérification au chargement
initialize_translation()