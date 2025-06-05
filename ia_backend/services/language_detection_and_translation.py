# language_detection_and_translation.py

import logging
from langdetect import detect, DetectorFactory, LangDetectException
import argostranslate.package
import argostranslate.translate
import hashlib
from typing import Tuple

# --------- Initialisation stable de langdetect ---------
DetectorFactory.seed = 0

# --------- Configurations globales ---------
SUPPORTED_LANGS = {'fr', 'en'}
MIN_TEXT_LENGTH = 20
CACHE_SIZE = 1000

# --------- Logging centralisé ---------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')  # ✅ Suppression timestamp
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --------- Cache de traduction ---------
_translation_cache = {}

def _get_cache_key(text: str, target_lang: str) -> str:
    return hashlib.md5(f"{text}_{target_lang}".encode()).hexdigest()

# --------- Détection de langue robuste ---------
def detect_language(text: str) -> str:
    if not text or len(text.strip().split()) < 3:
        logger.warning("Texte trop court pour détecter la langue.")
        return 'unknown'
    
    try:
        lang = detect(text)
        logger.info(f"🌐 Langue détectée : {lang}")
        return lang if lang in SUPPORTED_LANGS else 'unknown'
    except LangDetectException as e:
        logger.error(f"Erreur détection langue: {e}")
        return 'unknown'

# --------- Vérification des modèles Argos Translate ---------
def _check_argos_models(from_code: str = 'en', to_code: str = 'fr') -> bool:
    installed_languages = argostranslate.translate.get_installed_languages()
    available_codes = {lang.code for lang in installed_languages}

    if not {from_code, to_code}.issubset(available_codes):
        logger.error(f"Modèles Argos manquants. Disponibles: {available_codes}")
        raise ValueError(
            f"Modèles {from_code}->{to_code} non installés.\n"
            f"Télécharger via : argospm install {from_code} {to_code}"
        )
    return True

# --------- Traduction avec cache ---------
def translate_text(text: str, from_lang: str = 'en', to_lang: str = 'fr') -> str:
    if len(text) < MIN_TEXT_LENGTH:
        logger.debug("Texte trop court, traduction ignorée.")
        return text

    cache_key = _get_cache_key(text, to_lang)
    if cache_key in _translation_cache:
        logger.debug("✅ Traduction récupérée depuis le cache.")
        return _translation_cache[cache_key]

    _check_argos_models(from_lang, to_lang)

    installed_langs = argostranslate.translate.get_installed_languages()
    source_lang = next(lang for lang in installed_langs if lang.code == from_lang)
    target_lang = next(lang for lang in installed_langs if lang.code == to_lang)

    translation = source_lang.get_translation(target_lang)
    result = translation.translate(text)

    _translation_cache[cache_key] = result
    if len(_translation_cache) > CACHE_SIZE:
        _translation_cache.pop(next(iter(_translation_cache)))  # nettoyage du cache

    logger.info(f"✅ Traduction {from_lang}->{to_lang} réussie.")
    return result

# --------- Pipeline principal optimisé ---------
def process_text_block(text: str) -> Tuple[str, bool]:
    """
    Simplification : retourne toujours (texte final, bool traduction_effectuée)
    """
    if not text or not isinstance(text, str):
        logger.warning("Texte vide ou type invalide reçu.")
        return text, False

    try:
        lang = detect_language(text)
        if lang == 'en':
            logger.info("📝 Traduction EN->FR en cours...")
            translated = translate_text(text)
            return translated, True
        else:
            logger.info("📝 Texte non anglais → aucune traduction.")
            return text, False
    except Exception as e:
        logger.error(f"Erreur globale de traitement : {e}", exc_info=True)
        return text, False

# --------- Initialisation des modèles au démarrage ---------
def initialize_translation():
    try:
        _check_argos_models()
        logger.info("✅ Modèles de traduction vérifiés et prêts.")
    except Exception as e:
        logger.critical(f"❌ Échec initialisation Argos : {e}")
        raise

# Vérifie au chargement si les modèles sont bien présents
initialize_translation()
