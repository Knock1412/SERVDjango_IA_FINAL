# language_detection_and_translation.py

import logging
from langdetect import detect, DetectorFactory
import argostranslate.package
import argostranslate.translate
import os

# Initialisation stable de langdetect
DetectorFactory.seed = 0

# Config basique du logger (si pas déjà configuré ailleurs)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fonction de détection de langue
def detect_language(text):
    try:
        lang = detect(text)
        logger.info(f"🌐 Langue détectée : {lang}")
        return lang
    except Exception as e:
        logger.error(f"Erreur détection langue: {e}")
        return 'unknown'

# Vérification des modèles Argos Translate
def install_argos_model(from_code='en', to_code='fr'):
    installed_languages = argostranslate.translate.get_installed_languages()
    codes = [lang.code for lang in installed_languages]

    if from_code not in codes or to_code not in codes:
        logger.warning("Les langues nécessaires ne sont pas installées dans Argos Translate.")
        logger.warning("Télécharger manuellement le modèle en-fr depuis : https://www.argosopentech.com/argospm/index/")
        raise Exception("Modèle Argos Translate manquant")
    else:
        logger.info("Les modèles Argos Translate sont installés et disponibles.")

# Traduction proprement dite
def translate_to_french(text):
    installed_languages = argostranslate.translate.get_installed_languages()
    from_lang = next(lang for lang in installed_languages if lang.code == "en")
    to_lang = next(lang for lang in installed_languages if lang.code == "fr")
    translation = from_lang.get_translation(to_lang)
    translated_text = translation.translate(text)
    logger.info("✅ Traduction effectuée avec succès.")
    return translated_text

# Fonction principale d'intégration pipeline
def process_text_block(text):
    lang = detect_language(text)
    if lang == 'en':
        logger.info("📝 Texte en anglais détecté → lancement de la traduction vers le français...")
        try:
            translated_text = translate_to_french(text)
            return translated_text, True  # True = traduction effectuée
        except Exception as e:
            logger.error(f"Erreur de traduction : {e}")
            return text, False
    else:
        logger.info("📝 Texte non anglais détecté → aucune traduction nécessaire.")
        return text, False
