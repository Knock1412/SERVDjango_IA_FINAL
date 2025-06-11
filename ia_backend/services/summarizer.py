from .ollama_gateway import generate_ollama
from bert_score import score as bert_score
from keybert import KeyBERT
from typing import List, Optional
import numpy as np
import logging
import time

# Initialisation du logger
logger = logging.getLogger(__name__)

# Modèle KeyBERT initialisé une seule fois pour optimiser les performances
kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

# --------- Constantes optimisées ---------
BERT_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3
FULL_WEIGHT = 0.6
PARTIAL_WEIGHT = 0.4
MIN_TEXT_LENGTH = 50

# --------- Prompt Templates ---------
BLOCK_PROMPT = """[INST] Tu es un expert en synthèse de documents techniques. Rédige un résumé concis en français qui :
1. Extrait strictement les concepts clés
2. Ignore les exemples redondants
3. Structure en : Problématique || Méthode || Résultats
4. Conserve les données chiffrées importantes

Texte à résumer :
{text}
[/INST]"""

INTERMEDIATE_PROMPT = """[INST] Tu es un expert en fusion de résumés techniques. Combine ces extraits en :
1. Éliminant les redondances
2. Structurant par thématique
3. Conservant les données techniques
4. Gardant une longueur moyenne (10-15 phrases)

Résumés à fusionner :
{text}
[/INST]"""

FINAL_PROMPT = """[INST] Tu es un assistant IA expert en synthèse de documents professionnels pour entreprise.
À partir des résumés intermédiaires ci-dessous, rédige un résumé final structuré et professionnel selon le plan suivant :
**Introduction :** Présente le contexte général et l’objectif global du document.
**Points clés :** Regroupe et développe de manière claire et synthétique les idées principales en 2 à 5 paragraphes distincts.
**Conclusion :** Résume en une synthèse concise les principaux apports du document.
Contraintes rédactionnelles strictes :
- Utiliser un vocabulaire professionnel et fluide.
- Structurer chaque partie avec des paragraphes distincts.
Voici les résumés intermédiaires à synthétiser :
{text}
[/INST]"""

# --------- Résumés Bloc ---------
def summarize_block(text: str) -> str:
    try:
        if not text or len(text.split()) < MIN_TEXT_LENGTH:
            logger.warning(f"Texte trop court ou vide - {len(text.split())} mots")
            return ""

        logger.debug(f"Résumé demandé pour texte de {len(text)} caractères")

        
        start_time = time.time()

        result = generate_ollama(
            prompt=BLOCK_PROMPT.format(text=text),
            num_predict=650,
            models=["phi3"],
            temperature=0.3
        )

        if not result or not isinstance(result, str):
            logger.error("Réponse d'Ollama invalide", extra={
                'type_reçu': type(result),
                'temps_exec': time.time() - start_time
            })
            return ""

        logger.info(f"Résumé généré en {time.time()-start_time:.2f}s - {len(result)} caractères")
        return result

    except Exception as e:
        logger.error("Échec summarize_block", exc_info=True)
        return ""

# --------- Fusion globale ---------
def summarize_global(summary_list: List[str], is_final: bool = False) -> str:
    try:
        if not isinstance(summary_list, (list, tuple)):
            logger.error(f"Type invalide pour summary_list: {type(summary_list)}. Attendu: list")
            return ""

        safe_summary_list = [str(item) if item else "" for item in summary_list]
        joined = "\n---\n".join(safe_summary_list)

        if not joined.strip():
            logger.warning("Aucun texte valide à fusionner après conversion")
            return ""

        prompt_template = FINAL_PROMPT if is_final else INTERMEDIATE_PROMPT
        prompt = prompt_template.format(text=joined)

        model_to_use = "mistral" if is_final else "mistral:instruct"
        num_predict = 1200 if is_final else 1000

        logger.info(f"Fusion de {len(safe_summary_list)} résumés - Longueur totale: {len(prompt)} caractères")

        result = generate_ollama(
            prompt=prompt,
            num_predict=num_predict,
            models=[model_to_use],
            top_k=30
        )

        return result if result else ""

    except Exception as e:
        logger.error(f"Erreur critique dans summarize_global: {str(e)}", exc_info=True)
        return ""

# --------- Scoring optimisé ---------
def compute_bertscore(ref: str, hyp: str) -> float:
    P, R, F1 = bert_score([hyp], [ref], lang="fr", model_type="distilbert-base-multilingual-cased")
    return float(F1.mean())

def compute_keyword_overlap(ref: str, hyp: str, top_k: int = 12) -> float:
    ref_kw = {kw[0] for kw in kw_model.extract_keywords(ref, top_n=top_k) if kw[1] > 0.2}
    hyp_kw = {kw[0] for kw in kw_model.extract_keywords(hyp, top_n=top_k) if kw[1] > 0.2}
    return len(ref_kw & hyp_kw) / max(len(ref_kw), 1)

def evaluate_summary_score(reference_text: str, summary_text: str, partial_summaries: Optional[List[str]] = None) -> float:
    if not reference_text or not summary_text or len(summary_text.split()) < 10:
        return 0.0

    bert_full = compute_bertscore(reference_text, summary_text)
    kw_full = compute_keyword_overlap(reference_text, summary_text)
    score_full = (BERT_WEIGHT * bert_full) + (KEYWORD_WEIGHT * kw_full)

    if partial_summaries:
        partial_concat = "\n".join(ps for ps in partial_summaries if ps)
        bert_partial = compute_bertscore(partial_concat, summary_text)
        kw_partial = compute_keyword_overlap(partial_concat, summary_text)
        score_partial = (BERT_WEIGHT * bert_partial) + (KEYWORD_WEIGHT * kw_partial)
    else:
        score_partial = score_full

    return round((FULL_WEIGHT * score_full) + (PARTIAL_WEIGHT * score_partial), 4)

# --------- Post-traitement amélioration ---------
def improve_summary(original_text: str, current_summary: str) -> str:
    prompt = f"""[INST] Améliore ce résumé en :
1. Supprimant les répétitions
2. Clarifiant les concepts techniques
3. Gardant tous les points clés originaux

Texte source :
{original_text}

Résumé actuel :
{current_summary}

Nouveau résumé amélioré :[/INST]"""

    return generate_ollama(
        prompt=prompt,
        num_predict=750,
        models=["phi3"],
        repeat_penalty=1.2
    )

# --------- Vérification de couverture de mots-clés ---------
def validate_summary_coverage(summary: str, keywords: List[str]) -> float:
    summary_kw = {kw[0] for kw in kw_model.extract_keywords(summary, top_n=20)}
    return len(set(keywords) & summary_kw) / max(len(keywords), 1)
