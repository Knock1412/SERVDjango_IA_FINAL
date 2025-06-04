from .ollama_gateway import generate_ollama
from bert_score import score as bert_score
from keybert import KeyBERT

kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

# Pondérations
BERT_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3
FULL_WEIGHT = 0.6
PARTIAL_WEIGHT = 0.4

# --------- Résumés Bloc ---------

def summarize_block(text):
    prompt = (
        "Tu es un assistant IA spécialisé en synthèse de documents professionnels en français.\n"
        "Synthétise précisément le texte suivant en extrayant ses idées essentielles et strictement les concepts clés.\n"
        "Ignore les introductions vagues, phrases génériques et détails accessoires.\n"
        "Ne jamais inventer ni extrapoler, toujours rester fidèle au contenu fourni.\n\n"
        f"{text}\n"
    )
    predict_len = 650 if len(text) > 6000 else 500
    return generate_ollama(prompt, num_predict=predict_len, models=["mistral:instruct"])

# --------- Fusion globale et intermédiaire ---------

def summarize_global(summary_list, num_predict=650, is_final=False):
    joined = "\n".join(summary_list)

    if is_final:
        prompt = (
            "Tu es un assistant IA expert en synthèse de documents professionnels pour entreprise.\n"
            "À partir des résumés intermédiaires ci-dessous, rédige un résumé final structuré et professionnel selon le plan suivant :\n"
            "**Introduction :** Présente le contexte général et l’objectif global du document.\n"
            "**Points clés :** Regroupe et développe de manière claire et synthétique les idées principales en 2 à 5 paragraphes distincts.\n"
            "**Conclusion :** Résume en une synthèse concise les principaux apports du document.\n"
            "Contraintes rédactionnelles strictes :\n"
            "- Utiliser un vocabulaire professionnel et fluide.\n"
            "- Structurer chaque partie avec des paragraphes distincts.\n"
            "Voici les résumés intermédiaires à synthétiser :\n"
            f"{joined}"
        )
    else:
        prompt = (
            "Tu es un assistant IA de synthèse.\n"
            "Fusionne les résumés suivants en gardant l'essentiel de façon claire et compacte, sans reformuler excessivement :\n"
            f"{joined}"
        )

    return generate_ollama(prompt, num_predict=num_predict, models=["mistral:instruct"])

# --------- Scoring ---------

def compute_bertscore(ref, hyp):
    P, R, F1 = bert_score([hyp], [ref], lang="fr", model_type="distilbert-base-multilingual-cased")
    return F1[0].item()

def compute_keyword_overlap(ref, hyp, top_k=12):
    ref_kw = set([kw[0] for kw in kw_model.extract_keywords(ref, top_n=top_k)])
    hyp_kw = set([kw[0] for kw in kw_model.extract_keywords(hyp, top_n=top_k)])
    overlap = len(ref_kw.intersection(hyp_kw)) / max(len(ref_kw), 1)
    return overlap

def evaluate_summary_score(reference_text, summary_text, partial_summaries=None):
    if not reference_text or not summary_text:
        return 0.0

    bert_full = compute_bertscore(reference_text, summary_text)
    kw_full = compute_keyword_overlap(reference_text, summary_text)
    score_full = (BERT_WEIGHT * bert_full) + (KEYWORD_WEIGHT * kw_full)

    if partial_summaries:
        partial_concat = "\n".join(partial_summaries)
        bert_partial = compute_bertscore(partial_concat, summary_text)
        kw_partial = compute_keyword_overlap(partial_concat, summary_text)
        score_partial = (BERT_WEIGHT * bert_partial) + (KEYWORD_WEIGHT * kw_partial)
    else:
        score_partial = score_full

    final_score = (FULL_WEIGHT * score_full) + (PARTIAL_WEIGHT * score_partial)
    return round(final_score, 4)

def improve_summary(original_text, current_summary):
    prompt = (
        "Tu es un expert IA chargé d'améliorer les résumés.\n\n"
        f"Texte source :\n{original_text}\n\n"
        f"Résumé initial :\n{current_summary}\n\n"
        "Améliore la clarté et la précision du résumé, sans inventer d'informations, fidèle au texte."
    )
    return generate_ollama(prompt, num_predict=650, models=["mistral:instruct"])