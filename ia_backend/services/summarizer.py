from .ollama_gateway import generate_ollama
from bert_score import score as bert_score
from keybert import KeyBERT

# Initialisation modèle keywords multilingue cohérent
kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

# Pondérations centralisées
BERT_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3
FULL_WEIGHT = 0.6
PARTIAL_WEIGHT = 0.4

# ------------- Résumés Bloc et Global --------------

def summarize_block(text):
    prompt = (
        "Tu es un assistant IA spécialisé dans la synthèse de documents longs pour entreprise.\n"
        "Résume fidèlement le passage suivant en français, en extrayant les idées principales, sans inventer de contenu supplémentaire.\n\n"
        f"{text}\n"
    )
    return generate_ollama(prompt, num_predict=300, models=["mistral:instruct"])

def summarize_global(summary_list, num_predict=650):
    joined = "\n".join(summary_list)
    prompt = (
        "Tu es un assistant IA expert en synthèse de documents professionnels en français pour entreprise.\n\n"
        "Rédige un résumé structuré à partir des résumés partiels suivants :\n\n"
        "- Introduction : contexte du document\n"
        "- Points clés : idées majeures regroupées\n"
        "- Conclusion : résumé final.\n\n"
        f"{joined}"
    )
    return generate_ollama(prompt, num_predict=num_predict, models=["mistral"])

def determine_predict_length(block_count):
    if block_count < 15:
        return 800
    else:
        return 1000

# ------------- Scoring double pondéré --------------

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

# ------------- Amélioration automatique -------------

def improve_summary(original_text, current_summary):
    prompt = (
        "Tu es un expert IA chargé d'améliorer les résumés.\n\n"
        f"Texte source :\n{original_text}\n\n"
        f"Résumé initial :\n{current_summary}\n\n"
        "Améliore la clarté et la précision du résumé, sans inventer d'informations, en restant fidèle au contenu source."
    )
    return generate_ollama(prompt, num_predict=300, models=["mistral:instruct"])
