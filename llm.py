"""Génération des réponses du module I.

Deux modèles au choix :
- "flan-t5-small" : petit modèle Hugging Face, rapide, pour tester la chaîne ;
- "llama3:8b" : Llama 3 8B via Ollama (doit tourner sur le PC), pour les résultats.
"""

from functools import lru_cache

import ollama
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

FLAN_MODEL = "flan-t5-small"
LLAMA_MODEL = "llama3:8b"
MODELS = [LLAMA_MODEL, FLAN_MODEL]

MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 150
# Graine fixe : à température > 0, deux lancements donnent la même réponse.
SEED = 42
# Garde Llama 3 en mémoire 30 min (5 min par défaut) pour éviter
# ~1 min de rechargement entre deux questions.
OLLAMA_KEEP_ALIVE = "30m"
# Délai maximal d'une réponse (secondes) : sans lui, une connexion coupée
# (mise en veille du PC) bloque le programme indéfiniment.
OLLAMA_TIMEOUT = 300
_ollama = ollama.Client(timeout=OLLAMA_TIMEOUT)


def build_prompt(question, context):
    # Public visé : migrants et réfugiés dans l'Union européenne.
    # Consigne en français : en anglais, Llama 3 répondait en anglais
    # à 18 questions françaises sur 20, malgré la demande.
    return f"""Tu es un assistant qui aide les migrants et les réfugiés à comprendre
leurs droits et leurs démarches dans l'Union européenne.
Réponds uniquement à partir du contexte, en français, avec des mots simples
et respectueux, et explique les termes juridiques. Si le contexte ne suffit
pas, dis que l'information n'est pas disponible dans les documents.
Ne fais aucune supposition sur la nationalité, la religion ou l'origine
de la personne.

Contexte : {context}

Question : {question}

Réponse (en français) :"""


@lru_cache(maxsize=1)
def load_flan():
    name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.eval()
    return tokenizer, model


def _generate_flan(prompt, temperature):
    tokenizer, model = load_flan()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    # Température 0 = décodage glouton (déterministe).
    if temperature > 0:
        torch.manual_seed(SEED)
        sampling = {"do_sample": True, "temperature": temperature}
    else:
        sampling = {"do_sample": False}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, **sampling
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _generate_llama(prompt, temperature):
    result = _ollama.generate(
        model=LLAMA_MODEL,
        prompt=prompt,
        keep_alive=OLLAMA_KEEP_ALIVE,
        options={
            "temperature": temperature,
            "seed": SEED,
            "num_predict": MAX_NEW_TOKENS,
        },
    )
    return result["response"].strip()


def generate_response(question, context="", model=FLAN_MODEL, temperature=0.0):
    """Génère une réponse. Lève ConnectionError si Ollama ne tourne pas."""
    prompt = build_prompt(question, context)
    if model == LLAMA_MODEL:
        return _generate_llama(prompt, temperature)
    if model == FLAN_MODEL:
        return _generate_flan(prompt, temperature)
    raise ValueError(f"Unknown model: {model}")
