"""Génération des réponses de l’assistant.

Modèles proposés :
- "flan-t5-small" : petit modèle Hugging Face, rapide, pour tester la chaîne ;
- "llama3:8b" : Llama 3 8B via Ollama ;
- "llama3:8b-instruct-q2_K" : variante plus compressée, à évaluer séparément.
"""

import os
from functools import lru_cache

import ollama

FLAN_MODEL = "flan-t5-small"
LLAMA_MODEL = "llama3:8b"
LLAMA_LIGHT_MODEL = "llama3:8b-instruct-q2_K"
MODELS = [LLAMA_MODEL, LLAMA_LIGHT_MODEL, FLAN_MODEL]
DEFAULT_MODEL = os.environ.get("MIGRANT_LLM_MODEL", LLAMA_MODEL)

MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 150
# Graine fixe : à température > 0, deux lancements donnent la même réponse.
SEED = 42
PROMPT_VERSION = "scoped-context-v2"
# Durée de maintien en mémoire configurable selon la RAM disponible.
OLLAMA_KEEP_ALIVE = os.environ.get("MIGRANT_OLLAMA_KEEP_ALIVE", "5m")
# Délai maximal d'une réponse (secondes) : sans lui, une connexion coupée
# (mise en veille du PC) bloque le programme indéfiniment.
OLLAMA_TIMEOUT = 300
_ollama = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"), timeout=OLLAMA_TIMEOUT)


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
Vérifie le champ d'application de chaque passage : la démarche et le public concernés.
N'applique pas une exception ou un régime particulier à une question générale.
Sans précision dans la question, ne suppose pas que la personne est chercheur,
étudiant, réfugié ou titulaire d'un titre particulier. Si les extraits ne donnent
que des règles spéciales sans la règle générale demandée, réponds exactement :
L'information n'est pas disponible dans les documents.
Si plusieurs régimes sont présents, distingue la règle générale des dérogations.
Conserve les chiffres, les unités et les conditions du texte. Réponds brièvement
et indique le nom du document et la page qui justifient la réponse.

Contexte : {context}

Question : {question}

Réponse (en français) :"""


@lru_cache(maxsize=1)
def load_flan():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.eval()
    return tokenizer, model


def _generate_flan(prompt, temperature):
    import torch

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


def _generate_llama(prompt, temperature, model=LLAMA_MODEL):
    result = _ollama.generate(
        model=model,
        prompt=prompt,
        keep_alive=OLLAMA_KEEP_ALIVE,
        options={
            "temperature": temperature,
            "seed": SEED,
            "num_predict": MAX_NEW_TOKENS,
            **({"num_ctx": 2048} if model == LLAMA_LIGHT_MODEL else {}),
        },
    )
    return result["response"].strip()


def generate_response(question, context="", model=DEFAULT_MODEL, temperature=0.0):
    """Génère une réponse. Lève ConnectionError si Ollama ne tourne pas."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("La question est vide.")
    if not 0 <= temperature <= 1:
        raise ValueError("La temperature doit etre entre 0 et 1.")
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    if not context or not context.strip():
        return "L'information n'est pas disponible dans les documents."
    prompt = build_prompt(question, context)
    if model in (LLAMA_MODEL, LLAMA_LIGHT_MODEL):
        try:
            return _generate_llama(prompt, temperature, model)
        except ollama.ResponseError as error:
            raise RuntimeError(f"Ollama : {error}. Verifiez ollama pull {model}.") from error
    if model == FLAN_MODEL:
        return _generate_flan(prompt, temperature)
    raise ValueError(f"Unknown model: {model}")
