import time
from functools import partial
from pathlib import Path

import ollama
import streamlit as st

from evaluation import evaluate_response, evaluate_test_set
from llm import LLAMA_MODEL, MODELS, generate_response
from rag import (
    CHUNK_SIZES,
    CORPUS_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    TOP_K_VALUES,
    available_chunk_sizes,
    build_all_indexes,
    format_context,
    index_exists,
    load_index,
    NO_INFO_ANSWER,
    SIMILARITY_THRESHOLD,
    retrieve,
)

TEST_SET_PATH = "test_set.csv"
METRICS = {
    "bleu": ("BLEU", "Suites de mots identiques à la réponse de référence."),
    "rouge1": ("ROUGE-1", "Mots communs avec la réponse de référence."),
    "rouge2": ("ROUGE-2", "Paires de mots communes avec la référence."),
    "rougeL": ("ROUGE-L", "Plus longue suite de mots commune avec la référence."),
    "semantic_similarity": (
        "Similarité",
        "Proximité de sens avec la référence (embeddings de phrases).",
    ),
}
OLLAMA_DOWN = (
    "Ollama n'est pas lancé. Ouvrez l'application Ollama depuis le menu "
    "Démarrer, puis réessayez."
)

st.set_page_config(
    page_title="Assistant migrants UE",
    page_icon=":material/diversity_3:",
    layout="wide",
)

st.session_state.setdefault("answer", None)
st.session_state.setdefault("results", None)


@st.cache_data(ttl=10, show_spinner=False)
def ollama_is_running():
    try:
        ollama.list()
        return True
    except Exception:
        return False


def show_metrics(scores):
    with st.container(horizontal=True):
        for key, (label, help_text) in METRICS.items():
            st.metric(label, f"{scores[key]:.3f}", help=help_text, border=True)


def show_passage(p):
    st.markdown(
        f"**{p['source']}**, page {p['page']} · similarité {p['score']:.2f}"
    )
    st.caption(p["text"])


# --- Barre latérale : réglages -----------------------------------------------
with st.sidebar:
    st.header("Réglages", icon=":material/tune:")

    model = st.segmented_control(
        "Modèle",
        MODELS,
        default=LLAMA_MODEL,
        required=True,
    )
    temperature = st.slider(
        "Température",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0 = toujours la même réponse. Plus haut = réponses plus variées.",
    )

    if model == LLAMA_MODEL:
        if ollama_is_running():
            st.badge("Ollama lancé", icon=":material/check_circle:", color="green")
        else:
            st.badge("Ollama arrêté", icon=":material/error:", color="red")
            st.caption("Lancez l'application Ollama depuis le menu Démarrer.")
        st.caption(
            "Llama 3 tourne sur le processeur : 20 à 60 s par réponse, et "
            "environ 1 min de plus la première fois."
        )
    else:
        st.caption("Petit modèle rapide, seulement pour tester la chaîne.")

    st.subheader("Recherche (RAG)", icon=":material/manage_search:")
    built_sizes = available_chunk_sizes()
    chunk_size = st.segmented_control(
        "Taille des passages (mots)",
        CHUNK_SIZES,
        default=DEFAULT_CHUNK_SIZE,
        required=True,
        help="Longueur des morceaux de texte indexés. Petits passages = "
        "recherche précise ; grands passages = plus de contexte.",
    )
    top_k = st.segmented_control(
        "Nombre de passages (top-K)",
        TOP_K_VALUES,
        default=DEFAULT_TOP_K,
        required=True,
        help="Nombre de passages donnés au modèle pour répondre.",
    )
    if chunk_size not in built_sizes:
        st.caption(
            f":material/info: Pas d'index pour {chunk_size} mots. "
            "Construisez les index dans l'onglet Documents."
        )

    st.space("medium")
    st.caption(
        ":material/warning: Prototype. Les réponses s'appuient sur des textes "
        "officiels (directives UE, HCR) mais peuvent contenir des erreurs ou "
        "être dépassées. Ce n'est pas un avis juridique : vérifiez auprès de "
        "l'administration ou d'une association."
    )

generate = partial(generate_response, model=model, temperature=temperature)


def generate_with_rag(question, context=None):
    """Sans contexte fourni, cherche le contexte dans les documents."""
    if context is None:
        passages = retrieve(question, top_k, chunk_size)
        if not passages:
            return NO_INFO_ANSWER
        context = format_context(passages)
    return generate(question, context)


# --- En-tête -------------------------------------------------------------------
st.title("Assistant pour les migrants", icon=":material/diversity_3:")
st.caption(
    "Human-centered AI for migrant integration · RAG sur les directives "
    "européennes et le guide du HCR"
)

ask_tab, docs_tab, eval_tab = st.tabs(
    [
        ":material/chat: Poser une question",
        ":material/folder_open: Documents (RAG)",
        ":material/fact_check: Évaluation",
    ]
)

# --- Onglet 1 : poser une question --------------------------------------------
with ask_tab:
    left, right = st.columns([3, 2], gap="large")

    with left:
        question = st.text_area(
            "Votre question",
            "Quels membres de ma famille peuvent me rejoindre au titre du "
            "regroupement familial ?",
            height=100,
        )
        has_index = index_exists(chunk_size)
        use_rag = st.toggle(
            "Chercher dans les documents (RAG)",
            value=has_index,
            disabled=not has_index,
            help="Trouve les passages les plus pertinents dans les PDF "
            "indexés et les donne au modèle comme contexte.",
        )
        if not has_index:
            st.caption(
                ":material/info: Pas encore d'index. Ajoutez des PDF et "
                "construisez l'index dans l'onglet Documents."
            )
        if use_rag:
            context = None
        else:
            context = st.text_area(
                "Contexte (manuel)",
                "Exemple de contexte : le regroupement familial peut concerner "
                "le conjoint et les enfants mineurs non mariés.",
                height=120,
                help="Utilisé quand la recherche dans les documents est désactivée.",
            )
        with st.expander(
            "Réponse de référence (facultatif)", icon=":material/rule:"
        ):
            reference = st.text_area(
                "Réponse de référence",
                "",
                height=80,
                label_visibility="collapsed",
                placeholder="Collez ici la réponse attendue pour obtenir les scores.",
            )
            st.caption("Laissez vide pour ne pas calculer les scores.")

        if st.button(
            "Générer la réponse",
            type="primary",
            icon=":material/send:",
            width="stretch",
        ):
            if not question.strip():
                st.error("Veuillez saisir une question.", icon=":material/error:")
            else:
                passages = None
                if use_rag:
                    with st.spinner("Recherche dans les documents..."):
                        passages = retrieve(question, top_k, chunk_size)
                        context = format_context(passages)
                try:
                    with st.spinner(f"Génération avec {model}..."):
                        start = time.perf_counter()
                        # Aucun passage au-dessus du seuil : question hors
                        # corpus, on ne sollicite pas le LLM.
                        if passages == []:
                            answer = NO_INFO_ANSWER
                        else:
                            answer = generate(question, context)
                        elapsed = time.perf_counter() - start
                except ConnectionError:
                    st.error(OLLAMA_DOWN, icon=":material/error:")
                else:
                    scores = None
                    if reference.strip():
                        with st.spinner("Calcul des scores..."):
                            scores = evaluate_response(answer, reference)
                    st.session_state.answer = {
                        "question": question,
                        "text": answer,
                        "model": model,
                        "temperature": temperature,
                        "seconds": elapsed,
                        "scores": scores,
                        "passages": passages,
                        "rag": f"passages de {chunk_size} mots · top-{top_k}"
                        if use_rag
                        else "contexte manuel",
                    }

    with right:
        result = st.session_state.answer
        if result is None:
            with st.container(border=True, height="stretch"):
                st.caption(
                    ":material/info: La réponse s'affichera ici. Écrivez une "
                    "question, puis cliquez sur Générer la réponse."
                )
        else:
            with st.container(border=True):
                st.subheader("Réponse", icon=":material/smart_toy:")
                st.caption(f"Q : {result['question']}")
                st.markdown(result["text"])
                st.caption(
                    f"{result['model']} · température {result['temperature']} "
                    f"· {result['rag']} · {result['seconds']:.1f} s"
                )
                if result["passages"]:
                    with st.expander(
                        f"Sources ({len(result['passages'])} passages)",
                        icon=":material/description:",
                    ):
                        for p in result["passages"]:
                            show_passage(p)

    if result is not None and result["scores"] is not None:
        st.subheader("Scores", icon=":material/analytics:")
        show_metrics(result["scores"])

# --- Onglet 2 : documents et index (RAG) --------------------------------------
with docs_tab:
    st.caption(
        f"Les PDF sont rangés dans le dossier `{CORPUS_DIR}/`. Après un ajout "
        "ou une suppression, reconstruisez l'index. Les PDF scannés (images) "
        "ne peuvent pas être lus."
    )

    uploads = st.file_uploader(
        "Ajouter des documents PDF",
        type="pdf",
        accept_multiple_files=True,
    )
    if uploads:
        CORPUS_DIR.mkdir(exist_ok=True)
        for upload in uploads:
            target = CORPUS_DIR / Path(upload.name).name
            if not target.exists():
                target.write_bytes(upload.getvalue())

    pdfs = sorted(CORPUS_DIR.glob("*.pdf")) if CORPUS_DIR.exists() else []
    with st.container(horizontal=True):
        st.metric("Fichiers PDF", len(pdfs), border=True)
        st.metric(
            "Index construits",
            f"{len(built_sizes)} / {len(CHUNK_SIZES)}",
            border=True,
        )

    if built_sizes:
        index_rows = []
        for size in CHUNK_SIZES:
            if index_exists(size):
                index, chunks = load_index(size)
                index_rows.append(
                    {"Taille des passages (mots)": size,
                     "Passages indexés": index.ntotal,
                     "Documents": len({c["source"] for c in chunks})}
                )
            else:
                index_rows.append(
                    {"Taille des passages (mots)": size,
                     "Passages indexés": None,
                     "Documents": None}
                )
        st.dataframe(
            index_rows,
            hide_index=True,
            alt="Index FAISS construits pour chaque taille de passage",
        )

    if pdfs:
        st.dataframe(
            {"Fichier": [p.name for p in pdfs],
             "Taille (Ko)": [round(p.stat().st_size / 1024) for p in pdfs]},
            hide_index=True,
            alt="Fichiers PDF du dossier corpus",
        )

    if st.button(
        f"Construire les {len(CHUNK_SIZES)} index",
        type="primary",
        icon=":material/database:",
        disabled=not pdfs,
        help="Un index FAISS par taille de passage (50 à 1000 mots). "
        "Compter plusieurs minutes.",
    ):
        progress = st.progress(0.0, text="Lecture des PDF...")

        def on_build_progress(done, total, size):
            progress.progress(
                done / total,
                text=f"Index {done + 1} sur {total} : passages de {size} mots...",
            )

        try:
            stats = build_all_indexes(on_progress=on_build_progress)
        except ValueError as error:
            progress.empty()
            st.error(str(error), icon=":material/error:")
        else:
            # Relance la page pour activer la recherche dans l'onglet Question.
            st.session_state.index_message = (
                f"{len(stats)} index construits : {stats[0]['documents']} "
                f"documents, {stats[0]['pages']} pages, de "
                f"{stats[-1]['chunks']} à {stats[0]['chunks']} passages selon "
                "la taille."
            )
            st.rerun()

    if message := st.session_state.pop("index_message", None):
        st.success(message, icon=":material/check_circle:")

    if index_exists(chunk_size):
        st.subheader("Tester la recherche", icon=":material/search:")
        st.caption(
            f"Passages de {chunk_size} mots, top-{top_k} (réglables dans la "
            "barre latérale)."
        )
        query = st.text_input(
            "Recherche", placeholder="ex. : droit au travail des demandeurs d'asile"
        )
        if query.strip():
            found = retrieve(query, top_k, chunk_size)
            if not found:
                st.info(
                    "Aucun passage assez proche de la recherche (score < "
                    f"{SIMILARITY_THRESHOLD:.2f}) : la question sort du corpus.",
                    icon=":material/search_off:",
                )
            for p in found:
                with st.container(border=True):
                    show_passage(p)

# --- Onglet 3 : évaluation du jeu de test -------------------------------------
with eval_tab:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(
            f"Génère et note chaque question de `{TEST_SET_PATH}` avec "
            f"**{model}**, température **{temperature}**, passages de "
            f"**{chunk_size} mots**, **top-{top_k}**."
        )
        run_eval = st.button(
            "Lancer l'évaluation",
            type="primary",
            icon=":material/play_arrow:",
        )

    if run_eval and not index_exists(chunk_size):
        st.error(
            f"Pas d'index pour des passages de {chunk_size} mots. Construisez "
            "les index dans l'onglet Documents.",
            icon=":material/error:",
        )
    elif run_eval:
        progress = st.progress(0.0, text="Démarrage...")

        def on_progress(done, total):
            progress.progress(
                done / total, text=f"Question {done + 1} sur {total}"
            )

        try:
            start = time.perf_counter()
            results = evaluate_test_set(
                TEST_SET_PATH, generate_with_rag, on_progress
            )
            elapsed = time.perf_counter() - start
        except ConnectionError:
            progress.empty()
            st.error(OLLAMA_DOWN, icon=":material/error:")
        else:
            progress.empty()
            results.insert(0, "model", model)
            results.insert(1, "temperature", temperature)
            results.insert(2, "chunk_size", chunk_size)
            results.insert(3, "top_k", top_k)
            st.session_state.results = {"df": results, "seconds": elapsed}

    saved = st.session_state.results
    if saved is None:
        st.info(
            "Aucune évaluation pour l'instant. Cliquez sur Lancer "
            "l'évaluation pour noter le jeu de test.",
            icon=":material/info:",
        )
    else:
        df = saved["df"]
        run_model, run_temp = df["model"].iloc[0], df["temperature"].iloc[0]
        run_chunk, run_k = df["chunk_size"].iloc[0], df["top_k"].iloc[0]
        st.caption(
            f"Dernière évaluation : {run_model} · température {run_temp} · "
            f"passages de {run_chunk} mots · top-{run_k} · "
            f"{len(df)} questions · {saved['seconds']:.0f} s"
        )

        st.subheader("Scores moyens", icon=":material/analytics:")
        show_metrics(df[list(METRICS)].mean())

        st.subheader("Détail par question", icon=":material/table_chart:")
        score_columns = {
            key: st.column_config.ProgressColumn(
                label, help=help_text, format="%.2f", min_value=0, max_value=1
            )
            for key, (label, help_text) in METRICS.items()
        }
        st.dataframe(
            df.drop(
                columns=["model", "temperature", "chunk_size", "top_k", "context"],
                errors="ignore",
            ),
            hide_index=True,
            column_config={
                "question": st.column_config.TextColumn("Question", width="medium"),
                "reference": st.column_config.TextColumn("Référence"),
                "prediction": st.column_config.TextColumn(
                    "Réponse du modèle", width="large"
                ),
                **score_columns,
            },
            alt="Réponses générées et scores pour chaque question du jeu de test",
        )

        st.download_button(
            "Télécharger les résultats (CSV)",
            df.to_csv(index=False),
            file_name=(
                f"evaluation_{run_model.replace(':', '-')}_t{run_temp}"
                f"_chunk{run_chunk}_k{run_k}.csv"
            ),
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
        )
