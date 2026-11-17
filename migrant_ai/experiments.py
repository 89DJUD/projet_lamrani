"""Expériences de génération et évaluation des réponses."""
import argparse
from pathlib import Path
import pandas as pd
from .fairness import detect_bias

KEYS = ["model", "chunk_size", "top_k", "temperature"]

def configuration_keys(frame):
    return KEYS + (["prompt_version"] if "prompt_version" in frame else [])


def summarize(frame):
    metrics = [m for m in ["bleu", "rouge1", "rouge2", "rougeL", "semantic_similarity",
               "seconds", "stereotype_score", "hate_score", "energy_kwh", "co2_kg"] if m in frame]
    grouped = frame.groupby(configuration_keys(frame), dropna=False)
    summary = grouped[metrics].median().add_suffix("_median")
    summary["n_questions"] = grouped.size()
    if "energy_kwh" in frame:
        summary["n_energy_available"] = grouped["energy_kwh"].count()
    summary = summary.reset_index()
    summary.insert(0, "config", [f"C{i+1}" for i in range(len(summary))])
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-csv", type=Path, help="Reponses existantes : ajoute les biais, sans inventer l'energie historique")
    parser.add_argument("--test-set", default="test_set.csv")
    parser.add_argument("--output", type=Path, default=Path("results/evaluations"))
    parser.add_argument("--bias-method", choices=["heuristic", "model"], default="heuristic")
    parser.add_argument("--country", default="MAR")
    parser.add_argument("--energy", action="store_true")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[50, 100, 500])
    parser.add_argument("--top-k", type=int, nargs="+", default=[2])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--limit", type=int, help="Limiter les questions pour un essai reel court")
    parser.add_argument("--reevaluate", action="store_true", help="Recalculer les scores des reponses sauvegardees avec les metriques multilingues")
    parser.add_argument("--model", choices=["llama3:8b", "llama3:8b-instruct-q2_K"], default="llama3:8b")
    args = parser.parse_args()
    if args.from_csv:
        frame = pd.read_csv(args.from_csv)
        required = set(KEYS + ["question", "prediction"])
        if frame.empty or not required.issubset(frame):
            parser.error("CSV non vide attendu avec model, chunk_size, top_k, temperature, question, prediction")
        if not frame.prediction.map(lambda t: isinstance(t, str) and bool(t.strip())).all():
            parser.error("Predictions manquantes")
        if "prompt_version" not in frame:
            frame["prompt_version"]="unrecorded"
        if args.limit is not None:
            if args.limit < 1:
                parser.error("La limite doit etre positive")
            frame = frame.head(args.limit).reset_index(drop=True)
        if args.reevaluate:
            from .quality import evaluate_response, SIMILARITY_MODEL
            if "reference" not in frame or not frame.reference.map(lambda t: isinstance(t,str) and bool(t.strip())).all():
                parser.error("References non vides necessaires pour recalculer les scores")
            quality = pd.DataFrame([evaluate_response(row.prediction,row.reference) for row in frame.itertuples()])
            frame = frame.drop(columns=list(quality.columns), errors="ignore").join(quality)
            frame["evaluation_version"] = "multilingual-v1-reevaluated"
            frame["similarity_model"] = SIMILARITY_MODEL
        bias = pd.DataFrame([detect_bias(t, args.bias_method) for t in frame.prediction])
        frame = frame.drop(columns=list(bias.columns), errors="ignore").join(bias)
        # Aucune mesure historique disponible dans les fichiers originaux.
        if "energy_kwh" not in frame:
            frame["energy_kwh"] = None
            frame["co2_kg"] = None
            frame["energy_method"] = "unavailable_historical"
    else:
        from .quality import evaluate_test_set
        from .pipeline import generate_with_sources
        from rag import retrieve, format_context, NO_INFO_ANSWER, CHUNK_SIZES, TOP_K_VALUES
        if any(c not in CHUNK_SIZES for c in args.chunk_sizes) or any(k not in TOP_K_VALUES for k in args.top_k):
            parser.error("Taille de passage ou top-k non pris en charge")
        if any(not 0 <= t <= 1 for t in args.temperatures):
            parser.error("Temperature attendue entre 0 et 1")
        frames = []
        args.output.mkdir(parents=True, exist_ok=True)
        for c in args.chunk_sizes:
            for k in args.top_k:
                for t in args.temperatures:
                    def generate(question, context):
                        return generate_with_sources(question,context,model=args.model,
                            temperature=t,top_k=k,chunk_size=c)
                    current = evaluate_test_set(args.test_set, generate,
                        bias_method=args.bias_method, measure_energy=args.energy,
                        country_iso_code=args.country, limit=args.limit)
                    for key, value in zip(KEYS, [args.model, c, k, t]):
                        current[key] = value
                    frames.append(current)
                    # Checkpoint apres chaque configuration, resultat toujours separable.
                    current.to_csv(args.output / f"chunk{c}_k{k}_t{t}.csv", index=False)
        frame = pd.concat(frames, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "responses.csv", index=False)
    expected = pd.read_csv(args.test_set)
    if "question" not in expected or expected.empty or expected.question.duplicated().any():
        parser.error("Jeu de test invalide : questions uniques non vides attendues")
    expected_questions = set(expected.question)
    complete = frame.groupby(configuration_keys(frame), dropna=False).filter(
        lambda group: len(group) == len(expected_questions) and set(group.question) == expected_questions)
    all_configs = summarize(frame)
    all_configs["n_expected_questions"] = len(expected_questions)
    all_configs.to_csv(args.output / "configurations.csv", index=False)
    summarize(complete).to_csv(args.output / "configurations_completes.csv", index=False)
    print(f"{len(frame)} reponses exportees dans {args.output}")

if __name__ == "__main__":
    main()
