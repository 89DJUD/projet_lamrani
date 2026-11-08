"""Six configurations, durable per-question checkpoints and resumable bias scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pandas as pd
from .experiments import summarize, KEYS

CONFIGURATIONS = [(c, k, t) for c, k in [(50, 2), (100, 5)] for t in (0.1, 0.5, 0.9)]

def read_ledger(path):
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["record_id"] in records:
            raise ValueError("Duplicate checkpoint record")
        records[row["record_id"]] = row
    return records

def append_record(path, row):
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(previous + json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def atomic_csv(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/evaluations/campagne_complete"))
    args = parser.parse_args()
    # Set before importing generation; release Ollama between responses on small PCs.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MIGRANT_OLLAMA_KEEP_ALIVE", "0")
    import portalocker
    args.output.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(str(args.output / ".campaign.lock"), timeout=0):
        run(args.output)

def run(output):
    import torch
    import ollama
    from .generation import LLAMA_LIGHT_MODEL, PROMPT_VERSION
    from .pipeline import generate_with_sources
    from .quality import evaluate_test_set, load_similarity_model
    from .fairness import detect_bias
    torch.set_num_threads(4)
    data = pd.read_csv("test_set.csv")
    if len(data) != 20 or data.question.duplicated().any():
        raise ValueError("Expected 20 unique questions")
    tags = ollama.list().models
    matching = [m for m in tags if m.model == LLAMA_LIGHT_MODEL]
    if not matching:
        raise RuntimeError("Llama Q2 is not installed")
    paths = [Path("test_set.csv"), *sorted(Path("index").rglob("*")),
             *sorted(Path("migrant_ai").glob("*.py")), Path("rag.py")]
    protocol = {"model": LLAMA_LIGHT_MODEL, "model_digest": matching[0].digest,
                "prompt_version": PROMPT_VERSION, "configurations": CONFIGURATIONS,
                "country": "MAR", "energy_scope": "retrieval_and_generation_machine_estimate",
                "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}}
    protocol = json.loads(json.dumps(protocol))
    target = output / "protocol.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Protocol changed: use a new output directory")
    target.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    ledger = output / "generation.jsonl"
    completed = read_ledger(ledger)
    total = len(data) * len(CONFIGURATIONS)
    print(f"Generation: {len(completed)}/{total}", flush=True)
    for c, k, temperature in CONFIGURATIONS:
        for position in range(len(data)):
            record_id = f"{c}:{k}:{temperature}:{position}"
            if record_id in completed:
                continue
            def generate(question, context):
                return generate_with_sources(question, context, model=LLAMA_LIGHT_MODEL,
                    temperature=temperature, top_k=k, chunk_size=c)
            result = evaluate_test_set(data.iloc[[position]], generate, measure_energy=True,
                                       bias_method="heuristic", country_iso_code="MAR")
            row = json.loads(result.to_json(orient="records", force_ascii=False))[0]
            row.update(dict(zip(KEYS, [LLAMA_LIGHT_MODEL, c, k, temperature])))
            row["record_id"] = record_id
            append_record(ledger, row)
            completed[record_id] = row
            atomic_csv(pd.DataFrame(completed.values()), output / "responses_generation.csv")
            print(f"Generation: {len(completed)}/{total}", flush=True)
    # NLI is outside the measured retrieval/generation interval and in a separate stage.
    import gc
    import rag
    load_similarity_model.cache_clear()
    rag.load_embedder.cache_clear()
    gc.collect()
    bias_path = output / "bias.jsonl"
    biases = read_ledger(bias_path)
    for record_id, row in completed.items():
        if record_id not in biases:
            bias = {"record_id": record_id, **detect_bias(row["prediction"], method="model")}
            append_record(bias_path, bias)
            biases[record_id] = bias
            print(f"Bias: {len(biases)}/{total}", flush=True)
    frame = pd.DataFrame([{**row, **biases[key]} for key, row in completed.items()])
    if len(frame) != total:
        raise ValueError("Incomplete campaign")
    for _, group in frame.groupby(KEYS):
        if len(group) != len(data) or set(group.question) != set(data.question):
            raise ValueError("Incomplete configuration")
    atomic_csv(frame, output / "responses.csv")
    summary = summarize(frame)
    summary["n_expected_questions"] = len(data)
    atomic_csv(summary, output / "configurations.csv")
    atomic_csv(summary, output / "configurations_completes.csv")
    print(f"COMPLETE: {total} responses, {len(summary)} configurations; energy {frame.energy_kwh.notna().sum()}/{total}", flush=True)

if __name__ == "__main__":
    main()
