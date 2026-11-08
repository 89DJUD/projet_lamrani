"""Validated experiment plans: a subset is never labelled a complete reproduction."""
import hashlib
import itertools
import json
from pathlib import Path
import pandas as pd

CHUNK_SIZES = [50, 100, 300, 500, 1000]
TOP_K = [1, 2, 5, 8, 10]
TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configurations(config):
    sizes, ks, temperatures = (config[k] for k in ("chunk_sizes", "top_k", "temperatures"))
    if not sizes or not ks or not temperatures:
        raise ValueError("Empty hyperparameter grid")
    if any(type(x) is not int or x not in CHUNK_SIZES for x in sizes):
        raise ValueError("Unsupported chunk size")
    if any(type(x) is not int or x not in TOP_K for x in ks):
        raise ValueError("Unsupported top-k")
    if any(type(x) not in (int,float) or not 0 <= x <= 1 for x in temperatures):
        raise ValueError("Temperature must be in [0,1]")
    grid = list(itertools.product(sizes, ks, temperatures))
    if len(set(grid)) != len(grid):
        raise ValueError("Duplicate configurations")
    variants = config.get("variants", ["baseline", "integrated"])
    if not variants or len(set(variants)) != len(variants) or set(variants)-{"baseline","integrated"}:
        raise ValueError("Invalid variants")
    if config.get("language") not in ("fr", "es", "en"):
        raise ValueError("Explicit language fr/es/en required")
    return grid


def load_dataset(path, expected=None, language=None):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Question/reference file absent: {path}; do not substitute published metric rows")
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.empty or not {"question", "reference"}.issubset(frame):
        raise ValueError("Non-empty question/reference CSV required")
    if not frame[["question","reference"]].map(lambda t:isinstance(t,str) and bool(t.strip())).all().all():
        raise ValueError("Empty question or reference")
    if frame.question.str.strip().duplicated().any():
        raise ValueError("Duplicate questions")
    if expected is not None and len(frame) != expected:
        raise ValueError(f"Expected {expected} questions, got {len(frame)}")
    if "language" in frame and not frame.language.eq(language).all():
        raise ValueError("Dataset language differs from protocol")
    return frame


def plan(config):
    grid = configurations(config)
    path = Path(config["test_set"])
    present = path.is_file()
    count = len(load_dataset(path,config.get("expected_questions"),config["language"])) if present else config.get("expected_questions")
    return {"scope":config["scope"],"dataset":str(path),"dataset_present":present,
            "n_configurations":len(grid),"variants":config["variants"],"questions":count,
            "n_rows":len(grid)*len(config["variants"])*count if count else None,
            "reproduction_guaranteed":False,
            "warning":"Methodological plan; corpus, prompts, classifier thresholds and unspecified paper details require documentation."}


def protocol(config):
    data = load_dataset(config["test_set"], config.get("expected_questions"), config["language"])
    paths = [Path(config["test_set"]), *sorted(Path(config["corpus"]).glob("*.pdf")),
             *sorted(Path("migrant_ai").glob("*.py")), Path("rag.py")]
    if not list(Path(config["corpus"]).glob("*.pdf")):
        raise FileNotFoundError("Corpus PDF files absent")
    model_ids={config['embedding_model'],config['similarity_model'],config['summary_model']}
    if config['social_backend']=='paper':
        model_ids.update({'Narrativa/distilroberta-finetuned-stereotype-detection','Hate-speech-CNERG/dehatebert-mono-spanish'})
    else:
        model_ids.add('MoritzLaurer/mDeBERTa-v3-base-mnli-xnli')
    model_ids.add(config['generation'].get('tokenizer_model',config['generation']['model']))
    sources={}
    for model_id in sorted(model_ids):
        marker=Path('.runtime/paper_models')/model_id.replace('/','--')/'source.json'
        if marker.is_file(): sources[model_id]=json.loads(marker.read_text(encoding='utf-8'))
    manifest = {"settings":config,"sha256":{str(p):sha256(p) for p in paths},
                "model_snapshots":sources,"metric_version":"paper-methods-v1"}
    identity = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    return {**manifest,"study_id":identity}, data
