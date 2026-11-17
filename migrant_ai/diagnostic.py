"""Diagnostic sans telechargement, depuis n'importe quel dossier."""
import hashlib
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def inspect_project():
    payload = json.loads((ROOT / "tests/fixtures/source_manifest.json").read_text(encoding="utf-8"))
    manifest = payload["files"]
    def file_digest(path):
        data=path.read_bytes()
        if path.suffix in payload["text_extensions"]: data=data.replace(b"\r\n",b"\n")
        return hashlib.sha256(data).hexdigest()
    changed = [name for name, expected_digest in manifest.items()
        if not (ROOT/name).is_file() or file_digest(ROOT/name) != expected_digest]
    packages = ["streamlit", "torch", "transformers", "sentence_transformers", "sacrebleu", "rouge_score", "pandas", "ollama", "faiss", "pypdf", "codecarbon"]
    report = {"python":sys.version.split()[0],"root":str(ROOT),
        "original_files":len(manifest),"originals_unchanged":not changed,"changed_originals":changed,
        "dependencies":{name:importlib.util.find_spec(name) is not None for name in packages}}
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags",timeout=3) as response:
            models=json.load(response).get("models",[])
        report["ollama"]="running"
        report["models"]=[m["name"] for m in models]
        report["llama3_standard_ready"]="llama3:8b" in report["models"]
        report["llama3_light_ready"]="llama3:8b-instruct-q2_K" in report["models"]
        report["llama3_ready"]=report["llama3_standard_ready"] or report["llama3_light_ready"]
    except Exception as error:
        report.update(ollama="unavailable",llama3_ready=False,ollama_error=str(error))
    try:
        import psutil
        memory=psutil.virtual_memory()
        report["memory_gib"]={"total":round(memory.total/2**30,2),"available":round(memory.available/2**30,2)}
    except ImportError:
        pass
    return report

if __name__ == "__main__":
    print(json.dumps(inspect_project(),ensure_ascii=True,indent=2))
