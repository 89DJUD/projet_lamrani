"""Essai reel de deux questions avec Llama 3 Q2_K, apres disponibilite des poids."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
MODEL = "llama3:8b-instruct-q2_K"
REPORT = ROOT / "reports/generation_validation.json"

def save(data):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(REPORT)

def model_ready():
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags",timeout=5) as response:
            return any(m["name"] == MODEL for m in json.load(response).get("models",[]))
    except Exception:
        return False

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait",action="store_true",help="Attendre la fin du telechargement, sans le relancer")
    parser.add_argument("--timeout-minutes",type=int,default=120)
    args=parser.parse_args()
    start=time.monotonic()
    status={"model":MODEL,"status":"waiting_for_model","generation_verified":False,
            "started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save(status)
    while not model_ready():
        if not args.wait or time.monotonic()-start >= args.timeout_minutes*60:
            status.update(status="model_unavailable",error="Poids absents ou telechargement non termine.")
            save(status)
            print(status["error"],flush=True)
            return 2
        time.sleep(30)
    status["status"]="running_real_generation"
    save(status)
    environment=os.environ.copy()
    environment.update(MIGRANT_OLLAMA_KEEP_ALIVE="0",HF_HUB_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    output=ROOT/"results/evaluations/essai_court"
    command=[sys.executable,"-m","migrant_ai.experiments","--model",MODEL,
        "--chunk-sizes","50","--top-k","2","--temperatures","0.1","--limit","2",
        "--energy","--country","MAR","--output",str(output)]
    print("Essai reel de Llama 3 Q2_K...",flush=True)
    try:
        completed=subprocess.run(command,cwd=ROOT,env=environment,text=True,
            encoding="utf-8",errors="replace",capture_output=True,timeout=1800)
        log=ROOT/"reports/generation_validation.log"
        log.write_text(completed.stdout+"\n"+completed.stderr,encoding="utf-8")
        status["log"]=str(log.relative_to(ROOT))
        if completed.returncode:
            status.update(status="failed",error=completed.stderr[-3000:],returncode=completed.returncode)
        else:
            with (output/"responses.csv").open(encoding="utf-8",newline="") as source:
                rows=list(csv.DictReader(source))
            valid=len(rows)==2 and all(row.get("prediction","").strip() and row["model"]==MODEL for row in rows)
            status.update(status="passed" if valid else "failed",generation_verified=valid,
                n_questions=len(rows),output=str(output.relative_to(ROOT)),
                n_energy_available=sum(bool(row.get("energy_kwh")) for row in rows))
    except Exception as error:
        status.update(status="failed",error=f"{type(error).__name__}: {error}")
    status["finished_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(status)
    print(json.dumps(status,ensure_ascii=True,indent=2),flush=True)
    return 0 if status["generation_verified"] else 1

if __name__=="__main__":
    raise SystemExit(main())
