"""Add documented metrics to saved answers, without regenerating or inventing results."""
import argparse
import gc
import json
from pathlib import Path
import pandas as pd
from .paper_metrics import lexical_metrics, summarize
from .study_protocol import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--social",choices=["none","paper","multilingual_nli"],default="none")
    parser.add_argument("--perplexity-gguf",type=Path)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory to preserve prior experiments")
    frame=pd.read_csv(args.input,keep_default_na=False)
    required={"question","reference","prediction","model","chunk_size","top_k","temperature"}
    if frame.empty or not required.issubset(frame): parser.error("Missing required response/configuration columns")
    if not frame[["question","reference","prediction"]].map(lambda x:isinstance(x,str) and bool(x.strip())).all().all():
        parser.error("Empty questions, references or predictions")
    if frame.duplicated(["model","chunk_size","top_k","temperature","question"]).any():
        parser.error("Duplicate response/configuration keys")
    # Retain prior metric columns unchanged. New recall/precision columns are computed from saved texts.
    scores=pd.DataFrame([lexical_metrics(p,r) for p,r in zip(frame.prediction,frame.reference)])
    for key in scores:
        if key not in frame: frame[key]=scores[key]
    if args.social != "none":
        from .paper_metrics import PaperSocial, MultilingualSocial
        scorer=PaperSocial() if args.social=="paper" else MultilingualSocial()
        results=pd.DataFrame([scorer(text) for text in frame.prediction])
        for key in results:
            if key in frame: frame[key+"_previous"]=frame[key]
            frame[key]=results[key]
        frame["social_method"]=args.social
        frame["social_model_metadata"]=json.dumps(scorer.metadata(),ensure_ascii=False)
        del scorer
        from .fairness import load_classifier
        load_classifier.cache_clear()
        gc.collect()
    else:
        frame["social_additional_status"]="not_run_previous_scores_preserved"
    if args.perplexity_gguf:
        from .perplexity import GGUFPerplexity
        scorer=GGUFPerplexity(args.perplexity_gguf)
        values=pd.DataFrame([scorer(text) for text in frame.prediction])
        for key in values: frame[key]=values[key]
        frame["perplexity_status"]="measured"
    elif "perplexity" not in frame:
        frame["perplexity"]=None
        frame["perplexity_status"]="not_run"
    frame["additional_evaluation_version"]="saved-responses-rouge-details-v1"
    args.output.mkdir(parents=True)
    frame.to_csv(args.output/"responses.csv",index=False,lineterminator="\n")
    # Coverage is against the input questions, not an invented 91-question dataset.
    summary=summarize(frame,expected_questions=frame.question.unique())
    summary.to_csv(args.output/"configurations.csv",index=False,lineterminator="\n")
    summary[summary.complete].to_csv(args.output/"configurations_completes.csv",index=False,lineterminator="\n")
    report={"source":str(args.input),"source_sha256":sha256(args.input),"rows":len(frame),
            "new_metrics":"ROUGE precision/recall; existing values retained",
            "social_option":args.social,"perplexity_measured":bool(args.perplexity_gguf),
            "generation_rerun":False,"energy_reconstructed":False,
            "full_article_reproduction":False}
    (args.output/"evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))

if __name__=="__main__": main()
