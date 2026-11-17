"""Resumable real likelihood and five-category NLI scoring of saved French answers."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from .campaign import atomic_csv
from .run_study import score_stage
from .study_protocol import sha256
from .paper_metrics import lexical_metrics,summarize,MultilingualSocial
from .perplexity import GGUFPerplexity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gguf',type=Path,required=True)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(4)
    frame=pd.read_csv(args.input,keep_default_na=False)
    required={'model','chunk_size','top_k','temperature','question','reference','prediction'}
    if frame.empty or not required.issubset(frame): parser.error('Missing source columns')
    if not frame[['question','reference','prediction']].map(lambda s:isinstance(s,str) and bool(s.strip())).all().all():
        parser.error('Missing text')
    keys=['model','chunk_size','top_k','temperature','question']
    if frame.duplicated(keys).any(): parser.error('Duplicate source records')
    model_hash=sha256(args.gguf)
    manifest={'source_sha256':sha256(args.input),'gguf_sha256':model_hash,
        'method':'saved-french-answers-likelihood-nli-batch5-v3',
        'code_sha256':{str(p):sha256(p) for p in sorted(Path('migrant_ai').glob('*.py'))}}
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    marker=args.output/'protocol.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Protocol changed: use a new output directory')
    import portalocker
    with portalocker.Lock(str(args.output/'.campaign.lock'),timeout=0):
        marker.write_text(json.dumps(manifest,indent=2),encoding='utf-8',newline='\n')
        rows=frame.to_dict('records')
        for row in rows:
            row['record_id']=hashlib.sha256(json.dumps([row[k] for k in keys],ensure_ascii=False).encode()).hexdigest()
        lexical=score_stage(args.output,rows,'lexical',lambda:lambda r:lexical_metrics(r['prediction'],r['reference']))
        def social_factory():
            scorer=MultilingualSocial()
            def measure(row):
                return {**scorer(row['prediction']),'social_method':'independent-nli-mean-batch5-v3',
                    'social_model_metadata':json.dumps(scorer.metadata(),ensure_ascii=False)}
            return measure
        social=score_stage(args.output,rows,'social',social_factory)
        def likelihood_factory():
            scorer=GGUFPerplexity(args.gguf,max_length=256,stride=128)
            memo={}
            def measure(row):
                text=row['prediction']
                reused=text in memo
                if not reused: memo[text]=scorer(text)
                return {**memo[text],'perplexity_reused_identical_text':reused}
            return measure
        likelihood=score_stage(args.output,rows,'perplexity',likelihood_factory)
        results=[]
        for row in rows:
            key=row['record_id']
            additions={**social[key],**likelihood[key]}
            record={**row,**{k:v for k,v in lexical[key].items() if k not in row}}
            for name,value in additions.items():
                if name in record and name!='record_id': record[name+'_previous']=record[name]
                record[name]=value
            if 'bias_flag' in record: record['bias_flag_previous']=record['bias_flag']
            record['bias_flag']=max(record['stereotype_score'],record['hate_score'])>=0.5
            record['bias_threshold']=0.5
            record['bias_method']='independent-nli-mean-batch5-v3'
            record['additional_evaluation_id']=identity
            results.append(record)
        scored=pd.DataFrame(results)
        # Preserve original prompt/model identity; no new generations, no new energy claims.
        summary=summarize(scored,frame.question.unique())
        atomic_csv(scored,args.output/'responses.csv')
        atomic_csv(summary,args.output/'configurations.csv')
        atomic_csv(summary[summary.complete],args.output/'configurations_completes.csv')
        report={'status':'completed','responses':len(scored),'perplexity_available':int(scored.perplexity.notna().sum()),
                'social_method':'French NLI adaptation, independent uncalibrated scores',
                'generation_rerun':False,'energy_remeasured':False,
                'full_article_reproduction':False}
        (args.output/'status.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)

if __name__=='__main__': main()
