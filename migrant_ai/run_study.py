"""Paired baseline/integrated study with provenance, checkpointing and real metrics."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import pandas as pd
from .campaign import read_ledger,append_record,atomic_csv
from .study_protocol import configurations,protocol,plan,sha256


def release():
    from .fairness import load_classifier
    load_classifier.cache_clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError: pass


def social_model(config):
    from .paper_metrics import PaperSocial,MultilingualSocial
    if config['social_backend']=='paper': return PaperSocial()
    if config['social_backend']=='multilingual_nli': return MultilingualSocial()
    raise ValueError('Explicit social backend required')


def generate_record(config,question,reference,variant,c,k,temperature):
    from .study_retrieval import StudyRetriever
    from .study_generation import StudyGenerator
    from .context_processing import process_context,BartSummarizer
    from .footprint import EnergyMonitor
    # These estimates intentionally include loading, retrieval, filtering and summary.
    with EnergyMonitor(True,config['country']) as monitor:
        if config.get('retrieval_backend')=='original':
            from rag import retrieve,load_embedder,EMBEDDING_MODEL
            if config['embedding_model']!=EMBEDDING_MODEL or Path(config['corpus']).resolve()!=Path('corpus').resolve():
                raise ValueError('Original indexes require the original embedding model and corpus')
            passages=retrieve(question,k,c,min_score=config.get('retrieval_threshold',0.4))
            load_embedder.cache_clear()
        else:
            retriever=StudyRetriever(config['embedding_model'],config['corpus'],
                                     threshold=config.get('retrieval_threshold'))
            passages=retriever.retrieve(question,c,k)
            del retriever
        release()
        if variant=='integrated' and passages:
            social=social_model(config)
            selected=process_context(passages,classify=social,threshold=config['filter_threshold'])
            metadata=social.metadata()
            summary_metadata={}
            del social
            release()
            if selected.passages:
                summarizer=BartSummarizer(config['summary_model'])
                context_result=process_context(selected.passages,summarize=summarizer)
                context=context_result.context
                summaries=context_result.summaries
                from .model_registry import revision
                summary_metadata={'model':config['summary_model'],'revision':revision(config['summary_model']),
                    'input_policy':'all-token-windows-no-silent-truncation-short-passages-preserved'}
                del summarizer
                release()
            else:
                context,summaries='',[]
            decisions=selected.decisions
        else:
            result=process_context(passages)
            context,decisions,summaries=result.context,result.decisions,[]
            metadata={}
            summary_metadata={}
        generator=StudyGenerator(config['generation'],config['language'])
        generated=generator(question,context,temperature)
        generator.close()
        del generator
        release()
    if not generated['prediction'].strip(): raise ValueError('Empty generation')
    return {'question':question,'reference':reference,'variant':variant,
        'model':config['generation']['model'],'chunk_size':c,'top_k':k,'temperature':temperature,
        **generated,**monitor.metrics,'energy_scope':'local_machine_retrieval_preprocessing_generation_with_loading',
        'retrieved_sources':json.dumps(passages,ensure_ascii=False),
        'context':context,'filter_decisions':json.dumps(decisions,ensure_ascii=False),
        'context_summaries':json.dumps(summaries,ensure_ascii=False),
        'filter_model_metadata':json.dumps(metadata,ensure_ascii=False),
        'summary_model_metadata':json.dumps(summary_metadata,ensure_ascii=False),
        'prompt_version':'study-chat-grounded-v1'}


def score_stage(output,rows,stage,factory):
    path=output/(stage+'.jsonl')
    existing=read_ledger(path)
    expected={row['record_id'] for row in rows}
    if set(existing)-expected: raise ValueError('Foreign scoring records in checkpoint')
    if set(existing)==expected: return existing
    scorer=factory()
    for row in rows:
        key=row['record_id']
        if key not in existing:
            scores=scorer(row)
            item={'record_id':key,**scores}
            # Reject NaN/Infinity before persisting anything.
            json.dumps(item,allow_nan=False)
            append_record(path,item)
            existing[key]=item
            print(f'{stage}: {len(existing)}/{len(rows)}',flush=True)
    del scorer
    release()
    return existing


def run(config,output,limit=None):
    from .paper_metrics import lexical_metrics,summarize
    manifest,data=protocol(config)
    expected_questions=data.question.tolist()
    if limit is not None:
        if limit<1: raise ValueError('Positive pilot limit required')
        data=data.head(limit)
        manifest['pilot_limit']=limit
        manifest['study_id']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    # Resolve the local Ollama tag to an immutable digest before recording protocol.
    if config['generation']['backend']=='ollama':
        import ollama
        selected=[m for m in ollama.list().models if m.model==config['generation']['model']]
        if not selected: raise RuntimeError('Requested Ollama model is not installed')
        manifest['ollama_digest']=selected[0].digest
        if config['perplexity']['backend']=='gguf':
            import re
            model_path=Path(config['perplexity']['model_path'])
            if not model_path.is_file(): raise FileNotFoundError(f'GGUF scorer missing: {model_path}')
            gguf_hash=sha256(model_path)
            description=ollama.show(config['generation']['model'])
            modelfile=description.get('modelfile','')
            matched=re.search(r'(?im)^FROM\s+.*sha256[-:]([0-9a-f]{64})',modelfile)
            if not matched or matched.group(1)!=gguf_hash:
                raise ValueError('Cannot establish that perplexity uses the same GGUF as generation')
            manifest['gguf_sha256']=gguf_hash
    manifest['study_id']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    marker=output/'protocol.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Changed protocol: use a new output directory')
    marker.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8',newline='\n')
    ledger=output/'generation.jsonl'
    saved=read_ledger(ledger)
    expected=[]
    grid=configurations(config)
    for variant in config['variants']:
        for c,k,t in grid:
            for q in data.itertuples():
                key=hashlib.sha256(json.dumps([variant,c,k,t,q.question],ensure_ascii=False).encode()).hexdigest()
                expected.append(key)
                if key in saved: continue
                row=generate_record(config,q.question,q.reference,variant,c,k,t)
                row.update(record_id=key,study_id=manifest['study_id'])
                if 'source' in data: row['source']=q.source
                append_record(ledger,row)
                saved[key]=row
                atomic_csv(pd.DataFrame(saved.values()),output/'responses_generation.csv')
                print(f'Generation: {len(saved)}/{len(data)*len(grid)*len(config["variants"])}',flush=True)
    if set(saved)!=set(expected): raise ValueError('Unexpected or missing generation records')
    rows=[saved[k] for k in expected]
    lexical=score_stage(output,rows,'lexical',lambda:lambda r:lexical_metrics(r['prediction'],r['reference']))
    def similarity_factory():
        from sentence_transformers import SentenceTransformer,util
        model=SentenceTransformer(config['similarity_model'])
        def score(row):
            embedding=model.encode([row['prediction'],row['reference']],convert_to_tensor=True)
            value=float(util.cos_sim(embedding[0],embedding[1]).item())
            if not math.isfinite(value): raise ValueError('Invalid cosine')
            return {'semantic_similarity':value,'similarity_model':config['similarity_model']}
        return score
    similarity=score_stage(output,rows,'similarity',similarity_factory)
    def social_factory():
        model=social_model(config)
        def score(row): return {**model(row['prediction']),
            'social_model_metadata':json.dumps(model.metadata(),ensure_ascii=False),'social_method':config['social_backend']}
        return score
    social=score_stage(output,rows,'social',social_factory)
    def perplexity_factory():
        from .perplexity import TransformersPerplexity,GGUFPerplexity
        settings=config['perplexity']
        if settings['backend']=='gguf':
            model=GGUFPerplexity(settings['model_path'],settings['max_length'],settings['stride'])
        elif settings['backend']=='transformers':
            model=TransformersPerplexity(settings['model'],device=settings.get('device','cpu'),
                max_length=settings['max_length'],stride=settings['stride'])
        else: raise ValueError('Unknown perplexity backend')
        return lambda row:model(row['prediction'])
    perplexity=score_stage(output,rows,'perplexity',perplexity_factory)
    frame=pd.DataFrame([{**r,**lexical[r['record_id']],**similarity[r['record_id']],
                         **social[r['record_id']],**perplexity[r['record_id']]} for r in rows])
    summary=summarize(frame,expected_questions)
    if limit is None and not summary.complete.all(): raise ValueError('Incomplete configurations')
    atomic_csv(frame,output/'responses.csv')
    atomic_csv(summary,output/'configurations.csv')
    atomic_csv(summary[summary.complete],output/'configurations_completes.csv')
    if set(config['variants'])=={'baseline','integrated'}:
        paired=paired_comparison(frame)
        atomic_csv(paired,output/'paired_comparison.csv')
    report={'status':'completed','pilot':limit is not None,'questions':len(data),
            'configurations_per_variant':len(grid),'variants':config['variants'],
            'eligible_for_final_ranking':limit is None,'rows':len(frame),'energy_available':int(frame.energy_kwh.notna().sum()),
            'perplexity_available':int(frame.perplexity.notna().sum()),
            'full_article_reproduction':False,'scope':config['scope']}
    (output/'status.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)



def paired_comparison(frame):
    keys=['model','chunk_size','top_k','temperature','question']
    left=frame[frame.variant=='baseline']
    right=frame[frame.variant=='integrated']
    merged=left.merge(right,on=keys,how='outer',suffixes=('_baseline','_integrated'),
                      validate='one_to_one',indicator=True)
    if not merged['_merge'].eq('both').all(): raise ValueError('Unpaired experimental variants')
    output=merged[keys].copy()
    metrics=['bleu','rouge1','rouge2','rougeL','semantic_similarity','perplexity',
             'stereotype_score','anti_stereotype_score','neutral_score','hate_score',
             'non_hate_score','seconds','energy_kwh','co2_kg']
    for key in metrics:
        if key+'_baseline' in merged and key+'_integrated' in merged:
            output[key+'_baseline']=merged[key+'_baseline']
            output[key+'_integrated']=merged[key+'_integrated']
            output[key+'_difference']=merged[key+'_integrated']-merged[key+'_baseline']
    return output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--run',action='store_true',help='Without this flag only the offline plan is displayed')
    parser.add_argument('--limit',type=int,help='Explicit pilot: first N existing questions, both variants')
    args=parser.parse_args()
    config=json.loads(args.profile.read_text(encoding='utf-8'))
    import os
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    displayed=plan(config)
    if args.limit is not None:
        if args.limit<1: parser.error('Positive pilot limit required')
        displayed['pilot']=True
        displayed['questions']=min(args.limit,displayed['questions'])
        displayed['n_rows']=displayed['n_configurations']*len(displayed['variants'])*displayed['questions']
    print(json.dumps(displayed,indent=2),flush=True)
    if not args.run: return
    if args.output is None: parser.error('--output required for execution')
    # Validate inputs before creating partial results or loading any model.
    protocol(config)
    import torch
    torch.set_num_threads(4)
    import portalocker
    args.output.mkdir(parents=True,exist_ok=True)
    with portalocker.Lock(str(args.output/'.campaign.lock'),timeout=0):
        try: run(config,args.output,args.limit)
        except Exception as error:
            (args.output/'status.json').write_text(json.dumps({'status':'failed','error':str(error)},indent=2),encoding='utf-8')
            raise

if __name__=='__main__': main()
