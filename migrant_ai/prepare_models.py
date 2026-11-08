"""Explicit official model download with resumable HTTP ranges and SHA-256 verification."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

MODELS=["Narrativa/distilroberta-finetuned-stereotype-detection","Hate-speech-CNERG/dehatebert-mono-spanish","facebook/bart-large-cnn","NousResearch/Meta-Llama-3-8B-Instruct"]
TOKEN_FILES={"config.json","generation_config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.json","merges.txt","vocab.txt","tokenizer.model","added_tokens.json"}

def request(url,headers=None):
    return urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"class-project-reproduction",**(headers or {})}),timeout=60)

def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def download(name,revision,entry,folder,workers=8):
    filename=entry['rfilename']; path=folder/filename; size=entry['size']
    expected=entry.get('lfs',{}).get('sha256')
    if path.exists() and path.stat().st_size==size and (not expected or digest(path)==expected): return
    base=f'https://huggingface.co/{name}/resolve/{revision}/{filename}?download=true'
    temporary=path.with_suffix(path.suffix+'.part')
    if size<8*1024*1024:
        with request(base) as response,temporary.open('wb') as output:
            while block:=response.read(1024*1024): output.write(block)
    else:
        width=8*1024*1024
        parts=folder/(filename+'.parts');parts.mkdir(exist_ok=True)
        # Reuse full chunks of an earlier interrupted sequential download.
        if temporary.exists():
            with temporary.open('rb') as previous:
                for number in range(temporary.stat().st_size//width):
                    target=parts/str(number)
                    block=previous.read(width)
                    if not target.exists(): target.write_bytes(block)
        def get_part(number):
            start=number*width;end=min(size,start+width)-1;target=parts/str(number)
            if target.exists() and target.stat().st_size==end-start+1: return number
            for attempt in range(3):
                try:
                    with request(base+f'&part={number}&attempt={attempt}',{'Range':f'bytes={start}-{end}'}) as response:
                        if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                            raise RuntimeError('Server did not honor the exact byte range')
                        data=response.read(end-start+1)
                    if len(data)!=end-start+1: raise RuntimeError('Incomplete byte range')
                    target.write_bytes(data)
                    return number
                except Exception:
                    if attempt==2: raise
                    time.sleep(1)
        count=math.ceil(size/width)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for done,_ in enumerate(pool.map(get_part,range(count)),1):
                if done%10==0 or done==count: print(name,filename,f'{done}/{count} chunks',flush=True)
        with temporary.open('wb') as output:
            for number in range(count):
                with (parts/str(number)).open('rb') as chunk:
                    while block:=chunk.read(1024*1024): output.write(block)
    if temporary.stat().st_size!=size: raise RuntimeError('Size mismatch '+filename)
    if expected and digest(temporary)!=expected: raise RuntimeError('SHA256 mismatch '+filename)
    temporary.replace(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models',nargs='*',help='Official model IDs; defaults to the paper models and Llama tokenizer')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if set(args.models)-set(MODELS): parser.error('Unknown model identifier')
    if not 1 <= args.workers <= 16: parser.error('Workers must be between 1 and 16')
    root=Path('.runtime/paper_models');root.mkdir(parents=True,exist_ok=True)
    for name in args.models or MODELS:
        with request('https://huggingface.co/api/models/'+name+'?blobs=true') as response: info=json.load(response)
        revision=info['sha'];folder=root/name.replace('/','--');folder.mkdir(exist_ok=True)
        names={x['rfilename'] for x in info['siblings']}
        weight='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
        entries=[x for x in info['siblings'] if '/' not in x['rfilename'] and (x['rfilename'] in TOKEN_FILES or (name!=MODELS[-1] and x['rfilename']==weight))]
        entries.sort(key=lambda x:x['size'])
        manifest={'model':name,'revision':revision,'files':{}}
        for entry in entries:
            print('PREPARING',name,entry['rfilename'],entry['size'],flush=True)
            download(name,revision,entry,folder,args.workers)
            manifest['files'][entry['rfilename']]=digest(folder/entry['rfilename'])
        (folder/'source.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print('READY',name,revision,flush=True)

if __name__=='__main__': main()
