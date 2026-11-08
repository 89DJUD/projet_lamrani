"""Use verified local snapshots when installed; otherwise use the official HF identifier."""
import json
from pathlib import Path

def location(model_id):
    folder=Path('.runtime/paper_models')/model_id.replace('/','--')
    marker=folder/'source.json'
    if marker.is_file():
        info=json.loads(marker.read_text(encoding='utf-8'))
        if info['model']!=model_id: raise ValueError('Local model identity mismatch')
        return str(folder)
    return model_id

def revision(model_id):
    marker=Path('.runtime/paper_models')/model_id.replace('/','--')/'source.json'
    return json.loads(marker.read_text(encoding='utf-8'))['revision'] if marker.is_file() else None
