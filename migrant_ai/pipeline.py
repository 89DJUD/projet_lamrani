"""Génération avec contexte et provenance réellement utilisés."""
import json
from .generation import DEFAULT_MODEL, PROMPT_VERSION, generate_response

def generate_with_sources(question, context=None, model=DEFAULT_MODEL,
                          temperature=0.0, top_k=2, chunk_size=300):
    passages=[]
    origin="manual"
    skipped=False
    if context is None:
        from rag import retrieve, format_context, NO_INFO_ANSWER
        origin="retrieval"
        passages=retrieve(question,top_k,chunk_size)
        context=format_context(passages)
        if not passages:
            prediction=NO_INFO_ANSWER
            skipped=True
    if not skipped:
        prediction=generate_response(question,context,model,temperature)
        skipped=not bool(context and context.strip())
    sources=[{key:p[key] for key in ("source","page","score")} for p in passages]
    return {"prediction":prediction,"context":context or "",
            "retrieved_sources":json.dumps(sources,ensure_ascii=False),
            "context_origin":origin,"generation_skipped":skipped,
            "prompt_version":PROMPT_VERSION}
