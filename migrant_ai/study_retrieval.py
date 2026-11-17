"""Separate FAISS cache for study models; never overwrites original indexes."""
import hashlib
import json
from pathlib import Path
from .study_protocol import sha256


class StudyRetriever:
    def __init__(self, model_id, corpus="corpus", revision=None, cache=".runtime/study_indexes", threshold=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_id, revision=revision)
        self.model_id, self.revision = model_id, revision
        self.corpus, self.cache = Path(corpus), Path(cache)
        self.threshold = threshold
        self.loaded = {}

    def retrieve(self, question, chunk_size, top_k):
        import faiss
        import numpy as np
        from rag import load_pdfs, make_chunks
        signature = {"model":self.model_id,"revision":self.revision,
            "resolved_revision":getattr(self.model[0].auto_model.config,"_commit_hash",None),
            "corpus":{p.name:sha256(p) for p in sorted(self.corpus.glob("*.pdf"))},
            "chunk_size":chunk_size,"rag_sha256":sha256("rag.py")}
        identity = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        folder = self.cache / identity
        if identity not in self.loaded:
            if (folder/"faiss.index").is_file() and (folder/"chunks.json").is_file():
                index = faiss.read_index(str(folder/"faiss.index"))
                chunks = json.loads((folder/"chunks.json").read_text(encoding="utf-8"))
            else:
                chunks = make_chunks(load_pdfs(self.corpus),chunk_size)
                if not chunks:
                    raise ValueError("Empty corpus")
                vectors = np.asarray(self.model.encode([p["text"] for p in chunks],
                    normalize_embeddings=True, batch_size=16), dtype="float32")
                index = faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
                folder.mkdir(parents=True,exist_ok=True)
                faiss.write_index(index,str(folder/"faiss.index"))
                (folder/"chunks.json").write_text(json.dumps(chunks,ensure_ascii=False),encoding="utf-8")
            self.loaded[identity] = index, chunks
        index,chunks = self.loaded[identity]
        query = np.asarray(self.model.encode([question],normalize_embeddings=True),dtype="float32")
        scores,ids = index.search(query,min(top_k,index.ntotal))
        return [{**chunks[i],"score":float(s)} for i,s in zip(ids[0],scores[0])
                if i>=0 and (self.threshold is None or s>=self.threshold)]
