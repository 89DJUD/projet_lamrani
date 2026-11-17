from .model_registry import location
"""Optional pre-generation filtering and BART summarization, without changing corpus files."""
from dataclasses import dataclass
import math

SUMMARY_MODEL = "facebook/bart-large-cnn"


@dataclass(frozen=True)
class ContextResult:
    context: str
    passages: list
    decisions: list
    summaries: list


class BartSummarizer:
    def __init__(self, model_id=SUMMARY_MODEL, revision=None, max_new_tokens=120):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(location(model_id), revision=revision)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(location(model_id), revision=revision).eval()
        self.model_id, self.max_new_tokens = model_id, max_new_tokens

    def __call__(self, text):
        import torch
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        self.last_details=[]
        if len(ids) <= self.max_new_tokens:
            self.last_details.append({'applied':False,'reason':'short_input','input_tokens':len(ids)})
            return text
        capacity = min(self.tokenizer.model_max_length, self.model.config.max_position_embeddings)
        usable = capacity - self.tokenizer.num_special_tokens_to_add(pair=False)
        parts = []
        # No silent truncation: summarize every input window separately.
        for start in range(0, len(ids), usable):
            inputs = self.tokenizer.prepare_for_model(ids[start:start+usable], return_tensors="pt")
            inputs={k:v.unsqueeze(0) if v.ndim==1 else v for k,v in inputs.items()}
            with torch.inference_mode():
                summary = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            emitted=summary[0]
            eos=self.model.config.eos_token_id
            finished=bool(len(emitted) and int(emitted[-1])==eos)
            self.last_details.append({'applied':True,'input_tokens':len(ids[start:start+usable]),
                'output_tokens':len(emitted)-1,'finished_with_eos':finished,
                'length_limit_reached':len(emitted)-1>=self.max_new_tokens})
            parts.append(self.tokenizer.decode(emitted, skip_special_tokens=True))
        return " ".join(parts)


def process_context(passages, classify=None, summarize=None, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0,1]")
    kept, decisions, summaries, blocks = [], [], [], []
    for passage in passages:
        scores = classify(passage["text"]) if classify else {}
        if classify and any(not math.isfinite(float(scores[k])) or not 0<=scores[k]<=1 for k in ('stereotype_score','hate_score')):
            raise ValueError('Invalid filtering score')
        rejected = bool(classify and any(scores[k] >= threshold for k in ("stereotype_score", "hate_score")))
        decisions.append({"source": passage["source"], "page": passage["page"],
                          "scores": scores, "kept": not rejected})
        if rejected:
            continue
        kept.append(passage)
        text = summarize(passage["text"]) if summarize else passage["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Summarizer returned empty context")
        if summarize:
            summaries.append({"source": passage["source"], "page": passage["page"], "text": text,
                              "details": getattr(summarize,'last_details',None)})
        # The locator is provenance, not a claim that the generated summary is a direct quote.
        blocks.append(f"[{passage['source']}, p. {passage['page']}] {text}")
    return ContextResult("\n\n".join(blocks), kept, decisions, summaries)
