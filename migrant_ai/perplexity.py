"""Teacher-forced causal perplexity; never derived from generation time or BLEU."""
import math


def windows(n_tokens, max_length, stride):
    """Yield (start, end, first_target). Each target except token zero is scored once."""
    if n_tokens < 2 or not 1 <= stride < max_length:
        raise ValueError("At least two tokens and 1 <= stride < max_length required")
    scored = 1
    while scored < n_tokens:
        start = max(0, scored - (max_length - stride)) if scored > 1 else 0
        end = min(n_tokens, start + max_length)
        yield start, end, scored - start
        scored = end


def causal_perplexity(tokens, loss_sum, max_length=512, stride=256):
    """loss_sum(window, first_target) returns summed NLL, not a mean over padding."""
    total, count = 0.0, 0
    for start, end, first in windows(len(tokens), max_length, stride):
        nll = float(loss_sum(tokens[start:end], first))
        if not math.isfinite(nll) or nll < -1e-5:
            raise ValueError("Invalid negative log-likelihood")
        total += max(0.0, nll)
        count += end - start - first
    average = total / count
    # Overflow is a failed metric, never silently serialized as Infinity.
    perplexity = math.exp(average)
    if not math.isfinite(perplexity):
        raise ValueError("Non-finite perplexity")
    return {"perplexity": perplexity, "perplexity_nll": average,
            "perplexity_tokens": count, "perplexity_max_length": max_length,
            "perplexity_stride": stride, "perplexity_scope": "prediction_unconditional",
            "perplexity_method": "teacher-forcing-sliding-v1"}


class TransformersPerplexity:
    def __init__(self, model_id, revision=None, device="cpu", max_length=512, stride=256):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype="auto").to(device).eval()
        self.model_id, self.revision = model_id, revision
        self.max_length = min(max_length, self.model.config.max_position_embeddings)
        self.stride = min(stride, self.max_length - 1)
        self.device = device

    def __call__(self, text):
        import torch
        import torch.nn.functional as F
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        def loss_sum(part, first):
            ids = torch.tensor([part], device=self.device)
            with torch.inference_mode():
                logits = self.model(input_ids=ids, use_cache=False).logits
                return F.cross_entropy(logits[0, first-1:-1, :].float(),
                    ids[0, first:], reduction="sum").item()
        result = causal_perplexity(tokens, loss_sum, self.max_length, self.stride)
        return {**result, "perplexity_model": self.model_id,
                "perplexity_revision": getattr(self.model.config, "_commit_hash", None)}


class GGUFPerplexity:
    """Optional CPU backend for the *same* quantized GGUF used for generation."""
    def __init__(self, path, max_length=512, stride=256, threads=4):
        import hashlib
        from pathlib import Path
        from llama_cpp import Llama
        path = Path(path)
        # Streaming hash avoids reading a multi-GB model into RAM.
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
        self.digest = digest.hexdigest()
        self.model = Llama(model_path=str(path), n_ctx=max_length, n_batch=64,
            n_threads=threads, logits_all=True, n_gpu_layers=0, verbose=False)
        self.max_length, self.stride = max_length, stride

    def __call__(self, text):
        import numpy as np
        tokens = self.model.tokenize(text.encode("utf-8"), add_bos=True)
        def loss_sum(part, first):
            self.model.reset()
            self.model.eval(part)
            total = 0.0
            for i in range(first, len(part)):
                logits = np.asarray(self.model.scores[i-1], dtype=np.float64)
                maximum = float(logits.max())
                total += maximum + math.log(float(np.exp(logits-maximum).sum())) - float(logits[part[i]])
            return total
        return {**causal_perplexity(tokens, loss_sum, self.max_length, self.stride),
                "perplexity_model": "gguf-sha256:" + self.digest,
                "perplexity_revision": self.digest}
