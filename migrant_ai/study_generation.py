from .model_registry import location
"""Generation with explicit token budgets, no silent context truncation."""
import gc

INSTRUCTIONS = {
 "fr":"Réponds en français uniquement à partir du contexte. Distingue la règle générale des exceptions. Si les informations manquent, indique-le. Cite le document et la page. Une synthèse peut être inexacte : ne déduis pas de nouvelles règles.",
 "es":"Responde en español usando únicamente el contexto. Distingue la regla general de las excepciones. Si falta información, indícalo. Cita el documento y la página. Un resumen puede contener errores: no deduzcas reglas nuevas.",
 "en":"Answer in English using only the context. Distinguish general rules from exceptions. If information is missing, say so. Cite the document and page. A summary may be inaccurate: do not infer new rules."
}
ABSTENTIONS = {"fr":"L'information n'est pas disponible dans les documents.",
              "es":"La información no está disponible en los documentos.",
              "en":"The information is not available in the documents."}


def prompt(question,context,language):
    return f"{INSTRUCTIONS[language]}\n\nContexte / Contexto / Context:\n{context}\n\nQuestion / Pregunta: {question}\n\nRéponse / Respuesta:"


class StudyGenerator:
    def __init__(self,settings,language):
        self.settings,self.language = settings,language
        self.tokenizer,self.model = None,None

    def __call__(self,question,context,temperature):
        s=self.settings
        if not context.strip():
            return {"prediction":ABSTENTIONS[self.language],"generation_skipped":True,
                    "generation_status":"abstention","input_tokens":0,"output_tokens":0}
        body=prompt(question,context,self.language)
        if s["backend"] == "transformers":
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
            if self.model is None:
                self.tokenizer=AutoTokenizer.from_pretrained(s["model"],revision=s.get("revision"))
                self.model=AutoModelForCausalLM.from_pretrained(s["model"],revision=s.get("revision"),
                    torch_dtype="auto").to(s.get("device","cpu")).eval()
            ids=self.tokenizer.apply_chat_template([{"role":"user","content":body}],
                add_generation_prompt=True,return_tensors="pt").to(self.model.device)
            if ids.shape[1]+s["max_new_tokens"]>s["context_tokens"]:
                raise ValueError("Input exceeds explicit context budget; no truncation applied")
            set_seed(s["seed"])
            kwargs={"do_sample":temperature>0}
            if temperature>0: kwargs["temperature"]=temperature
            with torch.inference_mode():
                result=self.model.generate(ids,max_new_tokens=s["max_new_tokens"],
                    pad_token_id=self.tokenizer.eos_token_id,**kwargs)[0,ids.shape[1]:]
            ended=bool(len(result) and int(result[-1]) in self.model.generation_config.eos_token_id) if isinstance(self.model.generation_config.eos_token_id,list) else bool(len(result) and int(result[-1])==self.model.generation_config.eos_token_id)
            return {"prediction":self.tokenizer.decode(result,skip_special_tokens=True).strip(),
                    "generation_skipped":False,"generation_status":"length_limit" if len(result)>=s["max_new_tokens"] else ("stop" if ended else "unrecorded"),
                    "input_tokens":ids.shape[1],"output_tokens":len(result),
                    "generation_revision":getattr(self.model.config,"_commit_hash",None)}
        if s["backend"] != "ollama":
            raise ValueError("Unknown generation backend")
        import ollama
        from transformers import AutoTokenizer
        if self.tokenizer is None:
            self.tokenizer=AutoTokenizer.from_pretrained(location(s["tokenizer_model"]), revision=s.get("tokenizer_revision"))
        rendered=self.tokenizer.apply_chat_template([{"role":"user","content":body}],
            add_generation_prompt=True, tokenize=False)
        # This Ollama GGUF backend adds BOS automatically. Do not send a second BOS.
        added_bos=0
        if s.get('server_adds_bos',False):
            bos=self.tokenizer.bos_token
            if not bos or not rendered.startswith(bos):
                raise ValueError('Expected a BOS-prefixed chat template')
            rendered=rendered[len(bos):]
            added_bos=1
        expected=len(self.tokenizer.encode(rendered,add_special_tokens=False))+added_bos
        if expected+s["max_new_tokens"]>s["context_tokens"]:
            raise ValueError("Input exceeds explicit context budget; no truncation applied")
        result=ollama.Client(timeout=600).generate(model=s["model"],prompt=rendered,raw=True,
            keep_alive=0,options={"num_ctx":s["context_tokens"],
                "num_predict":s["max_new_tokens"],"seed":s["seed"],"temperature":temperature})
        used=result.get("prompt_eval_count",0)
        if used != expected:
            raise ValueError(f"Backend/tokenizer mismatch or context truncation: expected {expected}, got {used}; record rejected")
        if used+s["max_new_tokens"]>s["context_tokens"]:
            raise ValueError("Insufficient context budget; record rejected")
        return {"prediction":result["response"].strip(),"generation_skipped":False,
            "generation_status":result.get("done_reason","unrecorded"),
            "input_tokens":used,"output_tokens":result.get("eval_count"),
            "generation_revision":s.get("model_digest")}

    def close(self):
        self.model,self.tokenizer=None,None
        gc.collect()
