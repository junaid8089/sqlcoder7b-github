from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(title="SQLCoder 7B API")

class Query(BaseModel):
    prompt: str

@app.post("/generate")
def generate_sql(query: Query):
    prompt = query.prompt
    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        from huggingface_hub import InferenceApi
        infer = InferenceApi(repo_id="defog/sqlcoder-7b-2", token=hf_token)
        resp = infer(prompt, {"max_new_tokens": 128})
        return {"result": resp}

    # Fallback demo with GPT-2 (no HF_TOKEN)
    from transformers import pipeline
    gen = pipeline("text-generation", model="gpt2")
    out = gen(prompt, max_new_tokens=64)
    return {"result": out[0]["generated_text"]}

@app.get("/")
def root():
    return {"message": "SQLCoder 7B API running"}
