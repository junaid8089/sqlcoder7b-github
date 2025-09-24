import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="SQLCoder 7B API")

# Define request model
class Query(BaseModel):
    prompt: str

# Read Hugging Face token from environment
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in environment. Set it in Codespaces or GitHub Secrets.")

# Root endpoint
@app.get("/")
def root():
    return {"message": "SQLCoder 7B API running"}

# Generate SQL endpoint
@app.post("/generate")
def generate_sql(query: Query):
    prompt = query.prompt
    try:
        from huggingface_hub import InferenceApi
        infer = InferenceApi(repo_id="defog/sqlcoder-7b-2", token=hf_token)
        response = infer(prompt, {"max_new_tokens": 128})
        return {"result": response}
    except Exception as e:
        # Catch HF API errors and return clean JSON
        raise HTTPException(status_code=500, detail=str(e))
