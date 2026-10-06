import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference.engine import InferenceEngine

from fastapi.responses import FileResponse

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "configs/run_01.yaml"))
CHECKPOINT_PATH = Path(os.environ.get("CHECKPOINT_PATH", "checkpoints/best.pt"))
TOKENIZER_PATH = Path(os.environ.get("TOKENIZER_PATH", "tokenizer/tokenizer.json"))
BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="Our LLM", description="A small pretrained language model.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None

@app.on_event("startup")
def load_engine():
    global engine
    engine = InferenceEngine(CONFIG_PATH, CHECKPOINT_PATH, TOKENIZER_PATH)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    max_tokens: int = Field(50, ge=1, le=200)
    temperature: float = Field(0.8, gt=0.0, le=2.0)
    top_k: int = Field(50, ge=0, le=8000)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    latency_ms: float


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if engine is None:
        raise HTTPException(503, "model still loading, try again shortly")

    text, tokens_generated, latency_ms = engine.generate(
        req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )
    return GenerateResponse(
        text=text, tokens_generated=tokens_generated, latency_ms=latency_ms
    )


@app.get("/health")
def health():
    return {
        "status": "ok" if engine is not None else "loading",
        "model": "our-llm",
        "device": str(engine.device) if engine is not None else None,
    }


@app.get("/")
def root():
    return FileResponse(BASE_DIR / "ui" / "index.html")