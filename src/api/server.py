"""FastAPI server for Tensor Network Language Model inference.

Usage:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000

    # Or with auto-reload for development:
    uvicorn src.api.server:app --reload --port 8000

Endpoints:
    POST /generate     - Generate text from a prompt
    POST /perplexity   - Compute perplexity of text
    GET  /health       - Health check
    GET  /model/info   - Model information
"""

import os
import time
from contextlib import asynccontextmanager

import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import GPT2TokenizerFast

from src.models.gpt import build_gpt_model
from src.inference.generator import TextGenerator
from src.evaluation.metrics import count_parameters


# --- Request/Response schemas ---

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096, description="Input text prompt")
    max_new_tokens: int = Field(128, ge=1, le=1024, description="Max tokens to generate")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=0, le=500, description="Top-k filtering (0=disabled)")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")


class GenerateResponse(BaseModel):
    text: str
    prompt: str
    tokens_generated: int
    generation_time_ms: float
    tokens_per_second: float


class PerplexityRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8192, description="Text to evaluate")


class PerplexityResponse(BaseModel):
    perplexity: float
    text_length: int
    computation_time_ms: float


class ModelInfoResponse(BaseModel):
    model_type: str
    total_parameters: int
    parameter_breakdown: dict
    device: str
    max_seq_len: int
    vocab_size: int


# --- Global state ---

generator: TextGenerator = None
model_info: dict = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global generator, model_info

    config_path = os.environ.get("MODEL_CONFIG", "config/gpt_tn_85m.yaml")
    checkpoint_path = os.environ.get("MODEL_CHECKPOINT", None)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from config: {config_path}")
    print(f"Device: {device}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Build model
    model = build_gpt_model(config)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded from step {ckpt.get('step', '?')}")
    else:
        print("No checkpoint loaded - using random weights (demo mode)")

    model = model.to(device)

    # Quantize for faster CPU inference
    if device == "cpu":
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8,
        )
        print("Applied dynamic int8 quantization for CPU")

    # Setup generator
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    generator = TextGenerator(model, tokenizer, device)

    # Model info
    param_info = count_parameters(model)
    model_info = {
        "model_type": config["model"]["type"],
        "total_parameters": param_info["total"],
        "parameter_breakdown": {k: v for k, v in param_info["breakdown"].items()},
        "device": device,
        "max_seq_len": config["model"].get("max_seq_len", 1024),
        "vocab_size": config["model"].get("vocab_size", 50257),
    }

    print(f"Model loaded: {param_info['total']:,} parameters on {device}")
    yield
    print("Shutting down...")


# --- App ---

app = FastAPI(
    title="Tensor Network LM API",
    description="Language model inference API using MPS/Tensor Train architecture with parallel prefix scan",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    if model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_info


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    text = generator.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )
    elapsed_ms = (time.time() - start) * 1000

    prompt_tokens = len(generator.tokenizer.encode(req.prompt))
    total_tokens = len(generator.tokenizer.encode(text))
    new_tokens = total_tokens - prompt_tokens

    return GenerateResponse(
        text=text,
        prompt=req.prompt,
        tokens_generated=new_tokens,
        generation_time_ms=round(elapsed_ms, 1),
        tokens_per_second=round(new_tokens / (elapsed_ms / 1000), 1) if elapsed_ms > 0 else 0,
    )


@app.post("/perplexity", response_model=PerplexityResponse)
async def perplexity(req: PerplexityRequest):
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    ppl = generator.get_perplexity(req.text)
    elapsed_ms = (time.time() - start) * 1000

    return PerplexityResponse(
        perplexity=round(ppl, 2),
        text_length=len(req.text),
        computation_time_ms=round(elapsed_ms, 1),
    )
