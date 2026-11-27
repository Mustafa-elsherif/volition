"""Volition decoding API.

Loads the serialized decoder once at startup and serves validated endpoints:
inference, model metadata, sample trials, and the grounded assistant. The artifact
directory is set with the VOLITION_ARTIFACTS environment variable and defaults to
the repository's ``artifacts`` folder.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.assistant import Assistant
from app.inference import Decoder
from app.schemas import (
    AssistantRequest,
    AssistantResponse,
    DecodeRequest,
    DecodeResponse,
    ModelInfo,
)

ARTIFACT_DIR = Path(os.environ.get("VOLITION_ARTIFACTS",
                                   Path(__file__).resolve().parents[2] / "artifacts"))
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if (ARTIFACT_DIR / "decoder.onnx").exists():
        decoder = Decoder(ARTIFACT_DIR)
        STATE["decoder"] = decoder
        STATE["assistant"] = Assistant(metrics=decoder.metrics)
    else:
        STATE["decoder"] = None
        STATE["assistant"] = Assistant()
    yield
    STATE.clear()


app = FastAPI(
    title="Volition Decoding API",
    version="1.0.0",
    summary="Decode a mouse's pre-movement choice from neural population activity.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("VOLITION_CORS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _decoder() -> Decoder:
    decoder = STATE.get("decoder")
    if decoder is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")
    return decoder


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": STATE.get("decoder") is not None}


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    d = _decoder()
    return ModelInfo(
        name="Volition neuron-set transformer",
        version="1.0.0",
        task="Binary pre-movement choice decoding (IBL Brain-Wide Map).",
        n_regions=len(d.vocab),
        max_neurons=d.max_neurons,
        metrics=d.metrics,
        known_regions=d.vocab,
    )


@app.post("/decode", response_model=DecodeResponse)
def decode(req: DecodeRequest) -> DecodeResponse:
    d = _decoder()
    try:
        result = d.decode([n.rate for n in req.neurons], [n.region for n in req.neurons])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return DecodeResponse(**result)


@app.get("/samples")
def samples() -> list[dict]:
    path = ARTIFACT_DIR / "samples.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/assistant", response_model=AssistantResponse)
def assistant(req: AssistantRequest) -> AssistantResponse:
    engine: Assistant = STATE["assistant"]
    out = engine.answer(req.message, [t.model_dump() for t in req.history])
    return AssistantResponse(**out)
