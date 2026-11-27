# Volition API

FastAPI service that serves the Volition decoder. It loads the serialized artifact
(`artifacts/decoder.onnx` and `metadata.json`) once at startup and exposes validated
endpoints. Inference uses ONNX Runtime, so the server and the browser demo return
identical probabilities.

## Run

```
pip install -r requirements.txt
VOLITION_ARTIFACTS=../artifacts uvicorn app.main:app --reload
# interactive docs at http://localhost:8000/docs
```

Or with Docker, from the repository root: `docker build -f api/Dockerfile -t volition-api . && docker run -p 8000:8000 volition-api`.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | | service and model status |
| GET | `/model-info` | | model metadata and held-out metrics |
| POST | `/decode` | `{ "neurons": [{ "rate": float, "region": str }, ...] }` | probability, choice, confidence, per-neuron and per-region attributions |
| GET | `/samples` | | bundled sample trials |
| POST | `/assistant` | `{ "message": str, "history": [...] }` | grounded answer and follow-up questions |

`rate` is the z-scored pre-movement firing rate; `region` is a Beryl acronym. The
server does not re-normalize, because per-neuron normalization needs a whole session
and cannot be done from a single trial. Unknown regions are treated as padding.

## Tests

```
python -m pytest
```

The tests build a small dummy artifact and check the service contract: inference
shapes and ranges, attribution ordering, input validation, and that the assistant
stays grounded and declines off-topic questions.
