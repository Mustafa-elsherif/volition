# Deployment

Two deliverables deploy independently: the static dashboard (GitHub Pages, in-browser
inference, no server) and the FastAPI backend (Docker).

## Frontend: GitHub Pages

The dashboard in `web/` is fully static and runs the model in the browser with ONNX
Runtime Web (self-hosted under `web/vendor/ort/`, no CDN). The workflow at
`.github/workflows/pages.yml` publishes `web/` on every push to `main` that touches it.

One-time setup:

1. The repository must be public, or on a plan that allows Pages for private repositories.
2. In the repository settings, under **Pages**, set the source to **GitHub Actions**.
3. Push to `main`, or run the **Deploy site to GitHub Pages** workflow manually.

The site is then served at `https://<owner>.github.io/<repo>/`
(`https://mustafa-elsherif.github.io/volition/`). All asset paths are relative, so the
project subpath works without changes.

To run it locally:

```
cd web
python -m http.server 8099
# open http://localhost:8099
```

## Backend: Docker

The FastAPI service in `api/` loads the serialized artifact at startup and serves the
model as a validated REST API.

```
# from the repository root, so the build can copy artifacts/
docker build -f api/Dockerfile -t volition-api .
docker run -p 8000:8000 volition-api
# open http://localhost:8000/docs
```

Without Docker:

```
cd api
pip install -r requirements.txt
VOLITION_ARTIFACTS=../artifacts uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `VOLITION_ARTIFACTS` | `./artifacts` | Directory with `decoder.onnx`, `metadata.json`, `samples.json` |
| `VOLITION_CORS` | `*` | Comma-separated allowed origins |
| `GEMINI_API_KEY` | unset | Enables the assistant's Gemini backend; without it the assistant uses the local grounded fallback |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |

The Gemini key is read only on the server and is never exposed to the browser. The
dashboard's assistant works without a backend using in-browser retrieval; set
`window.VOLITION_API` in the page to point the assistant at a running backend.

### Hosting the backend

Any host that runs a container works (Fly.io, Render, Railway, a VM with Docker). The
image listens on port 8000 and has a `/health` check. Set `VOLITION_CORS` to the
Pages origin so the dashboard can call it.
