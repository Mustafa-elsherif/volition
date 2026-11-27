"""Grounded knowledge assistant for Volition.

The assistant answers only from the project knowledge base. When a backend Gemini
key is configured it uses Gemini to reply in the user's own language and to phrase
the grounded facts naturally; otherwise it falls back to an extractive answer built
from the best-matching knowledge entries. Either way the content is grounded in the
same knowledge base, there is no user-facing API key, and off-topic questions are
declined.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = Path(__file__).with_name("knowledge_base.json")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

FOLLOWUPS = {
    "overview": "What task does Volition solve?",
    "task": "What data is the model trained on?",
    "dataset": "How does the model work?",
    "model": "Does self-supervised pretraining help?",
    "pretraining": "How well does the decoder perform?",
    "performance": "Does it work on a new mouse?",
    "crosssession": "How robust is it to missing neurons?",
    "robustness": "How does it explain a prediction?",
    "explanation": "How do I use the API?",
    "api": "How do I use the dashboard?",
    "dashboard": "What are the limits of the model?",
    "limits": "Is my data kept private?",
    "privacy": "What can this assistant help with?",
    "scope_guard": "What is Volition?",
}

SYSTEM_PROMPT = (
    "You are the assistant for Volition, a neural choice decoder built on the IBL "
    "Brain-Wide Map. Answer only using the CONTEXT provided. If the context does not "
    "cover the question, say you can only help with Volition and its decoding work. "
    "Detect the language of the user's question and reply in that same language. Be "
    "concise, plain, and professional. Do not invent numbers. Do not use em dashes. "
    "Return strict JSON: {\"answer\": string, \"followups\": [three short questions in "
    "the user's language]}."
)


class Assistant:
    def __init__(self, metrics: dict | None = None):
        self.docs = json.loads(KB_PATH.read_text(encoding="utf-8"))
        self.metrics = metrics or {}
        corpus = [f"{d['title']} {' '.join(d['tags'])} {d['text']}" for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def _retrieve(self, query: str, k: int = 3):
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(sims)[::-1][:k]
        return [(self.docs[i], float(sims[i])) for i in order]

    def _metrics_line(self) -> str:
        auc = self.metrics.get("headline_auc", {})
        if "mean" in auc:
            return (f" Current held-out ROC AUC is about {auc['mean']:.2f} "
                    f"(plus or minus {auc.get('std', 0):.2f}).")
        return ""

    def answer(self, query: str, history: list[dict] | None = None) -> dict:
        query = (query or "").strip()
        if not query:
            return {"answer": "Ask me anything about Volition and how it decodes choice.",
                    "followups": ["What is Volition?", "How does the model work?",
                                  "How well does it perform?"], "grounded": True, "source": "empty"}

        hits = self._retrieve(query, k=3)
        top_doc, top_sim = hits[0]
        greet = query.lower() in {"hi", "hello", "hey", "salam", "bonjour", "hola", "مرحبا"}
        if top_sim < 0.06 and not greet:
            return {
                "answer": "I can only help with Volition: the decoding task, the dataset, the "
                          "model, its performance, robustness, how it explains predictions, and "
                          "how to use the API or dashboard. Ask me about any of those.",
                "followups": ["What is Volition?", "What data does it use?",
                              "How well does it perform?"],
                "grounded": True, "source": "guard",
            }

        context = "\n\n".join(f"[{d['title']}] {d['text']}" for d, _ in hits)
        if top_doc["id"] in ("performance", "crosssession") or "accuracy" in query.lower():
            context += self._metrics_line()

        key = os.environ.get("GEMINI_API_KEY")
        if key:
            out = self._gemini(query, context, history, key)
            if out:
                out["grounded"] = True
                out["source"] = "gemini"
                return out
        return self._fallback(query, hits, greet)

    def _fallback(self, query, hits, greet) -> dict:
        if greet:
            answer = "Hello. I am the Volition assistant. I can explain the decoder, the data, " \
                     "the results, and how to use the demo or API."
        else:
            top_doc = hits[0][0]
            answer = top_doc["text"]
            if top_doc["id"] in ("performance", "crosssession"):
                answer += self._metrics_line()
        used = {h[0]["id"] for h in hits}
        followups = [FOLLOWUPS[h[0]["id"]] for h in hits if h[0]["id"] in FOLLOWUPS][:3]
        if len(followups) < 3:
            followups += ["What is Volition?", "What are the limits of the model?"]
        return {"answer": answer, "followups": followups[:3], "grounded": True, "source": "local"}

    def _gemini(self, query, context, history, key) -> dict | None:
        import requests

        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        turns = []
        for h in (history or [])[-4:]:
            role = "user" if h.get("role") == "user" else "model"
            turns.append({"role": role, "parts": [{"text": h.get("content", "")}]})
        turns.append({"role": "user", "parts": [{"text": f"CONTEXT:\n{context}\n\nQUESTION: {query}"}]})
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": turns,
            "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"},
        }
        try:
            r = requests.post(GEMINI_URL.format(model=model), params={"key": key},
                              json=body, timeout=20)
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            fu = parsed.get("followups", [])[:3]
            return {"answer": str(parsed["answer"]).strip(), "followups": [str(x) for x in fu]}
        except Exception:
            return None
