// Grounded assistant. On the static site it retrieves from the project knowledge
// base in the browser; if a backend URL is configured (window.VOLITION_API) it calls
// the backend, which can use Gemini for multilingual answers. Either way answers stay
// grounded in Volition's own knowledge and off-topic questions are declined.

const FOLLOWUPS = {
  overview: "What task does Volition solve?",
  task: "What data is the model trained on?",
  dataset: "How does the model work?",
  model: "Does self-supervised pretraining help?",
  pretraining: "How well does the decoder perform?",
  performance: "Does it work on a new mouse?",
  crosssession: "How robust is it to missing neurons?",
  robustness: "How does it explain a prediction?",
  explanation: "How do I use the API?",
  api: "How do I use the dashboard?",
  dashboard: "What are the limits of the model?",
  limits: "Is my data kept private?",
  privacy: "What can this assistant help with?",
  scope_guard: "What is Volition?",
};

const GREETINGS = new Set(["hi", "hello", "hey", "salam", "bonjour", "hola", "مرحبا", "ciao", "hallo"]);
const STOP = new Set("the a an of to and or is are in on for with how what does do it its this that you your can".split(" "));

let docs = [];
let idf = {};

function tokenize(s) {
  return (s.toLowerCase().match(/[a-z0-9]+/g) || []).filter((t) => !STOP.has(t));
}

export async function loadAssistant(assetBase = "assets") {
  docs = await fetch(`${assetBase}/knowledge_base.json`).then((r) => r.json());
  const df = {};
  docs.forEach((d) => {
    const terms = new Set(tokenize(`${d.title} ${d.tags.join(" ")} ${d.text}`));
    terms.forEach((t) => (df[t] = (df[t] || 0) + 1));
    d._vec = termVec(`${d.title} ${d.title} ${d.tags.join(" ")} ${d.text}`);
  });
  const nDoc = docs.length;
  Object.keys(df).forEach((t) => (idf[t] = Math.log((1 + nDoc) / (1 + df[t])) + 1));
  docs.forEach((d) => (d._w = weighted(d._vec)));
}

function termVec(s) {
  const v = {};
  tokenize(s).forEach((t) => (v[t] = (v[t] || 0) + 1));
  return v;
}
function weighted(vec) {
  const w = {};
  let norm = 0;
  for (const t in vec) {
    const x = vec[t] * (idf[t] || 1);
    w[t] = x;
    norm += x * x;
  }
  norm = Math.sqrt(norm) || 1;
  for (const t in w) w[t] /= norm;
  return w;
}
function cosine(a, b) {
  let s = 0;
  for (const t in a) if (b[t]) s += a[t] * b[t];
  return s;
}

function retrieve(query, k = 3) {
  const qw = weighted(termVec(query));
  return docs
    .map((d) => ({ d, score: cosine(qw, d._w) }))
    .sort((x, y) => y.score - x.score)
    .slice(0, k);
}

export async function ask(message, history, metrics) {
  const api = window.VOLITION_API;
  if (api) {
    try {
      const r = await fetch(`${api.replace(/\/$/, "")}/assistant`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ message, history }),
      });
      if (r.ok) return await r.json();
    } catch (e) {
      /* fall back to local */
    }
  }
  return localAnswer(message, metrics);
}

function metricsLine(metrics) {
  const auc = metrics && metrics.cross_session_auc;
  const pw = metrics && metrics.pooled_within && metrics.pooled_within.set_transformer;
  if (pw && pw.mean) return ` The current held-out ROC AUC is about ${pw.mean.toFixed(2)} (plus or minus ${(pw.std || 0).toFixed(2)}).`;
  return "";
}

function localAnswer(message, metrics) {
  const q = (message || "").trim();
  if (!q) {
    return { answer: "Ask me anything about Volition and how it decodes choice.",
      followups: ["What is Volition?", "How does the model work?", "How well does it perform?"],
      grounded: true, source: "empty" };
  }
  const greet = GREETINGS.has(q.toLowerCase());
  const hits = retrieve(q, 3);
  if (hits[0].score < 0.04 && !greet) {
    return { answer: "I can only help with Volition: the decoding task, the dataset, the model, its performance, robustness, how it explains predictions, and how to use the API or dashboard. Ask me about any of those.",
      followups: ["What is Volition?", "What data does it use?", "How well does it perform?"],
      grounded: true, source: "guard" };
  }
  let answer;
  if (greet) {
    answer = "Hello. I am the Volition assistant. I can explain the decoder, the data, the results, and how to use the demo or API.";
  } else {
    const top = hits[0].d;
    answer = top.text;
    if (top.id === "performance" || top.id === "crosssession") answer += metricsLine(metrics);
  }
  let followups = hits.map((h) => FOLLOWUPS[h.d.id]).filter(Boolean).slice(0, 3);
  while (followups.length < 3) followups.push("What are the limits of the model?");
  return { answer, followups: followups.slice(0, 3), grounded: true, source: "local" };
}
