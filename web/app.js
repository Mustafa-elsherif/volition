import { loadModel, decode, explain, metadata } from "./inference.js";
import { loadAssistant, ask } from "./assistant.js";

const ASSETS = "assets";
const $ = (s) => document.querySelector(s);
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let samples = [];
let metrics = null;
let current = null; // current trial

/* ---------------- theme ---------------- */
$("#theme-toggle").addEventListener("click", () => {
  const now = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", now);
  try { localStorage.setItem("volition-theme", now); } catch (e) {}
  if (current) drawRaster(current);
});

/* ---------------- smooth scroll + CTA ---------------- */
document.querySelectorAll("[data-scroll]").forEach((b) =>
  b.addEventListener("click", () => {
    const t = $(b.getAttribute("data-scroll"));
    t.scrollIntoView({ behavior: reduced ? "auto" : "smooth" });
  })
);

/* ---------------- raster ---------------- */
function mulberry(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }

let rasterAnim = null;
function drawRaster(trial, attn) {
  const canvas = $("#raster");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const amber = cssVar("--amber"), azure = cssVar("--azure"), line = cssVar("--line-strong");

  const rates = trial.rates;
  const nCols = Math.min(rates.length, 150);
  const step = rates.length / nCols;
  const cols = [];
  for (let c = 0; c < nCols; c++) {
    const i = Math.floor(c * step);
    const v = 1 / (1 + Math.exp(-rates[i])); // squash z-score to 0..1
    cols.push({ i, v });
  }
  const maxAttn = attn ? Math.max(...attn) : 0;
  const pad = 14, gw = w - pad * 2, gh = h - pad * 2;
  const colW = gw / nCols;
  const rows = 9;
  const rng = mulberry(1337);
  const tickFor = cols.map(() => rng()); // stable jitter per column

  if (rasterAnim) cancelAnimationFrame(rasterAnim);
  const start = performance.now();
  const dur = reduced ? 0 : 900;

  function frame(now) {
    const p = dur === 0 ? 1 : Math.min(1, (now - start) / dur);
    ctx.clearRect(0, 0, w, h);
    // baseline grid line
    ctx.strokeStyle = line; ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, h - pad); ctx.lineTo(w - pad, h - pad); ctx.stroke();
    ctx.globalAlpha = 1;
    const reveal = p * nCols;
    for (let c = 0; c < nCols; c++) {
      if (c > reveal) break;
      const col = cols[c];
      const x = pad + c * colW + colW / 2;
      const nTicks = 1 + Math.round(col.v * (rows - 1));
      let strong = false, a = 0.22 + col.v * 0.55;
      if (attn) {
        const w2 = attn[col.i] / (maxAttn || 1);
        strong = w2 > 0.45;
        a = 0.14 + w2 * 0.85;
      }
      ctx.strokeStyle = strong || !attn ? amber : azure;
      ctx.globalAlpha = a * (c > reveal - 1 ? reveal - c : 1);
      ctx.lineWidth = Math.max(1.4, colW * 0.42);
      for (let t = 0; t < nTicks; t++) {
        const jitter = (tickFor[c] + t * 0.13) % 1;
        const y = pad + gh - (t / rows) * gh - jitter * (gh / rows) * 0.6;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y - 5); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    if (p < 1) rasterAnim = requestAnimationFrame(frame);
  }
  rasterAnim = requestAnimationFrame(frame);
}

/* ---------------- choice helpers ---------------- */
function choiceLabel(p) { return p >= 0.5 ? "Right" : "Left"; }
function setNeedle(el, p) { el.style.left = `${(p * 100).toFixed(1)}%`; }

/* ---------------- hero ---------------- */
async function runHero(trial) {
  $("#hero-session").textContent = `session ${trial.session_id.slice(0, 8)}`;
  drawRaster(trial);
  const { probability } = await decode(trial.rates, trial.region_ids);
  setNeedle($("#hero-needle"), probability);
  $("#hero-prob").textContent = `${(probability * 100).toFixed(0)}%`;
}

/* ---------------- workbench ---------------- */
function trialFacts(t) {
  const c = t.contrast;
  const side = c > 0 ? "left" : c < 0 ? "right" : "zero";
  const strength = Math.abs(c) === 0 ? "0" : Math.abs(c).toFixed(2);
  $("#trial-facts").innerHTML = `
    <span><b>neurons</b> ${t.rates.length}</span>
    <span><b>stimulus</b> ${side} contrast ${strength}</span>
    <span><b>prior P(left)</b> ${t.prob_left.toFixed(1)}</span>
    <span><b>recorded</b> hidden until decode</span>`;
}

function selectTrial(idx) {
  current = samples[idx];
  current._attn = null;
  trialFacts(current);
  $("#result-body").hidden = true;
  $("#result-empty").hidden = false;
  drawRaster(current);
}

async function runDecode() {
  const t = current;
  const btn = $("#decode-btn");
  btn.disabled = true; btn.textContent = "decoding…";
  const { probability, attention } = await decode(t.rates, t.region_ids);
  const { topNeurons, regionContrib } = explain(t.rates, t.regions, attention, 12);

  $("#result-empty").hidden = true;
  $("#result-body").hidden = false;
  const pred = choiceLabel(probability);
  $("#r-choice").textContent = `${pred} (choice ${probability >= 0.5 ? "+1" : "-1"})`;
  $("#r-prob").textContent = probability.toFixed(3);
  $("#r-conf").textContent = `${(Math.abs(probability - 0.5) * 200).toFixed(0)}%`;
  setNeedle($("#r-needle"), probability);

  const truth = t.y_true === 1 ? "Right" : "Left";
  const correct = pred === truth;
  const v = $("#r-verdict");
  v.textContent = correct ? `correct · recorded ${truth}` : `wrong · recorded ${truth}`;
  v.className = `verdict ${correct ? "correct" : "wrong"}`;

  const maxR = Math.max(...regionContrib.map((r) => r.attention), 0.0001);
  $("#region-bars").innerHTML = regionContrib.slice(0, 8).map((r) => `
    <div class="region-row">
      <span class="rname">${r.region}</span>
      <div class="region-bar-track"><div class="region-bar-fill" style="width:${(r.attention / maxR * 100).toFixed(1)}%"></div></div>
      <span class="rval">${(r.attention * 100).toFixed(1)}%</span>
    </div>`).join("");

  $("#neuron-list").innerHTML = topNeurons.map((n) => `
    <div class="nrow"><span>${n.region}</span><span>${n.attention.toFixed(3)}</span></div>`).join("");

  btn.disabled = false; btn.textContent = "Run decode";
}

/* ---------------- metrics + charts ---------------- */
function fmt(m) { return m && m.mean != null ? m.mean.toFixed(2) : "—"; }
function pm(m) { return m && m.std != null ? `±${m.std.toFixed(2)}` : ""; }

function renderMetrics() {
  if (!metrics) return;
  const cards = [
    { name: "Shared model · ROC AUC", v: fmt(metrics.shared_auc), pm: pm(metrics.shared_auc), sub: "one model, held-out trials, 10 sessions", lead: true },
    { name: "Shared model · balanced acc", v: fmt(metrics.shared_bacc), pm: pm(metrics.shared_bacc), sub: "chance is 0.50" },
    { name: "Calibration · ECE", v: fmt(metrics.shared_ece), pm: pm(metrics.shared_ece), sub: "lower is better, 0 is perfect" },
    { name: "Per-session logistic · AUC", v: fmt(metrics.logreg_auc), pm: pm(metrics.logreg_auc), sub: "retrained per animal (reference)" },
    { name: "Per-session forest · AUC", v: fmt(metrics.rf_auc), pm: pm(metrics.rf_auc), sub: "retrained per animal (reference)" },
    { name: "New animal · AUC", v: fmt(metrics.cross_auc), pm: pm(metrics.cross_auc), sub: "leave-one-session-out, near chance" },
  ];
  $("#metrics-grid").innerHTML = cards.map((c) => `
    <div class="metric ${c.lead ? "lead" : ""}">
      <span class="m-name">${c.name}</span>
      <span class="m-val">${c.v}<span class="pm"> ${c.pm}</span></span>
      <span class="m-sub">${c.sub}</span>
    </div>`).join("");

  if (metrics.shared_auc && metrics.cross_auc) {
    $("#finding-cross").textContent =
      `Predicting a mouse the model never saw stays near chance (AUC ${fmt(metrics.cross_auc)}) for the transformer and the linear baseline alike. Pre-movement choice signals here do not transfer across animals through shared regions.`;
  }
  if (metrics.robustness) drawRobust(metrics.robustness);
}

function drawRobust(points) {
  const el = $("#robust-chart");
  const W = el.clientWidth || 600, H = 240, m = { l: 44, r: 16, t: 16, b: 40 };
  const xs = points.map((p) => p.level);
  const ys = points.map((p) => p.balanced_accuracy);
  const x = (v) => m.l + (v / 0.9) * (W - m.l - m.r);
  const y = (v) => m.t + (1 - (v - 0.5) / 0.4) * (H - m.t - m.b); // 0.5..0.9 range
  const amber = cssVar("--amber"), line = cssVar("--line"), muted = cssVar("--muted");
  let path = points.map((p, i) => `${i ? "L" : "M"}${x(p.level).toFixed(1)} ${y(p.balanced_accuracy).toFixed(1)}`).join(" ");
  const grid = [0.5, 0.6, 0.7, 0.8].map((g) =>
    `<line x1="${m.l}" y1="${y(g)}" x2="${W - m.r}" y2="${y(g)}" stroke="${line}"/>
     <text x="${m.l - 8}" y="${y(g) + 4}" text-anchor="end" fill="${muted}" font-size="11" font-family="JetBrains Mono">${g.toFixed(1)}</text>`
  ).join("");
  const dots = points.map((p) => `<circle cx="${x(p.level)}" cy="${y(p.balanced_accuracy)}" r="4" fill="${amber}"/>`).join("");
  const xlabels = points.map((p) => `<text x="${x(p.level)}" y="${H - 14}" text-anchor="middle" fill="${muted}" font-size="11" font-family="JetBrains Mono">${Math.round(p.level * 100)}%</text>`).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img">
    ${grid}
    <path d="${path}" fill="none" stroke="${amber}" stroke-width="2.5" stroke-linejoin="round"/>
    ${dots}${xlabels}
    <text x="${(W) / 2}" y="${H}" text-anchor="middle" fill="${muted}" font-size="11" font-family="JetBrains Mono">neurons removed</text>
  </svg>`;
}

/* ---------------- assistant ---------------- */
const chatHistory = [];
function addMsg(text, who) {
  const d = document.createElement("div");
  d.className = `msg ${who}`;
  d.textContent = text;
  $("#chat-log").appendChild(d);
  d.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "nearest" });
  return d;
}
function setFollowups(list) {
  $("#chat-followups").innerHTML = "";
  list.forEach((q) => {
    const b = document.createElement("button");
    b.className = "chip"; b.type = "button"; b.textContent = q;
    b.addEventListener("click", () => sendChat(q));
    $("#chat-followups").appendChild(b);
  });
}
async function sendChat(message) {
  if (!message.trim()) return;
  addMsg(message, "user");
  chatHistory.push({ role: "user", content: message });
  $("#chat-input").value = "";
  $("#chat-followups").innerHTML = "";
  const thinking = addMsg("…", "bot"); thinking.classList.add("thinking");
  const res = await ask(message, chatHistory, metrics && metrics._summary);
  thinking.remove();
  addMsg(res.answer, "bot");
  chatHistory.push({ role: "assistant", content: res.answer });
  setFollowups(res.followups || []);
}
$("#chat-form").addEventListener("submit", (e) => { e.preventDefault(); sendChat($("#chat-input").value); });

/* ---------------- api sample ---------------- */
function renderApiSample() {
  const code = `<span class="c"># decode one trial</span>
curl -X POST <span class="s">"$API/decode"</span> \\
  -H <span class="s">"content-type: application/json"</span> \\
  -d <span class="s">'{
    "neurons": [
      {"rate": 1.8, "region": "VISp"},
      {"rate": -0.4, "region": "CP"},
      {"rate": 0.9, "region": "MOs"}
    ]
  }'</span>

<span class="c"># -> { "probability": 0.73, "choice": "Right (choice = +1)",</span>
<span class="c">#      "confidence": 0.46, "region_contributions": [...] }</span>`;
  $("#code-sample").innerHTML = code;
}

/* ---------------- boot ---------------- */
async function boot() {
  renderApiSample();
  try {
    await loadAssistant(ASSETS);
  } catch (e) { /* assistant optional */ }
  setFollowups(["What is Volition?", "How does the model work?", "How well does it perform?"]);
  addMsg("Hello. Ask me about the decoder, the data, the results, or how to use the demo.", "bot");

  try {
    metrics = await fetch(`${ASSETS}/metrics.json`).then((r) => r.json());
    renderMetrics();
  } catch (e) { /* metrics optional before sweep wiring */ }

  // Samples load independently of the model, so the hero and workbench render even
  // if the inference runtime is slow or blocked.
  try {
    samples = await fetch(`${ASSETS}/samples.json`).then((r) => r.json());
    const sel = $("#trial-select");
    sel.innerHTML = samples.map((s, i) =>
      `<option value="${i}">trial ${i + 1} · session ${s.session_id.slice(0, 8)} · ${s.rates.length} neurons</option>`).join("");
    sel.addEventListener("change", () => selectTrial(+sel.value));
    $("#random-btn").addEventListener("click", () => {
      const i = Math.floor(Math.random() * samples.length);
      sel.value = i; selectTrial(i);
    });
    selectTrial(0);
    $("#hero-session").textContent = `session ${samples[0].session_id.slice(0, 8)}`;
    drawRaster(samples[0]); // data-driven, no model needed
  } catch (e) {
    $("#hero-status").textContent = "sample data not found";
    console.error(e);
    return;
  }

  // The model enables live decoding and attention coloring.
  try {
    await loadModel(ASSETS);
    $("#decode-btn").addEventListener("click", runDecode);
    $("#decode-btn").disabled = false;
    $("#hero-status").textContent = "model ready · runs in your browser";
    await runHero(samples[0]);
    if (new URLSearchParams(location.search).has("demo")) await runDecode();
  } catch (e) {
    $("#hero-status").textContent = "decoder unavailable in this browser; raster shown from data";
    console.error(e);
  }
}
boot();
window.addEventListener("resize", () => { if (current) drawRaster(current); });
