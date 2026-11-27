// In-browser inference with ONNX Runtime Web. Mirrors the FastAPI backend: pad the
// neuron set to a fixed length, run the graph, read back the choice probability and
// the per-neuron pooling attention. The model runs entirely on the visitor's device.

let session = null;
let meta = null;

export async function loadModel(assetBase = "assets") {
  const ort = window.ort;
  ort.env.wasm.wasmPaths = new URL("vendor/ort/", window.location.href).href;
  ort.env.wasm.numThreads = 1; // GitHub Pages sends no COOP/COEP headers for threads
  meta = await fetch(`${assetBase}/metadata.json`).then((r) => r.json());
  session = await ort.InferenceSession.create(`${assetBase}/decoder.onnx`, {
    executionProviders: ["wasm"],
  });
  return meta;
}

export function metadata() {
  return meta;
}

export function regionId(region) {
  if (!meta) return 0;
  return meta.region_to_id[region] ?? 0;
}

// rates: number[], regionIds: number[] (same length). Returns { probability, attention }.
export async function decode(rates, regionIds) {
  const ort = window.ort;
  const N = meta.max_neurons;
  const n = rates.length;
  const R = new Float32Array(N);
  const I = new BigInt64Array(N);
  const M = new Uint8Array(N).fill(1); // 1 = padding
  for (let i = 0; i < n; i++) {
    R[i] = rates[i];
    I[i] = BigInt(regionIds[i] | 0);
    M[i] = 0;
  }
  const feeds = {
    rates: new ort.Tensor("float32", R, [1, N]),
    region_ids: new ort.Tensor("int64", I, [1, N]),
    pad_mask: new ort.Tensor("bool", M, [1, N]),
  };
  const out = await session.run(feeds);
  const probability = out.probability.data[0];
  const attention = Array.from(out.attention.data.slice(0, n));
  return { probability, attention };
}

// Build an interpretable summary from a decode result and the trial's regions.
export function explain(rates, regions, attention, topK = 12) {
  const idx = attention.map((a, i) => i).sort((a, b) => attention[b] - attention[a]);
  const topNeurons = idx.slice(0, topK).map((i) => ({
    region: regions[i],
    rate: rates[i],
    attention: attention[i],
  }));
  const byRegion = {};
  regions.forEach((r, i) => {
    byRegion[r] = byRegion[r] || { region: r, attention: 0, n: 0 };
    byRegion[r].attention += attention[i];
    byRegion[r].n += 1;
  });
  const regionContrib = Object.values(byRegion).sort((a, b) => b.attention - a.attention);
  return { topNeurons, regionContrib };
}
