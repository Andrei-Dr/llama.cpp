// expert residency: tiers, hit rate and misses/s, resident and hit rate per layer (worst layer marked), the placement plane

import { $, isNum, txt, num, pct, bytes, width, show, sized } from "../util.js";
import { store, latest, lastNum } from "../store.js";

// bars on an absolute 0..1 axis (shares and rates), the worst layer marked
function layerBars(id, values, il, fmt, worstIsLow = true) {
  const bars = sized($(id), values.length);
  const vmax = 1;
  let worst = -1;
  values.forEach((v, i) => {
    if (isNum(v) && (worst < 0 || (worstIsLow ? v < values[worst] : v > values[worst]))) worst = i;
  });
  values.forEach((v, i) => {
    const b = bars[i];
    b.style.height = (isNum(v) ? Math.max(2, (100 * v) / vmax) : 2).toFixed(1) + "%";
    b.className = i === worst ? "worst" : "";
    b.title = "L" + il[i] + ": " + fmt(v);
  });
  return worst;
}

function drawPlane(m) {
  const cv = $("x-plane"), L = m.n_layer, E = m.n_expert;
  if (cv.width !== E || cv.height !== L) { cv.width = E; cv.height = L; }
  const ctx = cv.getContext("2d"), img = ctx.createImageData(E, L), px = img.data;
  for (let i = 0; i < px.length; i += 4) { px[i] = 17; px[i + 1] = 17; px[i + 2] = 19; px[i + 3] = 255; }
  let res = 0, inFlight = 0;
  m.slot_expert.forEach((slots, l) => slots.forEach((e) => {
    if (e === -2) { inFlight++; return; }
    if (e < 0 || e >= E) return;
    const o = 4 * (l * E + e);
    px[o] = 245; px[o + 1] = 245; px[o + 2] = 245;
    res++;
  }));
  ctx.putImageData(img, 0, 0);
  txt("x-plane-meta", num(res) + " of " + num(L * E) + " in VRAM · " + inFlight + " uploading");
}

export function render(st) {
  const m = st.moe_cache;
  show("p-experts", !!m);
  if (!m) return;
  const d = latest(), L = m.n_layer, E = m.n_expert, total = L * E;
  const resident = m.layers.resident.reduce((a, b) => a + b, 0), perExpert = L ? m.slot_bytes / L : 0;
  const mode = m.suspended ? (m.pressure ? "released: memory pressure" : "released: prefill mode")
    : m.n_live < m.n_slots ? "partial " + m.n_live + "/" + m.n_slots : "decode";
  txt("x-meta", L + " layers × " + E + " experts · " + m.n_live + " slots/layer · " + mode + (m.sync_publish ? " · SYNC" : ""));

  width("x-tier-v", resident / total);
  width("x-tier-p", 1 - resident / total);
  txt("x-vram", num(resident) + " · " + bytes(resident * perExpert) + " · " + pct(resident / total));
  txt("x-pinned", num(total - resident) + " · " + bytes((total - resident) * perExpert) + " · " + pct(1 - resident / total));

  const c = m.counters, cum = c.hit + c.miss ? c.hit / (c.hit + c.miss) : null;
  const hr = isNum(d.moe_hit_rate) ? d.moe_hit_rate : lastNum("moe_hit_rate");
  txt("x-hit", pct(hr) + " · " + num(d.moe_miss_s, 0) + " missed/s · cumulative " + pct(cum));
  width("x-hit-bar", hr);

  const il = m.layers.il, share = m.layers.resident.map((r) => (E ? r / E : null));
  const wr = layerBars("x-res", share, il, (v) => pct(v));
  const mean = share.reduce((a, b) => a + b, 0) / Math.max(1, share.length);
  txt("x-res-meta", pct(mean) + " mean" + (wr >= 0 ? " · worst L" + il[wr] + " " + pct(share[wr]) : ""));

  let rates = store.layers.rates();
  if (!rates.some(isNum)) {
    rates = m.layers.hit.map((h, i) => (h + m.layers.miss[i] ? h / (h + m.layers.miss[i]) : null));
    txt("x-layer-win", "since start");
  } else {
    txt("x-layer-win", "last " + store.layers.windowS + " s");
  }
  const wl = layerBars("x-layers", rates, il, (v) => pct(v));
  const vals = rates.filter(isNum), rm = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  txt("x-layer-meta", pct(rm) + " mean" + (wl >= 0 ? " · worst L" + il[wl] + " " + pct(rates[wl]) : ""));
  txt("x-l0", "L" + il[0]);
  txt("x-ln", "L" + il[L - 1]);
  drawPlane(m);
}
