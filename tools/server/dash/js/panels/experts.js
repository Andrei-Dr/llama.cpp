// expert residency: tiers, hit rate and misses/s, resident and hit rate per layer (worst layer marked), the placement plane

import { $, isNum, txt, num, pct, bytes, width, show, sized, val } from "../util.js";
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

// the placement plane: one cell per (layer, expert), drawn at a capped cell size and left-aligned, so a 6 x 4 model is
// a small grid and a 40 x 256 model fills the width. Colors come from the page's tokens.
const MAX_CELL = 24, MAX_H = 280;
let tone = null;

function colors() {
  if (!tone) {
    const cs = getComputedStyle(document.documentElement), c = (n) => cs.getPropertyValue(n).trim();
    tone = { none: c("--plane-none"), res: c("--plane-res"), up: c("--plane-up") };
  }
  return tone;
}

function drawPlane(m) {
  const cv = $("x-plane"), L = m.n_layer, E = m.n_expert, dpr = window.devicePixelRatio || 1;
  const avail = Math.max(1, Math.floor(cv.parentElement.clientWidth));
  // fractional column width so the plane fills the box when the cells are small, a capped cell when there are few experts
  const cw = Math.min(MAX_CELL, avail / Math.max(1, E)), ch = Math.max(3, Math.min(MAX_CELL, Math.floor(MAX_H / Math.max(1, L))));
  const gap = cw >= 6 && ch >= 6 ? 1 : 0;
  const w = Math.round(cw * E), h = ch * L;
  if (cv.dataset.k !== w + "x" + h + "@" + dpr) {
    cv.dataset.k = w + "x" + h + "@" + dpr;
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    cv.style.width = w + "px";
    cv.style.height = h + "px";
  }
  const ctx = cv.getContext("2d"), t = colors();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = t.none;
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = t.res;
  let res = 0, inFlight = 0;
  m.slot_expert.forEach((slots, l) => slots.forEach((e) => {
    if (e === -2) {
      inFlight++;
      return;
    }
    if (e < 0 || e >= E) return;
    const x0 = Math.round(e * cw), x1 = Math.round((e + 1) * cw);
    ctx.fillRect(x0, l * ch, Math.max(1, x1 - x0 - gap), ch - gap);
    res++;
  }));
  // in-flight uploads: their experts are not known yet, so they are counted, not placed
  txt("x-plane-meta", num(res) + " of " + num(L * E) + " in VRAM" + (inFlight ? " · " + inFlight + " uploading" : ""));
}

// the per-layer rows (residency, hit rate, placement) need a cache that reports each layer; a cache that pools its slots
// over the layers reports totals only, and those rows stay hidden
function renderLayers(m) {
  const L = m.n_layer, E = m.n_expert_layer, il = m.layers.il, share = m.layers.resident.map((r) => (E ? r / E : null));
  const wr = layerBars("x-res", share, il, (v) => pct(v));
  const mean = share.reduce((a, b) => a + b, 0) / Math.max(1, share.length);
  val("x-res-meta", [pct(mean), "mean"], wr >= 0 ? "worst L" + il[wr] + " " + pct(share[wr]) : null);

  let rates = store.layers.rates();
  if (!rates.some(isNum)) {
    rates = m.layers.hit.map((h, i) => (h + m.layers.miss[i] ? h / (h + m.layers.miss[i]) : null));
    txt("x-layer-win", "since start");
  } else {
    txt("x-layer-win", "last " + store.layers.windowS + " s");
  }
  const wl = layerBars("x-layers", rates, il, (v) => pct(v));
  const vals = rates.filter(isNum), rm = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  val("x-layer-meta", [pct(rm), "mean"], wl >= 0 ? "worst L" + il[wl] + " " + pct(rates[wl]) : null);
  txt("x-l0", "L" + il[0]);
  txt("x-ln", "L" + il[L - 1]);
  drawPlane({ n_layer: L, n_expert: E, slot_expert: m.slot_expert });
}

export function render(st) {
  const m = st.moe_cache;
  show("p-experts", !!m);
  if (!m) return;
  // n_expert: the experts of the cached layers, all layers; n_slots: the cache's slots, all layers
  const d = latest(), L = m.n_layer, total = m.n_expert, resident = m.n_resident;
  txt("x-meta", L + " layers · " + num(total) + " experts · " + num(m.n_slots) + " slots · " +
    (m.suspended ? "released: prefill mode" : "decode"));

  const share = total ? resident / total : null;
  width("x-tier-v", share || 0);
  width("x-tier-p", isNum(share) ? 1 - share : 0);
  val("x-vram", pct(share), num(resident) + " experts", m.device_bytes ? bytes(m.device_bytes) : null);
  val("x-pinned", isNum(share) ? pct(1 - share) : null, num(total - resident) + " experts");

  const c = m.counters, cum = c.hit + c.miss ? c.hit / (c.hit + c.miss) : null;
  const hr = isNum(d.moe_hit_rate) ? d.moe_hit_rate : lastNum("moe_hit_rate");
  val("x-hit", pct(isNum(hr) ? hr : cum), isNum(d.moe_miss_s) ? num(d.moe_miss_s, 0) + " missed/s" : null, "cumulative " + pct(cum));
  width("x-hit-bar", isNum(hr) ? hr : cum || 0);

  const perLayer = !!(m.layers && m.slot_expert && m.n_expert_layer);
  show("x-per-layer", perLayer);
  show("x-plane-head", perLayer);
  show("x-planebox", perLayer);
  if (perLayer) renderLayers(m);
}
