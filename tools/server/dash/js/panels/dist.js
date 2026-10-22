// distributions (the server's histograms) and the page's events

import { $, txt, num, el, sized } from "../util.js";

const ms = (v) => (v >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 0 : 1) + " s" : v.toFixed(0) + " ms");

// one histogram: bars on the largest bucket (a used bucket is never thinner than 2 px), three axis ticks, and a summary line
function hist(id, sid, lid, h, fmt) {
  if (!h) return;
  const box = $(id), n = h.n || 0;
  box.classList.toggle("none", !n);
  const bars = sized(box, h.counts.length);
  const mx = Math.max(1, ...h.counts);
  h.counts.forEach((c, i) => {
    bars[i].style.height = c ? Math.max(3, (100 * c) / mx).toFixed(1) + "%" : "0";
    bars[i].className = c ? "on" : "";
    const lo = i === 0 ? "≤ " : i < h.edges.length ? fmt(h.edges[i - 1]) + "–" : "> ";
    bars[i].title = lo + fmt(h.edges[Math.min(i, h.edges.length - 1)]) + ": " + num(c);
  });
  const e = h.edges, mid = e[Math.floor((e.length - 1) / 2)];
  $(lid).replaceChildren(el("span", "", "≤" + fmt(e[0])), el("span", "", fmt(mid)), el("span", "", ">" + fmt(e[e.length - 1])));
  const s = $(sid);
  s.replaceChildren(...(n ? [el("span", "", "n"), el("b", "", num(n)), el("span", "", "mean"), el("b", "", fmt(h.sum / n))] : [el("span", "", "no samples yet")]));
}

export function renderHists(st) {
  const H = st.hist;
  hist("h-ttft", "hs-ttft", "hl-ttft", H.ttft_ms, ms);
  hist("h-mpt", "hs-mpt", "hl-mpt", H.gen_ms_per_token, ms);
  hist("h-step", "hs-step", "hl-step", H.step_ms, ms);
  hist("h-acc", "hs-acc", "hl-acc", H.accept, (v) => (100 * v).toFixed(0) + "%");
}

export function renderEvents(evs) {
  const frag = document.createDocumentFragment();
  for (const e of evs.slice(-30).reverse()) {
    const row = el("div");
    row.appendChild(el("b", "", new Date(e.t).toLocaleString() + "  " + e.name));
    if (e.detail) row.appendChild(document.createTextNode("  " + e.detail));
    frag.appendChild(row);
  }
  $("events").replaceChildren(frag);
  txt("ev-meta", evs.length + " since this page opened");
}
