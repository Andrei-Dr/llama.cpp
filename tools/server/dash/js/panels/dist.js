// distributions (the server's histograms) and the page's events

import { $, txt, num, el, sized } from "../util.js";

const ms = (v) => (v >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 0 : 1) + "s" : v.toFixed(0));

function hist(id, lid, h, fmt) {
  if (!h) return;
  const bars = sized($(id), h.counts.length);
  const mx = Math.max(1, ...h.counts);
  h.counts.forEach((n, i) => {
    bars[i].style.height = ((100 * n) / mx).toFixed(1) + "%";
    const lo = i === 0 ? "≤ " : i < h.edges.length ? fmt(h.edges[i - 1]) + "–" : "> ";
    bars[i].title = lo + fmt(h.edges[Math.min(i, h.edges.length - 1)]) + ": " + n;
  });
  $(lid).replaceChildren(el("span", "", "≤" + fmt(h.edges[0])),
    el("span", "", "n " + num(h.n) + (h.n ? " · mean " + fmt(h.sum / h.n) : "") + " · >" + fmt(h.edges[h.edges.length - 1])));
}

export function renderHists(st) {
  const H = st.hist;
  hist("h-ttft", "hl-ttft", H.ttft_ms, ms);
  hist("h-mpt", "hl-mpt", H.gen_ms_per_token, ms);
  hist("h-step", "hl-step", H.step_ms, ms);
  hist("h-acc", "hl-acc", H.accept, (v) => (100 * v).toFixed(0) + "%");
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
