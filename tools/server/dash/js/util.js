// formatting and DOM helpers shared by every panel. Server data is only ever written with textContent.

import { isNum } from "./derive.js";

export { isNum };
export const $ = (id) => document.getElementById(id);

export function txt(id, s) {
  const e = typeof id === "string" ? $(id) : id;
  if (e && e.textContent !== s) e.textContent = s;
}

export function show(id, on) {
  const e = $(id);
  if (e) e.hidden = !on;
}

export function num(v, d = 0) {
  return isNum(v) ? v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }) : "—";
}

export function pct(v, d = 1) {
  return isNum(v) ? (100 * v).toFixed(d) + "%" : "—";
}

export function bytes(b) {
  if (!isNum(b)) return "—";
  const u = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0;
  while (Math.abs(b) >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return b.toFixed(i ? 1 : 0) + " " + u[i];
}

export function dur(ms) {
  if (!isNum(ms)) return "—";
  if (ms < 1) return ms.toFixed(2) + " ms";
  if (ms < 1000) return ms.toFixed(0) + " ms";
  if (ms < 60000) return (ms / 1000).toFixed(2) + " s";
  if (ms < 3600000) return (ms / 60000).toFixed(1) + " min";
  return (ms / 3600000).toFixed(1) + " h";
}

export function compact(n) {
  if (!isNum(n)) return "—";
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (Math.abs(n) >= 1e4) return (n / 1e3).toFixed(1) + "k";
  return num(n);
}

const unknown = (v) => v === null || v === undefined || v === "" || v === "—";

// a row's value: the main figure, then muted secondary parts. A row with no main figure has no source and is hidden
// instead of showing a dash.
export function val(id, main, ...rest) {
  const e = $(id);
  if (!e) return;
  const [m, unit] = Array.isArray(main) ? main : [main, null];
  const key = [m, unit, ...rest].join("\u0001");
  const row = e.closest(".kv, .m");
  if (e.dataset.k !== key) {
    e.dataset.k = key;
    const lead = el("span", "mv", unknown(m) ? "" : m);
    if (unit && !unknown(m)) lead.appendChild(el("small", "u", unit));
    e.replaceChildren(lead);
    // a key-value row puts its figure at the right edge (row-reverse), so its secondary parts are appended backwards
    const parts = rest.filter((r) => !unknown(r));
    for (const r of row && row.classList.contains("kv") ? parts.reverse() : parts) e.appendChild(el("small", "", r));
  }
  if (row) row.hidden = unknown(m);
}

// "a · b" of the parts that exist
export function known(...parts) {
  const p = parts.filter((x) => !unknown(x));
  return p.length ? p.join(" · ") : null;
}

// groups (subheads) with no visible row are hidden too
export function tidy() {
  for (const g of document.querySelectorAll(".grp")) {
    g.hidden = !g.querySelector(".kv:not([hidden]), .plotbox");
  }
  // a section heading with every panel under it hidden is hidden too
  for (const b of document.querySelectorAll(".band")) b.dataset.empty = b.dataset.for.split(" ").every((id) => $(id).hidden) ? "1" : "";
}

// a big figure with its unit in a <small>; a stat with no value has no source and is hidden
export function figure(id, value, unit) {
  const e = $(id);
  if (!e) return;
  const stat = e.closest(".stat");
  if (stat) stat.hidden = unknown(value);
  e.textContent = value;
  if (unit) {
    const s = document.createElement("small");
    s.textContent = unit;
    e.appendChild(s);
  }
}

export function width(el, frac) {
  const e = typeof el === "string" ? $(el) : el;
  if (e) e.style.width = (Math.max(0, Math.min(1, isNum(frac) ? frac : 0)) * 100).toFixed(2) + "%";
}

export function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

// keep a container's children at n elements of one tag
export function sized(box, n, tag = "i") {
  if (box.children.length !== n) box.replaceChildren(...Array.from({ length: n }, () => document.createElement(tag)));
  return box.children;
}
