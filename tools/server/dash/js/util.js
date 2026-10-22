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

// a big figure with its unit in a <small>
export function figure(id, value, unit) {
  const e = $(id);
  if (!e) return;
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
