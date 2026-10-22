// the timeline: one small chart per series group over the last 15 minutes (every sample) or 24 hours (minute rows),
// a zero-based axis with its top rounded to a readable number, gaps where nothing was measured, and one crosshair shared
// by every chart: the pointer over any of them shows every chart's values at that moment.

import { isNum, el } from "./util.js";

const SVG = "http://www.w3.org/2000/svg";
const W = 300, H = 80;

function svgEl(tag, cls) {
  const e = document.createElementNS(SVG, tag);
  if (cls) e.setAttribute("class", cls);
  return e;
}

// 1, 2, 2.5, 5 x 10^k at or above v
export function niceMax(v) {
  if (!isNum(v) || v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v * 0.9999) return m * p;
  return 10 * p;
}

// an axis label: whole numbers as they are, the rest with the digits a nice step needs, 10000 and up in k
export function axisNum(v) {
  if (v >= 1e4) return (v / 1e3).toLocaleString("en-US", { maximumFractionDigits: 1 }) + "k";
  return v.toLocaleString("en-US", { maximumFractionDigits: Number.isInteger(v) ? 0 : v < 1 ? 3 : 2 });
}

// index of the sample nearest to t (samples in time order)
export function nearest(samples, t) {
  let lo = 0, hi = samples.length - 1;
  if (hi < 0) return -1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (samples[mid].t < t) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && Math.abs(samples[lo - 1].t - t) <= Math.abs(samples[lo].t - t)) lo--;
  return lo;
}

// an svg path of [t, v] points over [t0, t1] x [0, hi]; the pen lifts at a null and across a gap longer than gapS
function pathOf(points, t0, t1, hi, gapS) {
  let d = "", pen = false, tPrev = null;
  for (const [t, v] of points) {
    if (!isNum(v) || (tPrev !== null && t - tPrev > gapS)) pen = false;
    tPrev = t;
    if (!isNum(v)) continue;
    const x = ((t - t0) / Math.max(1e-9, t1 - t0)) * W;
    const y = H - 1 - Math.max(0, Math.min(1, v / hi)) * (H - 2);
    d += (pen ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
    pen = true;
  }
  return d;
}

// spec: { id, cap, unit, series: [{ key, cls, name, k }], hi?, fmt, show(stats) }
export class Chart {
  constructor(box, spec, onHover) {
    this.spec = spec;
    this.root = el("figure", "tl");
    const head = el("figcaption");
    head.appendChild(el("span", "cap", spec.cap));
    this.vals = el("span", "vals");
    head.appendChild(this.vals);
    this.valEls = spec.series.map((s) => {
      const b = el("b", "c-" + s.cls);
      b.title = s.name || s.key;
      this.vals.appendChild(b);
      return b;
    });
    this.unit = el("small", "", spec.unit);
    this.vals.appendChild(this.unit);

    const body = el("div", "tlbody");
    this.yax = el("div", "yax");
    this.yLabels = [el("span"), el("span"), el("span")];
    this.yLabels.forEach((s) => this.yax.appendChild(s));
    this.svg = svgEl("svg", "plot");
    this.svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    this.svg.setAttribute("preserveAspectRatio", "none");
    for (const f of [0.5]) {
      const r = svgEl("line", "rule");
      r.setAttribute("x1", "0"); r.setAttribute("x2", String(W));
      r.setAttribute("y1", String(H - 1 - f * (H - 2))); r.setAttribute("y2", String(H - 1 - f * (H - 2)));
      this.svg.appendChild(r);
    }
    this.paths = spec.series.map((s) => {
      const p = svgEl("path", "s-" + s.cls);
      this.svg.appendChild(p);
      return p;
    });
    this.cursor = svgEl("line", "cursor");
    this.cursor.setAttribute("y1", "0");
    this.cursor.setAttribute("y2", String(H));
    this.svg.appendChild(this.cursor);
    this.xax = el("div", "xax");
    body.append(this.yax, this.svg, el("div"), this.xax);
    this.root.append(head, body);
    if (spec.names) this.root.appendChild(el("div", "tlkey", spec.names));
    box.appendChild(this.root);

    this.svg.addEventListener("pointermove", (e) => {
      const r = this.svg.getBoundingClientRect();
      if (r.width > 0 && this.t1 > this.t0) onHover(this.t0 + ((e.clientX - r.left) / r.width) * (this.t1 - this.t0));
    });
    this.svg.addEventListener("pointerleave", () => onHover(null));
  }

  // samples: time-ordered derived samples; [t0, t1] the window; hoverT the shared crosshair (null = newest)
  render(samples, t0, t1, gapS, hoverT, stats) {
    const spec = this.spec;
    const on = !spec.show || spec.show(stats);
    this.root.hidden = !on;
    if (!on) return;
    this.t0 = t0;
    this.t1 = t1;
    const win = samples.filter((s) => s.t >= t0 && s.t <= t1);
    const val = (s, ser) => (isNum(s[ser.key]) ? s[ser.key] * (ser.k || 1) : null);

    let vmax = 0;
    for (const ser of spec.series) for (const s of win) { const v = val(s, ser); if (isNum(v) && v > vmax) vmax = v; }
    const hi = spec.hi || niceMax(vmax);
    spec.series.forEach((ser, i) => {
      this.paths[i].setAttribute("d", pathOf(win.map((s) => [s.t, val(s, ser)]), t0, t1, hi, gapS));
    });
    axisLabels(this.xax, t1 - t0);
    const f = spec.fmt;
    this.yLabels[0].textContent = axisNum(hi);
    this.yLabels[1].textContent = axisNum(hi / 2);
    this.yLabels[2].textContent = "0";

    // the values at the crosshair, or the newest measured ones
    // (a moment with no sample within a gap of it shows nothing, not the nearest sample minutes away)
    const hovering = hoverT !== null && hoverT >= t0 && hoverT <= t1;
    let at = null;
    if (hovering) {
      const i = win.length ? nearest(win, hoverT) : -1;
      at = i >= 0 && Math.abs(win[i].t - hoverT) <= gapS ? win[i] : null;
      const x = (((at ? at.t : hoverT) - t0) / (t1 - t0)) * W;
      this.cursor.setAttribute("x1", x.toFixed(1));
      this.cursor.setAttribute("x2", x.toFixed(1));
    }
    this.cursor.classList.toggle("on", hovering);
    spec.series.forEach((ser, i) => {
      let v = null;
      if (hovering) v = at ? val(at, ser) : null;
      else for (let j = win.length - 1; j >= 0 && !isNum(v); j--) v = val(win[j], ser);
      const s = isNum(v) ? f(v) : "—";
      if (this.valEls[i].textContent !== s) this.valEls[i].textContent = s;
    });
  }
}

// a span back from now: "-90 s", "-15 min", "-24 h"
function ago(s) {
  if (s >= 7200) return "-" + +(s / 3600).toFixed(1) + " h";
  if (s >= 120) return "-" + +(s / 60).toFixed(1) + " min";
  return "-" + Math.round(s) + " s";
}

// the x axis: thirds of the window back from now
export function axisLabels(box, spanS) {
  const labs = [spanS, (2 * spanS) / 3, spanS / 3].map(ago).concat(["now"]);
  if (box.children.length !== labs.length) box.replaceChildren(...labs.map((l) => el("span", "", l)));
  else labs.forEach((l, i) => { if (box.children[i].textContent !== l) box.children[i].textContent = l; });
}

