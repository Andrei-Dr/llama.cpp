// sparklines (the cards).
//
// A sparkline shows variation; the figure above it states the level. So the axis is the window's own range, and its ends
// are the 5th and 95th percentiles, not the extremes: on a min/max axis one rare spike owns the box and the typical
// variation is squeezed into a pixel. Values past the 95th draw pinned to the top. A window that is flat apart from rare
// spikes has both percentiles on one value; the extremes then serve, so the spikes still show. A flat window draws down
// the middle. (Design after radiance's dashboard, re-implemented here.)

import { isNum } from "./util.js";

function quantile(sorted, p) {
  return sorted[Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)))];
}

export function percentileRange(values) {
  const v = values.filter(isNum);
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b);
  let lo = quantile(s, 0.05), hi = quantile(s, 0.95);
  if (!(hi > lo)) { lo = s[0]; hi = s[s.length - 1]; }
  return { lo, hi };
}

// points: [t, v] in time order (v null = a gap). Draws into an svg path over [t0, t1] x the svg's viewBox.
export function pathOf(points, t0, t1, range, w, h) {
  const inset = 1.5;
  let d = "", pen = false;
  for (const [t, v] of points) {
    if (!isNum(v)) { pen = false; continue; }
    const x = ((t - t0) / Math.max(1e-9, t1 - t0)) * w;
    let f = 0.5;
    if (range && range.hi > range.lo) f = Math.max(0, Math.min(1, (v - range.lo) / (range.hi - range.lo)));
    const y = h - inset - f * (h - 2 * inset);
    d += (pen ? "L" : "M") + x.toFixed(2) + " " + y.toFixed(2);
    pen = true;
  }
  return d;
}

// one or more series (keys of the samples) into the paths of an svg, over the last span seconds. shared: one range for
// every series (they share a unit), else each its own. fixed: {lo, hi} instead of percentiles.
export function spark(svg, samples, keys, now, { span = 300, shared = true, fixed = null } = {}) {
  if (!svg) return;
  const vb = svg.viewBox.baseVal, w = vb.width || 200, h = vb.height || 24;
  const t0 = Math.max(now - span, samples.length ? samples[0].t : now - 1);
  const win = samples.filter((s) => s.t >= t0);
  const paths = svg.querySelectorAll("path");
  const all = shared ? percentileRange(win.flatMap((s) => keys.map((k) => s[k]))) : null;
  keys.forEach((k, i) => {
    if (!paths[i]) return;
    const pts = win.map((s) => [s.t, s[k]]);
    const range = fixed || (shared ? all : percentileRange(pts.map((p) => p[1])));
    paths[i].setAttribute("d", pts.filter((p) => isNum(p[1])).length < 2 ? "" : pathOf(pts, t0, now, range, w, h));
  });
}
