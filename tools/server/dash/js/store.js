// the page's data: the latest /stats, the history rows (/stats/history, tier s every poll from the newest row on, tier m
// every minute) and the samples derived from them, the completed tasks, the 60 s windows, and the page's own events.
// Every request goes to this server, relative to /dash/ (so an --api-prefix works); an API key, when the server needs
// one, comes from the page's #key=... fragment once and stays in this tab's sessionStorage.

import { rowsOf, deriveAll, mergeTasks, LayerWindow, PhaseWindow, isNum } from "./derive.js";

const KEEP_TASKS = 60;
const KEEP_EVENTS = 50;

(function takeKey() {
  const m = /(?:^#|&)key=([^&]+)/.exec(location.hash);
  if (m) {
    let k = m[1];
    try {
      k = decodeURIComponent(k);
    } catch (e) {
      /* a key that is not percent-encoded is taken as typed */
    }
    sessionStorage.setItem("llama-dash-key", k);
    history.replaceState(null, "", location.pathname + location.search); // the key leaves the address bar
  }
})();

function headers() {
  const k = sessionStorage.getItem("llama-dash-key");
  return k ? { Authorization: "Bearer " + k } : {};
}

export class HttpError extends Error {
  constructor(status, text) {
    super(status + " " + text);
    this.status = status;
  }
}

async function getJSON(path) {
  // a dead server must not hang the poll loop: give up after 4 s
  const r = await fetch("../" + path, { headers: headers(), cache: "no-store", signal: AbortSignal.timeout(4000) });
  if (!r.ok) throw new HttpError(r.status, r.statusText);
  return r.json();
}

export const store = {
  stats: null,       // the latest /stats
  props: null,       // /props, every 15 s
  now: 0,            // the server's clock at the latest answer (s, same clock as the samples' t)
  rows: [],          // tier s rows, oldest first
  samples: [],       // derived from rows, aligned
  minuteRows: [],
  minuteSamples: [],
  historyOn: true,   // the server samples (LLAMA_SERVER_SAMPLE_MS > 0)
  periodS: 1,        // the tiers' row periods and spans (period x rows the server keeps)
  minutePeriodS: 60,
  spanS: 900,
  minuteSpanS: 86400,
  tasks: [],
  layers: new LayerWindow(60),
  phase: new PhaseWindow(60),
  events: [],
  instance: null,
  ok: false,
  lastOk: 0,         // Date.now() of the last full poll
  error: null,
  hoverT: null,      // the time the pointer is over in the timeline (null = the newest)
};

export function event(name, detail = "") {
  store.events.push({ t: Date.now(), name, detail });
  if (store.events.length > KEEP_EVENTS) store.events.splice(0, store.events.length - KEEP_EVENTS);
}

function appendRows(h) {
  const fresh = rowsOf(h);
  const last = store.rows[store.rows.length - 1] || null;
  const add = last ? fresh.filter((r) => r.t_us > last.t_us) : fresh;
  if (!add.length) return;
  store.rows.push(...add);
  store.samples.push(...deriveAll(add, last));
  const cut = add[add.length - 1].t_us - store.spanS * 1e6;
  let i = 0;
  while (i < store.rows.length && store.rows[i].t_us < cut) i++;
  if (i) {
    store.rows.splice(0, i);
    store.samples.splice(0, i);
  }
}

export async function poll() {
  const first = !store.stats;
  const last = store.rows[store.rows.length - 1];
  const [st, h] = await Promise.all([
    getJSON("stats" + (first ? "" : "?tasks=16")),
    store.historyOn ? getJSON("stats/history" + (last ? "?since_us=" + last.t_us : "")) : Promise.resolve(null),
  ]);
  if (store.instance !== null && st.t_start_us && st.t_start_us !== store.instance) {
    event("server restarted", "up " + Math.round(st.uptime_s || 0) + " s");
    store.layers = new LayerWindow(60);
    store.phase = new PhaseWindow(60);
  }
  if (st.t_start_us) store.instance = st.t_start_us;
  // the server's clock went back (a reboot, or a platform whose clock starts with the process): start the rows over
  if (last && st.t_serve_us < last.t_us) {
    store.rows = [];
    store.samples = [];
  }
  store.stats = st;
  store.now = st.t_serve_us / 1e6;
  if (h) {
    store.historyOn = h.enabled !== false;
    if (h.period_ms > 0) store.periodS = h.period_ms / 1000;
    if (h.period_ms > 0 && h.cap > 0) store.spanS = (h.period_ms * h.cap) / 1000;
    if (!(last && st.t_serve_us < last.t_us)) appendRows(h);
  }
  store.tasks = mergeTasks(store.tasks, st.tasks, st.t_start_us, KEEP_TASKS);
  const m = st.moe_cache;
  if (m) store.layers.push(store.now, m.layers.hit, m.layers.miss);
  if (st.phase) store.phase.push(store.now, st.phase);
  if (!store.ok) event(store.lastOk ? "reachable again" : "connected");
  store.ok = true;
  store.error = null;
  store.lastOk = Date.now();
}

export function failed(e) {
  if (store.ok) event("unreachable", e.message);
  store.ok = false;
  store.error = e;
}

export async function pollMinutes() {
  if (!store.historyOn) return;
  const h = await getJSON("stats/history?tier=m");
  if (h.period_ms > 0) store.minutePeriodS = h.period_ms / 1000;
  if (h.period_ms > 0 && h.cap > 0) store.minuteSpanS = (h.period_ms * h.cap) / 1000;
  store.minuteRows = rowsOf(h);
  store.minuteSamples = deriveAll(store.minuteRows);
}

export async function pollProps() {
  const p = await getJSON("props");
  if (store.props && !!store.props.is_sleeping !== !!p.is_sleeping) event(p.is_sleeping ? "sleeping" : "awake");
  store.props = p;
}

export function latest() {
  return store.samples.length ? store.samples[store.samples.length - 1] : {};
}

// the newest measured value of a series (a sample without it or with null does not count)
export function lastNum(key) {
  for (let i = store.samples.length - 1; i >= 0; i--) if (isNum(store.samples[i][key])) return store.samples[i][key];
  return null;
}

// samples of the last span seconds
export function windowOf(span) {
  return store.samples.filter((s) => s.t >= store.now - span);
}

export const sum = (arr, f) => arr.reduce((a, x) => a + f(x), 0);
