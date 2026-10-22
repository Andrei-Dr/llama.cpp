// the dashboard: /stats and the newest /stats/history rows every second (one poll in flight at most), the minute rows
// every minute, /props every 15 s; every panel renders from the store

import { $, txt, dur, show } from "./util.js";
import { store, poll, failed, pollMinutes, pollProps, HttpError } from "./store.js";
import * as cards from "./panels/cards.js";
import * as timeline from "./panels/timeline.js";
import * as requests from "./panels/requests.js";
import * as slots from "./panels/slots.js";
import * as memory from "./panels/memory.js";
import * as experts from "./panels/experts.js";
import * as link from "./panels/link.js";
import * as machine from "./panels/machine.js";
import * as dist from "./panels/dist.js";

function header() {
  const st = store.stats, p = store.props || {};
  $("dot").className = "dot " + (store.ok ? "ok" : "bad");
  txt("title", p.model_alias || "llama-server");
  txt("chip-build", p.build_info || "—");
  txt("chip-model", (p.model_ftype || "—") + (p.is_sleeping ? " · sleeping" : ""));
  if (store.ok && st) {
    txt("hdr-right", st.seq ? "up " + dur(1000 * st.uptime_s) + " · snapshot " + st.seq + " · " + new Date().toLocaleTimeString() : "no snapshot yet");
  }
  const e = store.error, auth = e instanceof HttpError && e.status === 401, off = e instanceof HttpError && e.status === 501;
  show("banner-auth", auth);
  show("banner-metrics", off);
  show("banner-history", store.ok && !store.historyOn);
  const down = $("banner-down");
  down.hidden = store.ok || auth || off;
  if (!down.hidden) {
    down.textContent = "llama-server unreachable: " + (e ? e.message : "?") + (store.lastOk ? " (last answer " + dur(Date.now() - store.lastOk) + " ago)" : "");
  }
}

// one panel failing must not blank the others: each renders in its own try
function guarded(name, fn) {
  try {
    fn();
  } catch (e) {
    console.error("dashboard: panel " + name + " failed", e);
  }
}

function render() {
  header();
  const st = store.stats;
  if (!st || !st.counters) return;
  guarded("cards", () => cards.render(st));
  guarded("timeline", () => timeline.render());
  guarded("requests", () => requests.render(st));
  guarded("slots", () => slots.renderSlots(st));
  guarded("spec", () => slots.renderSpec(st));
  guarded("memory", () => memory.render(st));
  guarded("experts", () => experts.render(st));
  guarded("link", () => link.render(st));
  guarded("machine", () => machine.renderMachine(st));
  guarded("phase", () => machine.renderPhase());
  guarded("hists", () => dist.renderHists(st));
  guarded("events", () => dist.renderEvents(store.events));
}

let busy = false;
async function tick() {
  if (busy) return; // a slow answer must not overlap the next poll
  busy = true;
  try {
    await poll();
  } catch (e) {
    failed(e);
  } finally {
    busy = false;
  }
  render();
}

async function quiet(fn) {
  try {
    await fn();
  } catch (e) {
    /* the next round; the 1 s poll reports an unreachable server */
  }
}

quiet(pollProps);
tick().then(() => quiet(pollMinutes));
setInterval(tick, 1000);
setInterval(() => quiet(pollMinutes), 60000);
setInterval(() => quiet(pollProps), 15000);
