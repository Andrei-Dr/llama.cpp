// Pure functions over the server's numbers: rates from two /stats/history rows, the completed-task merge, and the
// 60-second windows over /stats (per-layer hit rate, the CPU-expert phase). No DOM: tests/dash runs them under node.
//
// A history row holds cumulative counters and gauges (tools/server/server-stats.h, SERVER_STATS_HISTORY_COLUMNS). A
// rate is a difference over the clock of the snapshot the counters come from: the server's (snap_t_us), the expert
// cache's (moe_t_us), or the sampler's (t_us) for the machine. Across a server restart or a counter that went back,
// a rate is null, never a negative or a spike.

export const isNum = (v) => typeof v === "number" && Number.isFinite(v);

// columnar /stats/history -> rows (objects keyed by column name), oldest first
export function rowsOf(h) {
  const cols = (h && h.cols) || {};
  const keys = Object.keys(cols);
  const n = cols.t_us ? cols.t_us.length : 0;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const r = {};
    for (const k of keys) r[k] = cols[k][i];
    out[i] = r;
  }
  return out;
}

function delta(p, c, k) {
  if (!isNum(p[k]) || !isNum(c[k])) return null;
  const d = c[k] - p[k];
  return d >= 0 ? d : null;
}

// k * v, null stays null (100 * null would be 0)
const scale = (k, v) => (isNum(v) ? k * v : null);

function ratio(num, den) {
  return isNum(num) && isNum(den) && den > 0 ? num / den : null;
}

// a rate over a clock that may not have moved: no newer snapshot and nothing counted is a zero when the source is idle,
// unknown when it is busy (a long batch that has not published yet)
function rateOver(d, dt, busy) {
  if (!isNum(d)) return null;
  if (isNum(dt) && dt > 0) return d / dt;
  return d === 0 && !busy ? 0 : null;
}

// the server's uptime at the time it answered (t_serve_us), not at its last snapshot: an idle server stops publishing, so
// /stats uptime_s stands still between requests
export function uptimeS(st) {
  return st && isNum(st.t_serve_us) && isNum(st.t_start_us) && st.t_start_us > 0 ? (st.t_serve_us - st.t_start_us) / 1e6 : st && st.uptime_s;
}

// seconds of prompt processing the disk tier saved: the reused tokens that came from a restore (prompt_tokens_cached_disk) at
// the measured prefill rate, less the time the restores took (all of them, also those that reused nothing). Inputs are
// cumulative counters of one server; null unless the prefill rate is known and the disk counters exist.
export function diskSavedS(promptTokens, promptUs, cachedDisk, restoreUs) {
  if (!isNum(promptTokens) || !isNum(promptUs) || promptTokens <= 0 || promptUs <= 0) return null;
  if (!isNum(cachedDisk) || !isNum(restoreUs) || restoreUs <= 0) return null; // no restore yet: nothing saved or spent
  return (cachedDisk * promptUs) / promptTokens / 1e6 - restoreUs / 1e6;
}

const SERVER_KEYS = ["decode_tps", "prefill_tput", "tasks_s", "step_ms", "decode_ms", "tokens_per_step", "accept",
  "prefill_tps", "cpu_ms", "cpu_share", "publish_us", "disk_restore_mbs", "disk_save_mbs", "disk_hit_rate", "prompt_reuse"];
const MOE_KEYS = ["moe_hit_rate", "moe_miss_s", "moe_upload_s", "moe_evict_s", "moe_upload_mbs", "moe_warm_mbs",
  "moe_fill_mbs", "moe_readmit_s"];
const MACHINE_RATE_KEYS = ["cpu_pct", "iowait_pct", "server_cpu_pct", "server_share_pct"];

// one sample (rates and gauges) at row c, from the row before it (p may be null: gauges only)
export function derive(p, c) {
  const s = {
    t: c.t_us / 1e6,
    unix_ms: c.unix_ms,
    instance: c.t_start_us,
    // gauges
    in_flight: c.n_busy, waiting: c.n_deferred, kv_used: c.kv_used, kv_max: c.kv_max,
    moe_resident: c.moe_resident, disk_bytes: c.disk_bytes, disk_entries: c.disk_entries,
    gpu_sm: c.gpu_util, gpu_mem_bw: c.gpu_mem_util,
    pcie_rx_mbs: isNum(c.pcie_rx) ? c.pcie_rx / 1e6 : null, pcie_tx_mbs: isNum(c.pcie_tx) ? c.pcie_tx / 1e6 : null,
    vram_used: c.vram_used, vram_total: c.vram_total, gpu_power_w: c.gpu_power_w, gpu_temp_c: c.gpu_temp_c,
    gpu_clock_mhz: c.gpu_clock_mhz, mem_avail: c.mem_avail, mem_total: c.mem_total, swap_used: c.swap_used,
    sample_us: c.sample_us,
    kv_fill_pct: ratio(scale(100, c.kv_used), c.kv_max),
    // since the server started (cumulative counters of this row): the disk tier against prefill
    prefill_tps_avg: ratio(scale(1e6, c.prompt_tokens), c.prompt_us),
    // the share of the prompt tokens not processed again (a slot held them or a disk restore loaded them). Since start:
    // the reuse is counted when a prompt starts and its processed tokens as they are computed, so a per-interval ratio
    // of the two jumps to ~100% and then reads 0% while a long remainder is processed
    prompt_reuse_avg: ratio(c.prompt_cached, isNum(c.prompt_cached) && isNum(c.prompt_tokens) ? c.prompt_cached + c.prompt_tokens : null),
    disk_restore_tps_avg: ratio(scale(1e6, c.disk_restore_tokens), c.disk_restore_us),
    disk_saved_s: diskSavedS(c.prompt_tokens, c.prompt_us, c.prompt_cached_disk, c.disk_restore_us),
  };
  for (const k of [...SERVER_KEYS, ...MOE_KEYS, ...MACHINE_RATE_KEYS]) s[k] = null;
  if (!p) return s;

  // the machine: the sampler's clock
  const dt = (c.t_us - p.t_us) / 1e6;
  const dTotal = delta(p, c, "host_total_us");
  s.cpu_pct = ratio(scale(100, delta(p, c, "host_busy_us")), dTotal);
  s.iowait_pct = ratio(scale(100, delta(p, c, "host_iowait_us")), dTotal);
  const dProc = delta(p, c, "proc_cpu_us");
  s.server_cpu_pct = dt > 0 && isNum(dProc) ? (100 * dProc) / (dt * 1e6) : null; // of one core
  s.server_share_pct = ratio(scale(100, dProc), dTotal); // of the whole machine

  // the server: same instance, the snapshot's clock
  if (isNum(c.t_start_us) && c.t_start_us === p.t_start_us) {
    const busy = c.n_busy > 0;
    const dts = (c.snap_t_us - p.snap_t_us) / 1e6;
    const dGen = delta(p, c, "gen_tokens"), dPrompt = delta(p, c, "prompt_tokens");
    const dStepN = delta(p, c, "step_n"), dDecN = delta(p, c, "decode_n");
    s.decode_tps = rateOver(dGen, dts, busy);
    s.prefill_tput = rateOver(dPrompt, dts, busy);
    s.tasks_s = rateOver(delta(p, c, "tasks"), dts, busy);
    s.step_ms = ratio(delta(p, c, "step_us"), scale(1000, dStepN));
    s.decode_ms = ratio(delta(p, c, "decode_us"), scale(1000, dDecN));
    s.tokens_per_step = ratio(dGen, dStepN);
    s.accept = ratio(delta(p, c, "draft_accepted"), delta(p, c, "draft_tokens"));
    s.prefill_tps = ratio(scale(1e6, dPrompt), delta(p, c, "prompt_us"));
    s.cpu_ms = ratio(delta(p, c, "phase_cpu_us"), scale(1000, dDecN));
    s.cpu_share = ratio(delta(p, c, "phase_cpu_us"), delta(p, c, "decode_us"));
    s.publish_us = ratio(delta(p, c, "publish_us"), delta(p, c, "publish_n"));
    s.disk_restore_mbs = rateOver(delta(p, c, "disk_restore_bytes"), dts, busy);
    s.disk_save_mbs = rateOver(delta(p, c, "disk_save_bytes"), dts, busy);
    s.disk_hit_rate = ratio(delta(p, c, "disk_find_hit"), delta(p, c, "disk_find"));
    // the share of the prompt tokens not processed again (a slot held them or a disk restore loaded them), in the interval
    // (only intervals that read a prompt count)
    const dCached = delta(p, c, "prompt_cached");
    s.prompt_reuse = ratio(dCached, isNum(dCached) && isNum(dPrompt) ? dCached + dPrompt : null);
    if (isNum(s.disk_restore_mbs)) s.disk_restore_mbs /= 1e6;
    if (isNum(s.disk_save_mbs)) s.disk_save_mbs /= 1e6;
  }

  // the expert cache: its own snapshot's clock (it publishes at every decode step, so an idle cache stands still)
  if (isNum(c.moe_t_us) && isNum(p.moe_t_us) && c.moe_t_us >= p.moe_t_us) {
    const dtm = (c.moe_t_us - p.moe_t_us) / 1e6;
    const dHit = delta(p, c, "moe_hit"), dMiss = delta(p, c, "moe_miss");
    s.moe_hit_rate = isNum(dHit) && isNum(dMiss) ? ratio(dHit, dHit + dMiss) : null;
    s.moe_miss_s = rateOver(dMiss, dtm, false);
    s.moe_upload_s = rateOver(delta(p, c, "moe_upload"), dtm, false);
    s.moe_evict_s = rateOver(delta(p, c, "moe_evict"), dtm, false);
    s.moe_readmit_s = rateOver(delta(p, c, "moe_readmit"), dtm, false);
    const up = rateOver(delta(p, c, "moe_upload_bytes"), dtm, false);
    const warm = rateOver(delta(p, c, "moe_upload_bytes_warm"), dtm, false);
    s.moe_upload_mbs = isNum(up) ? up / 1e6 : null;
    s.moe_warm_mbs = isNum(warm) ? warm / 1e6 : null;
    s.moe_fill_mbs = isNum(up) && isNum(warm) ? Math.max(0, up - warm) / 1e6 : s.moe_upload_mbs;
  }
  return s;
}

// samples for rows appended after prev (the last row already derived, or null)
export function deriveAll(rows, prev = null) {
  const out = [];
  let p = prev;
  for (const r of rows) {
    out.push(derive(p, r));
    p = r;
  }
  return out;
}

// completed tasks from /stats merged into the page's list: keyed by server instance and id (an id repeats after a
// restart), oldest first, at most keep
export function mergeTasks(list, tasks, instance, keep = 60) {
  const seen = new Set(list.map((t) => t.instance + ":" + t.id));
  const out = list.slice();
  for (const t of tasks || []) {
    const key = instance + ":" + t.id;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ ...t, instance });
  }
  out.sort((a, b) => (a.instance === b.instance ? a.t_end_us - b.t_end_us : (a.t_end_us || 0) - (b.t_end_us || 0)));
  return out.length > keep ? out.slice(out.length - keep) : out;
}

// a window of cumulative per-layer readings: the base is the oldest reading inside window_s; reset when the counters go
// back or the layer count changes (a restart, a new context)
class Window {
  constructor(windowS = 60) {
    this.windowS = windowS;
    this.q = [];
  }

  pushRow(t, row, back) {
    const last = this.q[this.q.length - 1];
    if (last && back(last.row, row)) this.q = [];
    this.q.push({ t, row });
    while (this.q.length > 2 && this.q[0].t < t - this.windowS) this.q.shift();
  }

  ends() {
    return this.q.length < 2 ? null : [this.q[0].row, this.q[this.q.length - 1].row];
  }
}

// expert-cache hit rate per layer over the window, from /stats moe_cache.layers
export class LayerWindow extends Window {
  push(t, hit, miss) {
    this.pushRow(t, { hit: hit.slice(), miss: miss.slice() }, (a, b) =>
      a.hit.length !== b.hit.length || b.hit.some((h, i) => h < a.hit[i]) || b.miss.some((m, i) => m < a.miss[i]));
  }

  rates() {
    const e = this.ends();
    if (!e) return [];
    const [a, b] = e;
    return b.hit.map((h, i) => ratio(h - a.hit[i], h - a.hit[i] + (b.miss[i] - a.miss[i])));
  }
}

// the critical link per generation decode over the window, from /stats "phase" (cumulative steps, step_us, cpu_us,
// cpu_us per layer): decode ms, CPU-expert ms, CPU-expert ms per layer
export class PhaseWindow extends Window {
  push(t, ph) {
    this.pushRow(t, { steps: ph.steps, step_us: ph.step_us, cpu_us: ph.cpu_us, layers: (ph.cpu_us_layer || []).slice(), il: (ph.il || []).slice() },
      (a, b) => b.steps < a.steps || b.step_us < a.step_us || b.cpu_us < a.cpu_us || a.layers.length !== b.layers.length);
  }

  view() {
    const e = this.ends();
    if (!e) return null;
    const [a, b] = e;
    const dn = b.steps - a.steps;
    if (dn <= 0) return null;
    return {
      steps: dn,
      step_ms: (b.step_us - a.step_us) / dn / 1e3,
      cpu_ms: (b.cpu_us - a.cpu_us) / dn / 1e3,
      layers_ms: b.layers.map((v, i) => (v - a.layers[i]) / dn / 1e3),
      il: b.il,
      window_s: this.windowS,
    };
  }
}

// GB/s of one PCIe lane by generation (after encoding), for the link's ceiling
const LANE_GBS = { 1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563 };

export function pcieCeilingGBs(gpu) {
  if (!gpu) return null;
  const gen = gpu.pcie_gen_max, w = gpu.pcie_width_max;
  return LANE_GBS[gen] && isNum(w) && w > 0 ? LANE_GBS[gen] * w : null;
}
