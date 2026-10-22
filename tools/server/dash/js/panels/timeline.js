// the timeline panel: the charts (chart.js) over the server's history rows, 15 minutes or 24 hours

import { $, txt, num } from "../util.js";
import { Chart } from "../chart.js";
import { store } from "../store.js";

const GiB = 1 / 1073741824;
const f0 = (v) => num(v, 0);
const f1 = (v) => num(v, v < 10 ? 1 : 0);
const f2 = (v) => num(v, v < 1 ? 2 : v < 10 ? 1 : 0);

const hasGpu = (st) => !!(st && st.machine && st.machine.gpus && st.machine.gpus.length);
const hasMoe = (st) => !!(st && st.moe_cache);
const hasHost = (st) => !!(st && st.machine && st.machine.host && st.machine.host.host_total_us !== null);

const hasDisk = (st) => !!(st && st.disk_cache);

// serving first (what the clients feel, the KV cache and both prompt-cache tiers), then the expert cache, the GPU and the host
const SPECS = [
  { cap: "decode", unit: "tok/s", fmt: f1, series: [{ key: "decode_tps", cls: "fg" }] },
  { cap: "prefill", unit: "tok/s", fmt: f0, series: [{ key: "prefill_tps", cls: "cool" }] },
  { cap: "kv cache · context fill", unit: "%", fmt: f0, hi: 100, names: "tokens held / capacity", series: [{ key: "kv_fill_pct", cls: "fg" }] },
  { cap: "prompt reuse · slot hits", unit: "%", fmt: f0, hi: 100, names: "prompt tokens not processed again (slot or SSD), per interval",
    series: [{ key: "prompt_reuse", cls: "ok", k: 100 }] },
  { cap: "ssd restores · saves", unit: "MB/s", fmt: f1, show: hasDisk, names: "restored from disk · saved to disk",
    series: [{ key: "disk_restore_mbs", cls: "cool", name: "restores" }, { key: "disk_save_mbs", cls: "pink", name: "saves" }] },
  { cap: "ssd restore speed", unit: "tok/s", fmt: f0, show: hasDisk, names: "restored · prefill, tokens per second of work, since start",
    series: [{ key: "disk_restore_tps_avg", cls: "cool", name: "restored" }, { key: "prefill_tps_avg", cls: "fg", name: "prefill" }] },
  { cap: "ssd time saved", unit: "s", fmt: f1, show: hasDisk, names: "prefill time the reused restores replaced, less the restores, since start",
    series: [{ key: "disk_saved_s", cls: "ok" }] },
  { cap: "ssd lookup hit rate", unit: "%", fmt: f0, hi: 100, show: hasDisk, names: "disk-cache lookups that found an entry, per interval",
    series: [{ key: "disk_hit_rate", cls: "cool", k: 100 }] },
  { cap: "ssd cache size", unit: "GiB", fmt: f2, show: hasDisk, series: [{ key: "disk_bytes", cls: "cool", k: GiB }] },
  { cap: "requests", unit: "", fmt: f0, names: "in flight · waiting for a slot",
    series: [{ key: "in_flight", cls: "ok", name: "in flight" }, { key: "waiting", cls: "warn", name: "waiting" }] },
  { cap: "decode step", unit: "ms", fmt: f1, names: "step · CPU experts per decode",
    series: [{ key: "step_ms", cls: "fg", name: "step" }, { key: "cpu_ms", cls: "warn", name: "CPU experts" }] },
  { cap: "draft acceptance", unit: "%", fmt: f0, hi: 100, show: (st) => st && st.counters && st.counters.draft_tokens > 0,
    series: [{ key: "accept", cls: "ok", k: 100 }] },
  { cap: "expert hit rate", unit: "%", fmt: f0, hi: 100, show: hasMoe, series: [{ key: "moe_hit_rate", cls: "fg", k: 100 }] },
  { cap: "expert moves", unit: "/s", fmt: f0, show: hasMoe, names: "uploads · evictions",
    series: [{ key: "moe_upload_s", cls: "fg", name: "uploads" }, { key: "moe_evict_s", cls: "warn", name: "evictions" }] },
  { cap: "host → GPU", unit: "GB/s", fmt: f2, show: (st) => hasGpu(st) || hasMoe(st), names: "PCIe (20 ms samples) · cache fills (1 s)",
    series: [{ key: "pcie_rx_mbs", cls: "fg", k: 1e-3, name: "PCIe" }, { key: "moe_fill_mbs", cls: "ok", k: 1e-3, name: "cache fills" }] },
  { cap: "GPU busy", unit: "%", fmt: f0, hi: 100, show: hasGpu, names: "SM · memory",
    series: [{ key: "gpu_sm", cls: "fg", name: "SM" }, { key: "gpu_mem_bw", cls: "warn", name: "memory" }] },
  { cap: "VRAM", unit: "GiB", fmt: f2, show: hasGpu, series: [{ key: "vram_used", cls: "fg", k: GiB }] },
  { cap: "GPU power", unit: "W", fmt: f0, show: hasGpu, series: [{ key: "gpu_power_w", cls: "fg" }] },
  { cap: "GPU temperature", unit: "°C", fmt: f0, show: hasGpu, series: [{ key: "gpu_temp_c", cls: "warn" }] },
  { cap: "SM clock", unit: "MHz", fmt: f0, show: hasGpu, series: [{ key: "gpu_clock_mhz", cls: "fg" }] },
  { cap: "CPU", unit: "%", fmt: f0, hi: 100, show: hasHost, names: "all cores · this server's share",
    series: [{ key: "cpu_pct", cls: "fg", name: "all cores" }, { key: "server_share_pct", cls: "warn", name: "server" }] },
  { cap: "host memory", unit: "GiB", fmt: f1, show: hasHost, names: "available · swap used",
    series: [{ key: "mem_avail", cls: "fg", k: GiB, name: "available" }, { key: "swap_used", cls: "warn", k: GiB, name: "swap" }] },
];

let charts = null;
let range = "15m";

function build() {
  const box = $("timeline");
  charts = SPECS.map((s) => new Chart(box, s, (t) => {
    store.hoverT = t;
    render();
  }));
  for (const [id, r] of [["tl-15m", "15m"], ["tl-24h", "24h"]]) {
    $(id).addEventListener("click", () => {
      range = r;
      $("tl-15m").classList.toggle("on", r === "15m");
      $("tl-24h").classList.toggle("on", r === "24h");
      render();
    });
  }
}

// a window's length for its button: "15 min", "24 h", "90 s"
const spanText = (s) => (s >= 7200 ? +(s / 3600).toFixed(1) + " h" : s >= 120 ? +(s / 60).toFixed(1) + " min" : Math.round(s) + " s");

export function render() {
  if (!charts) build();
  txt("tl-15m", spanText(store.spanS));
  txt("tl-24h", spanText(store.minuteSpanS));
  const t1 = store.now;
  const minute = range === "24h";
  const samples = minute ? store.minuteSamples : store.samples;
  const span = minute ? store.minuteSpanS : store.spanS;
  // a gap: three periods without a row (the server was down or not sampling)
  const gapS = 3 * (minute ? store.minutePeriodS : store.periodS);
  for (const c of charts) c.render(samples, t1 - span, t1, gapS, store.hoverT, store.stats);
  txt("tl-meta", minute ? store.minuteSamples.length + " minutes" : store.samples.length + " samples");
}
