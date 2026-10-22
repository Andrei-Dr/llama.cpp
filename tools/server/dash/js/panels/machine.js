// the machine (GPU busy and idle while decoding, CPU, the server's CPU, power, RAM, the cost of the statistics) and the
// critical link: CPU-expert phase vs everything else per generation decode, per layer (when the server reports it)

import { $, isNum, txt, num, pct, bytes, figure, width, show, val } from "../util.js";
import { spark } from "../spark.js";
import { store, latest, lastNum, windowOf, sum } from "../store.js";

const avgUs = (t) => (t && t.n ? num(t.sum_us / t.n, 1) + " µs" : "—");

export function renderMachine(st) {
  const d = latest();
  const sm = lastNum("gpu_sm");
  figure("k-gpu", num(sm, 0), "%");
  const dec = windowOf(60).filter((s) => s.decode_tps > 0 && isNum(s.gpu_sm));
  const idle = dec.length ? 1 - sum(dec, (s) => s.gpu_sm) / dec.length / 100 : null;
  figure("k-idle", isNum(idle) ? (100 * idle).toFixed(0) : "—", "%");
  spark($("sp-gpu"), store.samples, ["gpu_sm", "cpu_pct"], store.now, { fixed: { lo: 0, hi: 100 } });
  const cpu = lastNum("cpu_pct"), io = lastNum("iowait_pct");
  val("k-cpu", isNum(cpu) ? num(cpu, 0) + "%" : null);
  const srv = lastNum("server_cpu_pct");
  val("k-srv", isNum(srv) ? [num(srv / 100, 2), "cores"] : null);
  val("k-io", isNum(io) ? num(io, 1) + "%" : null);
  const w = lastNum("gpu_power_w"), tc = lastNum("gpu_temp_c"), clk = lastNum("gpu_clock_mhz");
  val("k-pwr", isNum(w) ? [num(w, 0), "W"] : null, isNum(tc) ? num(tc) + " °C" : null, isNum(clk) ? num(clk) + " MHz" : null);
  const avail = lastNum("mem_avail"), swap = lastNum("swap_used");
  val("k-ram", isNum(avail) ? bytes(avail) : null, isNum(swap) ? "swap " + bytes(swap) : null);
  const mp = st.moe_cache ? st.moe_cache.publish : null, sp = st.machine ? st.machine.sampler : null;
  val("k-pub", st.publish && st.publish.n ? avgUs(st.publish) : null, mp && mp.n ? avgUs(mp) : null, sp && sp.n ? avgUs(sp) : null);
  show("k-plot-empty", !(isNum(sm) || isNum(cpu)));
  const nvml = st.machine ? st.machine.nvml : null;
  $("k-meta").title = nvml && nvml !== "ok" ? nvml : "";
  txt("k-meta", nvml && nvml !== "ok" ? (nvml === "off" ? "GPU sampling off" : "no GPU data (NVML not loaded)")
    : isNum(idle) && idle > 0.5 && d.decode_tps > 0 ? "GPU waits on the host" : "");
}

// the 60 s window over /stats "phase" (derive.js PhaseWindow): per generation decode
export function renderPhase() {
  const p = store.phase.view();
  const has = !!(p && isNum(p.step_ms));
  show("p-phase", has);
  if (!has) return;
  const cpu = p.cpu_ms, other = Math.max(0, p.step_ms - cpu);
  figure("ph-cpu", num(cpu, 2), "ms");
  figure("ph-dev", num(other, 2), "ms");
  width("ph-b-cpu", p.step_ms ? cpu / p.step_ms : 0);
  width("ph-b-dev", p.step_ms ? other / p.step_ms : 0);
  txt("ph-meta", "decode " + num(p.step_ms, 2) + " ms · CPU experts " + pct(p.step_ms ? cpu / p.step_ms : null, 0) + " · " + p.steps + " decodes");
  const L = p.layers_ms || [], il = p.il || [];
  show("ph-layer-box", L.length > 0);
  const bars = $("ph-layers");
  if (bars.children.length !== L.length) bars.replaceChildren(...L.map(() => document.createElement("i")));
  const vmax = Math.max(1e-9, ...L.filter(isNum));
  let worst = -1;
  L.forEach((v, i) => { if (isNum(v) && (worst < 0 || v > L[worst])) worst = i; });
  L.forEach((v, i) => {
    bars.children[i].style.height = (isNum(v) ? Math.max(2, (100 * v) / vmax) : 2).toFixed(1) + "%";
    bars.children[i].className = i === worst ? "worst" : "";
    bars.children[i].title = "L" + il[i] + ": " + num(v, 3) + " ms";
  });
  const mean = L.length ? L.reduce((a, b) => a + b, 0) / L.length : null;
  txt("ph-layer-meta", num(mean, 3) + " ms mean" + (worst >= 0 ? " · slowest L" + il[worst] + " " + num(L[worst], 3) + " ms" : ""));
  txt("ph-l0", il.length ? "L" + il[0] : "");
  txt("ph-ln", il.length ? "L" + il[il.length - 1] : "");
}
