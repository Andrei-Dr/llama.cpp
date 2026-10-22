// the machine (GPU busy and idle while decoding, CPU, the server's CPU, power, RAM, the cost of the statistics) and the
// critical link: CPU-expert phase vs everything else per generation decode, per layer (when the server reports it)

import { $, isNum, txt, num, pct, bytes, figure, width, show } from "../util.js";
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
  txt("k-cpu", isNum(lastNum("cpu_pct")) ? num(lastNum("cpu_pct"), 0) + "%" : "—");
  const srv = lastNum("server_cpu_pct");
  txt("k-srv", isNum(srv) ? num(srv, 0) + "% of one core" : "—");
  txt("k-io", isNum(lastNum("iowait_pct")) ? num(lastNum("iowait_pct"), 1) + "%" : "—");
  txt("k-pwr", num(lastNum("gpu_power_w"), 0) + " W · " + num(lastNum("gpu_temp_c")) + " °C · " + num(lastNum("gpu_clock_mhz")) + " MHz");
  const avail = lastNum("mem_avail"), swap = lastNum("swap_used");
  txt("k-ram", isNum(avail) ? bytes(avail) + " · " + bytes(swap) : "—");
  const mp = st.moe_cache ? st.moe_cache.publish : null, sp = st.machine ? st.machine.sampler : null;
  txt("k-pub", avgUs(st.publish) + " · " + avgUs(mp) + " · " + avgUs(sp));
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
  $("p-machine").classList.toggle("alone", !has);
  if (!has) return;
  const cpu = p.cpu_ms, other = Math.max(0, p.step_ms - cpu);
  figure("ph-cpu", num(cpu, 2), "ms");
  figure("ph-dev", num(other, 2), "ms");
  width("ph-b-cpu", p.step_ms ? cpu / p.step_ms : 0);
  width("ph-b-dev", p.step_ms ? other / p.step_ms : 0);
  txt("ph-meta", "decode " + num(p.step_ms, 2) + " ms · CPU experts " + pct(p.step_ms ? cpu / p.step_ms : null, 0) + " · " + p.steps + " decodes");
  const L = p.layers_ms || [], il = p.il || [];
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
