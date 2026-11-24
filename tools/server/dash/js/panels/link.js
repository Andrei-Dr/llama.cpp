// interconnect: host -> GPU traffic (NVML) against the link's ceiling, split into expert-cache fills (and prompt warm-up
// uploads, when the cache reports them) and the rest; expert moves a second; the expert-cache counters

import { $, isNum, txt, num, pct, bytes, figure, width, show, val, known } from "../util.js";
import { spark } from "../spark.js";
import { pcieCeilingGBs } from "../derive.js";
import { store, latest, lastNum } from "../store.js";

// a counter's increase over the newest span seconds of history rows
function rowDelta(key, span) {
  const rows = store.rows;
  if (rows.length < 2) return null;
  const last = rows[rows.length - 1];
  let i = rows.length - 1;
  while (i > 0 && rows[i - 1].t_us >= last.t_us - span * 1e6 && rows[i - 1].t_start_us === last.t_start_us) i--;
  const a = rows[i][key], b = last[key];
  return isNum(a) && isNum(b) && b >= a ? b - a : null;
}

// "gen3 x16" when both are known
const linkText = (gen, w) => (isNum(gen) && isNum(w) ? "gen" + gen + " x" + w : null);

export function render(st) {
  const d = latest(), m = st.moe_cache;
  const gpus = (st.machine && st.machine.gpus) || [];
  show("p-link", gpus.length > 0 || !!m);
  show("l-bar", !!m);
  show("l-moves-grp", !!m);
  // the ceiling of every link together (pcie_rx is summed over the GPUs too)
  const ceils = gpus.map(pcieCeilingGBs);
  const ceilGBs = ceils.length && ceils.every(isNum) ? ceils.reduce((a, b) => a + b, 0) : null;
  const g = gpus[0], most = g && linkText(g.pcie_gen_max, g.pcie_width_max), now = g && linkText(g.pcie_gen, g.pcie_width);
  txt("link-meta", isNum(ceilGBs) ? "PCIe " + most + (gpus.length > 1 ? " x " + gpus.length + " GPUs" : "") + " · " + ceilGBs.toFixed(1) + " GB/s" +
    (now && now !== most ? " · now " + now : "") : gpus.length ? "PCIe link unknown" : "no GPU data");

  // NVML's PCIe counter is a 20 ms window: shown on its own, never mixed into the 1 s cache averages
  const rx = lastNum("pcie_rx_mbs"), tx = lastNum("pcie_tx_mbs");
  const up = isNum(d.moe_upload_mbs) ? d.moe_upload_mbs : 0, warm = isNum(d.moe_warm_mbs) ? d.moe_warm_mbs : 0;
  const ceil = isNum(ceilGBs) ? ceilGBs * 1000 : Math.max(1, up), fills = Math.max(0, up - warm);

  figure("l-rx", isNum(rx) ? (rx / 1000).toFixed(2) : "—", "GB/s");
  figure("l-moves", num(isNum(d.moe_upload_s) ? d.moe_upload_s : null, 1), "/s");
  width("l-b-up", fills / ceil);
  width("l-b-warm", warm / ceil);
  val("l-up", m ? [num(fills, 1), "MB/s"] : null);
  val("l-warm", m && isNum(d.moe_warm_mbs) ? [num(warm, 1), "MB/s"] : null);
  val("l-other", isNum(rx) ? [num(rx, 0), "MB/s"] : null);
  val("l-tx", isNum(tx) ? [num(tx, 0), "MB/s"] : null);

  const ups = rowDelta("moe_upload", 60), cpus = rowDelta("moe_miss_cpu", 60);
  txt("l-mv-meta", known(isNum(ups) ? num(ups, 0) + " uploaded" : null, isNum(cpus) ? num(cpus, 0) + " on the CPU" : null) || "");
  spark($("sp-moves"), store.samples, ["moe_upload_s", "moe_evict_s"], store.now, { span: 60 });

  show("x-counters", !!m);
  if (!m) return;
  const c = m.counters, lookups = c.hit + c.miss;
  const split = (k) => (k && k.hit + k.miss ? [num(k.hit + k.miss), pct(k.hit / (k.hit + k.miss)) + " hit"] : [null, null]);
  val("x-miss", num(c.miss), lookups ? pct(c.miss / lookups) + " of lookups" : null);
  val("x-up", num(c.upload), bytes(c.upload_bytes));
  val("x-cpu", num(c.miss_cpu), c.miss ? pct(c.miss_cpu / c.miss, 0) + " of misses" : null);
  val("x-small", ...split(m.ubatch_small));
  val("x-large", ...split(m.ubatch_large));
  val("x-copy", c.copy ? num(c.copy) : null, bytes(c.copy_bytes));
  val("x-dev", m.suspended ? "released: prefill mode" : bytes(m.device_bytes), num(m.n_resident) + " of " + num(m.n_slots) + " slots filled");
}
