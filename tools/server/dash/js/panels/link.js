// interconnect: host -> GPU traffic (NVML) against the link's ceiling, split into expert-cache fills, prompt warm-up
// uploads and the rest; expert moves a second; the expert-cache counters

import { $, isNum, txt, num, pct, bytes, figure, width, show } from "../util.js";
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
  txt("l-up", num(fills, 1) + " MB/s");
  txt("l-warm", num(warm, 1) + " MB/s");
  txt("l-other", isNum(rx) ? num(rx, 0) + " MB/s" : "—");
  txt("l-tx", isNum(tx) ? num(tx, 0) + " MB/s" : "—");

  const ups = rowDelta("moe_upload", 60), evs = rowDelta("moe_evict", 60);
  txt("l-mv-meta", num(ups, 0) + " up · " + num(evs, 0) + " evicted");
  spark($("sp-moves"), store.samples, ["moe_upload_s", "moe_evict_s"], store.now, { span: 60 });

  show("x-counters", !!m);
  if (!m) return;
  const c = m.counters;
  txt("x-up", num(c.upload) + " · " + bytes(c.upload_bytes) + (c.upload_warm ? " · warm-up " + num(c.upload_warm) + " · " + bytes(c.upload_bytes_warm) : ""));
  txt("x-ev", num(c.evict));
  txt("x-readmit", num(c.readmit) + " · " + pct(c.upload ? c.readmit / c.upload : null, 0) + " of uploads");
  txt("x-refused", num(c.refused) + " · " + pct(c.miss ? c.refused / c.miss : null, 0) + " of misses");
  txt("x-noslot", num(c.no_slot));
  txt("x-inflight", num(c.miss_in_flight));
  txt("x-susp", num(c.suspend) + " · resumed " + num(c.resume) + (c.resume_partial ? " (" + c.resume_partial + " partial)" : ""));
  txt("x-borrow", m.suspended ? bytes(c.bytes_released) + " released now"
    : m.n_live < m.n_slots ? (m.n_slots - m.n_live) + " slots/layer short · " + bytes((m.n_slots - m.n_live) * m.slot_bytes)
    : "none · last release " + bytes(c.bytes_released));
}
