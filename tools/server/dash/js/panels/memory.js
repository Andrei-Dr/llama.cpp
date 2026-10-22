// memory: VRAM (NVML, all GPUs), the KV / context fill, prompt reuse, and the prompt disk cache as the SSD tier

import { isNum, txt, num, pct, bytes, dur, compact, width, show, val } from "../util.js";
import { sum } from "../store.js";

export function render(st) {
  const gpus = (st.machine && st.machine.gpus) || [];
  // a GPU whose query failed makes the sum unknown, not smaller
  const gpuSum = (k) => (gpus.length && gpus.every((g) => isNum(g[k])) ? sum(gpus, (g) => g[k]) : null);
  const used = gpuSum("vram_used"), total = gpuSum("vram_total");
  const haveVram = isNum(used) && total > 0;
  val("m-vram", haveVram ? bytes(used) : null, haveVram ? "/ " + bytes(total) + " · " + pct(used / total) : null);
  width("m-vram-bar", haveVram ? used / total : 0);
  txt("mem-meta", haveVram ? "GPU " + pct(used / total, 0) : "");

  const kv = sum(st.slots, (s) => s.n_ctx), kvMax = sum(st.slots, (s) => s.n_ctx_max);
  val("m-kv", [compact(kv), "tok"], "/ " + compact(kvMax) + " · " + pct(kvMax ? kv / kvMax : null));
  width("m-kv-bar", kvMax ? kv / kvMax : 0);

  const c = st.counters, all = c.prompt_tokens + c.prompt_tokens_cached;
  val("m-prefix", pct(all ? c.prompt_tokens_cached / all : null), "of " + compact(all) + " tok");
  width("m-prefix-bar", all ? c.prompt_tokens_cached / all : 0);

  const dc = st.disk_cache;
  show("ssd", !!dc);
  if (!dc) return;
  val("d-bytes", bytes(dc.bytes), dc.limit_bytes ? "/ " + bytes(dc.limit_bytes) : null);
  width("d-bar", dc.limit_bytes ? dc.bytes / dc.limit_bytes : 0);
  val("d-entries", num(dc.entries));
  val("d-find", num(dc.find_hit) + (dc.find ? " of " + num(dc.find) : ""), dc.find ? pct(dc.find_hit / dc.find, 0) : null,
    dc.f_sim_hit >= 0 ? "last sim " + dc.f_sim_hit.toFixed(2) + " keep " + dc.f_keep_hit.toFixed(2) : null);
  val("d-restore", num(dc.restore), compact(dc.restore_tokens) + " tok · " + bytes(dc.restore_bytes),
    dc.restore ? dur(dc.restore_us / 1000 / dc.restore) + " avg" : null, dc.restore_failed ? dc.restore_failed + " failed" : null);
  val("d-save", num(dc.save), bytes(dc.save_bytes), dc.save ? dur(dc.save_us / 1000 / dc.save) + " avg" : null);
  val("d-nosave", [dc.save_skipped, dc.save_refused, dc.save_cancelled, dc.save_failed].map((v) => num(v)).join(" · "));
  val("d-evict", [dc.evict_expired, dc.evict_space, dc.superseded].map((v) => num(v)).join(" · "), bytes(dc.evict_bytes));
}
