// memory: VRAM (NVML, all GPUs), the KV / context fill, prompt reuse, and the prompt disk cache as the SSD tier

import { isNum, txt, num, pct, bytes, dur, compact, width, show } from "../util.js";
import { sum } from "../store.js";

export function render(st) {
  const gpus = (st.machine && st.machine.gpus) || [];
  // a GPU whose query failed makes the sum unknown, not smaller
  const gpuSum = (k) => (gpus.length && gpus.every((g) => isNum(g[k])) ? sum(gpus, (g) => g[k]) : null);
  const used = gpuSum("vram_used"), total = gpuSum("vram_total");
  txt("m-vram", isNum(used) && total > 0 ? bytes(used) + " / " + bytes(total) + " · " + pct(used / total) : "no GPU data");
  width("m-vram-bar", isNum(used) && total > 0 ? used / total : 0);
  txt("mem-meta", isNum(used) && total > 0 ? "GPU " + pct(used / total, 0) : "—");

  const kv = sum(st.slots, (s) => s.n_ctx), kvMax = sum(st.slots, (s) => s.n_ctx_max);
  txt("m-kv", compact(kv) + " / " + compact(kvMax) + " tok · " + pct(kvMax ? kv / kvMax : null));
  width("m-kv-bar", kvMax ? kv / kvMax : 0);

  const c = st.counters, all = c.prompt_tokens + c.prompt_tokens_cached;
  txt("m-prefix", pct(all ? c.prompt_tokens_cached / all : null) + " of " + compact(all));
  width("m-prefix-bar", all ? c.prompt_tokens_cached / all : 0);

  const dc = st.disk_cache;
  show("ssd", !!dc);
  if (!dc) return;
  txt("d-bytes", bytes(dc.bytes) + (dc.limit_bytes ? " / " + bytes(dc.limit_bytes) : ""));
  width("d-bar", dc.limit_bytes ? dc.bytes / dc.limit_bytes : 0);
  txt("d-entries", num(dc.entries));
  txt("d-find", num(dc.find) + " · " + num(dc.find_hit) + (dc.find ? " (" + pct(dc.find_hit / dc.find, 0) + ")" : "") +
    (dc.f_sim_hit >= 0 ? " · last sim " + dc.f_sim_hit.toFixed(2) + " keep " + dc.f_keep_hit.toFixed(2) : ""));
  txt("d-restore", num(dc.restore) + " · " + compact(dc.restore_tokens) + " tok · " + bytes(dc.restore_bytes) +
    (dc.restore ? " · " + dur(dc.restore_us / 1000 / dc.restore) + " avg" : "") + (dc.restore_failed ? " · " + dc.restore_failed + " failed" : ""));
  txt("d-save", num(dc.save) + " · " + bytes(dc.save_bytes) + (dc.save ? " · " + dur(dc.save_us / 1000 / dc.save) + " avg" : ""));
  txt("d-nosave", [dc.save_skipped, dc.save_refused, dc.save_cancelled, dc.save_failed].map((v) => num(v)).join(" · "));
  txt("d-evict", [dc.evict_expired, dc.evict_space, dc.superseded].map((v) => num(v)).join(" · ") + " · " + bytes(dc.evict_bytes));
}
