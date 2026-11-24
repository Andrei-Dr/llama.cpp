// the cards along the top: decode, prefill, the KV cache, the SSD prompt cache, then decode step, in flight, MTP acceptance

import { $, isNum, txt, num, pct, bytes, compact, figure, show } from "../util.js";
import { spark } from "../spark.js";
import { store, latest, lastNum, sum } from "../store.js";

export function render(st) {
  const d = latest(), now = store.now, S = store.samples, c = st.counters;

  const step = isNum(d.step_ms) ? d.step_ms : lastNum("step_ms");
  figure("c-step", num(step, 2), "ms");
  const dec = isNum(d.decode_ms) ? d.decode_ms : lastNum("decode_ms");
  txt("c-step-s", "target decode " + num(dec, 1) + " ms" + (isNum(d.tokens_per_step) ? " · " + num(d.tokens_per_step, 2) + " tok/step" : ""));
  spark($("sp-step"), S, ["step_ms"], now);

  figure("c-dec", num(d.decode_tps, 1), "tok/s");
  txt("c-dec-s", "generated " + compact(c.gen_tokens + sum(st.slots, (s) => s.n_gen)));
  spark($("sp-dec"), S, ["decode_tps"], now);

  // the rate while reading a prompt: the last interval that read one (between prompts there is none)
  const pf = isNum(d.prefill_tps) ? d.prefill_tps : lastNum("prefill_tps");
  figure("c-pf", isNum(pf) ? num(pf, 0) : "—", "tok/s");
  txt("c-pf-s", "cumulative " + num(c.prompt_us > 0 ? 1e6 * c.prompt_tokens / c.prompt_us : null, 0) + " tok/s over " + compact(c.prompt_tokens));
  spark($("sp-pf"), S, ["prefill_tput"], now);

  const fl = st.slots.filter((s) => s.state !== "idle").length;
  figure("c-fl", String(fl), "/ " + st.n_slots);
  txt("c-fl-s", (st.n_deferred || 0) + " waiting");
  spark($("sp-fl"), S, ["in_flight", "waiting"], now, { fixed: { lo: 0, hi: Math.max(1, st.n_slots) } });

  const kv = sum(st.slots, (s) => s.n_ctx), kvMax = sum(st.slots, (s) => s.n_ctx_max);
  figure("c-kv", compact(kv), "tok");
  txt("c-kv-s", "of " + compact(kvMax) + " · " + pct(kvMax ? kv / kvMax : null) + " full");
  const all = c.prompt_tokens + c.prompt_tokens_cached;
  txt("c-kv-s2", all ? "prompt reuse " + pct(c.prompt_tokens_cached / all) + " of " + compact(all) + " tok" : "");
  spark($("sp-kv"), S, ["kv_used"], now);

  const drafting = c.draft_tokens > 0 || sum(st.slots, (s) => s.n_draft) > 0;
  const cum = c.draft_tokens > 0 ? c.draft_accepted / c.draft_tokens : null;
  const acc = isNum(d.accept) ? d.accept : cum;
  figure("c-acc", drafting && acc !== null ? (100 * acc).toFixed(0) : "—", "%");
  txt("c-acc-s", drafting ? "cumulative " + pct(cum) : "no drafting");
  spark($("sp-acc"), S, ["accept"], now);
}

// the SSD prompt cache: hit rate of the lookups, what it restored, how full it is, restores and saves per second
export function renderSsd(st) {
  const dc = st.disk_cache;
  show("card-ssd", !!dc);
  if (!dc) return;
  const d = latest();
  figure("c-ssd", dc.find ? (100 * dc.find_hit / dc.find).toFixed(0) : "—", "%");
  const rate = isNum(d.disk_restore_mbs) && d.disk_restore_mbs > 0 ? " · now " + num(d.disk_restore_mbs, 1) + " MB/s" : "";
  txt("c-ssd-s", dc.restore ? "restored " + compact(dc.restore_tokens) + " tok · " + bytes(dc.restore_bytes) + rate : "nothing restored yet");
  txt("c-ssd-s2", bytes(dc.bytes) + (dc.limit_bytes ? " / " + bytes(dc.limit_bytes) : "") + " · " + num(dc.entries) + " entries");
  spark($("sp-ssd"), store.samples, ["disk_restore_mbs", "disk_save_mbs"], store.now);
}
