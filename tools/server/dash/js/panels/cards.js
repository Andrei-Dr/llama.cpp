// the six cards along the top: decode step, decode, prefill, in flight, kv / context fill, MTP acceptance

import { $, isNum, txt, num, pct, compact, figure } from "../util.js";
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
  txt("c-kv-s", "of " + compact(kvMax) + " · " + pct(kvMax ? kv / kvMax : null));
  spark($("sp-kv"), S, ["kv_used"], now);

  const drafting = c.draft_tokens > 0 || sum(st.slots, (s) => s.n_draft) > 0;
  const cum = c.draft_tokens > 0 ? c.draft_accepted / c.draft_tokens : null;
  const acc = isNum(d.accept) ? d.accept : cum;
  figure("c-acc", drafting && acc !== null ? (100 * acc).toFixed(0) : "—", "%");
  txt("c-acc-s", drafting ? "cumulative " + pct(cum) : "no drafting");
  spark($("sp-acc"), S, ["accept"], now);
}
