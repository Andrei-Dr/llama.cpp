// the requests table: live slots first, then completed tasks (the fields print_timings, release and the slot selection log)

import { $, isNum, txt, num, pct, dur, compact, el } from "../util.js";
import { store } from "../store.js";

function stateCell(tr, state) {
  tr.insertCell().appendChild(el("span", "st " + state, state));
}

function kvCell(tr, used, max) {
  const td = tr.insertCell(), b = el("span", "kvcell"), i = el("i");
  b.appendChild(i);
  i.style.width = (max ? Math.min(100, (100 * used) / max) : 0).toFixed(1) + "%";
  td.append(b, document.createTextNode(max ? pct(used / max) : "—"));
}

// the prompt size, then how much of it the slot already held (muted)
function promptCell(tr, n, cached) {
  const td = tr.insertCell();
  td.append(document.createTextNode(compact(n)));
  if (cached) td.appendChild(el("small", "", compact(cached) + " hit"));
}

// how the slot was chosen and how much of the prompt it already held
function prefixText(sel) {
  if (!sel || !sel.how) return "—";
  // how · f_sim/f_keep (share of the request, of the slot's old prompt) · a disk restore
  let s = sel.how;
  const f = (v) => v.toFixed(2).replace(/^0/, "");
  if (isNum(sel.f_sim) && sel.f_sim >= 0) s += " " + f(sel.f_sim) + "/" + (isNum(sel.f_keep) && sel.f_keep >= 0 ? f(sel.f_keep) : "–");
  if (sel.disk_restored) s += " ssd " + compact(sel.n_disk_tokens);
  return s;
}

export function render(st) {
  const frag = document.createDocumentFragment(), now = store.now;
  const live = st.slots.filter((s) => s.state !== "idle");
  for (const s of live) {
    const tr = el("tr");
    tr.insertCell().textContent = "#" + s.id_task;
    stateCell(tr, s.state);
    tr.insertCell().textContent = s.id;
    promptCell(tr, s.n_prompt, s.n_prompt_cached);
    tr.insertCell().textContent = num(s.n_gen);
    tr.insertCell().textContent = compact(s.n_ctx);
    kvCell(tr, s.n_ctx, s.n_ctx_max);
    tr.insertCell().textContent = s.n_draft ? pct(s.n_draft_accepted / s.n_draft, 0) : "—";
    tr.insertCell().textContent = dur(s.t_queue_ms);
    tr.insertCell().textContent = s.state === "generating" ? dur(s.t_queue_ms + s.t_prompt_ms) : "—";
    tr.insertCell().textContent = s.n_gen > 1 && s.t_gen_ms > 0 ? num((s.n_gen - 1) / (s.t_gen_ms / 1000), 1) : "—";
    tr.insertCell().textContent = "—";
    tr.insertCell().textContent = dur(s.t_age_ms);
    frag.appendChild(tr);
  }
  for (const t of store.tasks.slice().reverse()) {
    const tr = el("tr", "done");
    tr.insertCell().textContent = "#" + t.id;
    stateCell(tr, t.stop || "none");
    tr.insertCell().textContent = t.id_slot;
    promptCell(tr, t.n_prompt, t.n_prompt_cached);
    tr.insertCell().textContent = num(t.n_gen) + (t.truncated ? " ✂" : "");
    tr.insertCell().textContent = compact(t.n_ctx);
    kvCell(tr, t.n_ctx, (st.slots[t.id_slot] || {}).n_ctx_max || 0);
    tr.insertCell().textContent = t.n_draft ? pct(t.accept_rate, 0) : "—";
    tr.insertCell().textContent = dur(t.t_queue_ms);
    tr.insertCell().textContent = dur(t.ttft_ms);
    tr.insertCell().textContent = isNum(t.gen_ms_per_token) && t.gen_ms_per_token > 0 ? num(1000 / t.gen_ms_per_token, 1) : "—";
    tr.insertCell().textContent = prefixText(t.sel);
    tr.insertCell().textContent = isNum(t.t_end_us) && t.instance === store.instance ? dur(1000 * (now - t.t_end_us / 1e6)) : "—";
    frag.appendChild(tr);
  }
  $("req-body").replaceChildren(frag);
  $("req-empty").hidden = frag.childNodes.length > 0;
  const dec = live.filter((s) => s.state === "generating").length;
  txt("req-meta", dec + " decoding · " + (live.length - dec) + " prefill · " + (st.n_deferred || 0) + " waiting · " + store.tasks.length + " shown");
}
