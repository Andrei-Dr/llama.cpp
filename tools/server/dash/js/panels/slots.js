// slot state, draft acceptance by position and the speculation panel

import { $, isNum, txt, num, pct, compact, figure, show, sized, el } from "../util.js";
import { latest } from "../store.js";

export function renderSlots(st) {
  const sq = sized($("slotsq"), st.n_slots);
  st.slots.forEach((s, i) => {
    if (!sq[i]) return;
    sq[i].className = s.state;
    sq[i].title = "slot " + s.id + ": " + s.state;
  });
  const c = st.counters, sel = st.selection || {};
  txt("slots-meta", st.slots.filter((s) => s.state !== "idle").length + " / " + st.n_slots + " busy");
  txt("s-wait", num(st.n_deferred || 0));
  txt("s-req", num(c.tasks));
  txt("s-steps", num(c.n_decode));
  txt("s-trunc", num(c.truncated));
  txt("s-sel", ["lcp", "lru", "id"].map((k) => k + " " + num(sel[k] || 0)).join(" · ") + (sel.disk_restored ? " · ssd " + num(sel.disk_restored) : ""));
  txt("s-stop", Object.entries(c.stop || {}).map(([k, v]) => k + " " + num(v)).join(" · ") || "—");
}

export function renderSpec(st) {
  const c = st.counters, acc = c.accepted_per_pos || [], has = c.draft_tokens > 0;
  show("p-accpos", has);
  show("p-spec", has);
  if (!has) return;

  const bars = $("accpos-bars"), labels = $("accpos-labels");
  if (bars.children.length !== acc.length) {
    bars.replaceChildren(...acc.map(() => { const b = el("div", "b"); b.appendChild(el("span")); return b; }));
    labels.replaceChildren(...acc.map(() => el("div")));
  }
  acc.forEach((a, j) => {
    // conditional: of the verification steps that reached position j, the share that accepted it
    const reach = j === 0 ? c.draft_verif_steps : acc[j - 1], r = reach > 0 ? a / reach : 0;
    bars.children[j].style.height = (100 * r).toFixed(1) + "%";
    bars.children[j].firstChild.textContent = (100 * r).toFixed(0) + "%";
    labels.children[j].textContent = j + 1 + " · reach " + compact(reach);
  });

  const d = latest();
  const perStep = c.draft_verif_steps > 0 ? 1 + c.draft_accepted / c.draft_verif_steps : null;
  figure("spec-tps", num(isNum(d.tokens_per_step) ? d.tokens_per_step : perStep, 2), "tok");
  figure("spec-acc", (100 * c.draft_accepted / c.draft_tokens).toFixed(0), "%");
  txt("spec-meta", "depth " + acc.length + " · cumulative " + num(perStep, 2) + " tok/step");
  txt("spec-drafted", num(c.draft_tokens));
  txt("spec-accepted", num(c.draft_accepted) + " · " + pct(c.draft_accepted / c.draft_tokens));
  txt("spec-verif", num(c.draft_verif_steps));
  const g = st.decode.gen;
  txt("spec-dec", g.n ? num(g.sum_us / g.n / 1000, 2) + " ms (ema " + num(g.ema_us / 1000, 2) + ")" : "—");
}
