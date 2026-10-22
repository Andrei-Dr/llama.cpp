// node --test tools/server/tests/dash/: the dashboard's arithmetic (tools/server/dash/js/derive.js) on row sequences
import { test } from "node:test";
import assert from "node:assert/strict";

import { rowsOf, derive, deriveAll, mergeTasks, LayerWindow, PhaseWindow, pcieCeilingGBs } from "../../dash/js/derive.js";

function row(over = {}) {
  return {
    t_us: 10e6, unix_ms: 1, t_start_us: 5e6, snap_t_us: 10e6, gen_tokens: 0, draft_tokens: 0, draft_accepted: 0,
    prompt_tokens: 0, prompt_us: 0, prompt_cached: 0, tasks: 0, n_decode: 0, step_n: 0, step_us: 0, decode_n: 0,
    decode_us: 0, phase_graphs: 0, phase_cpu_us: 0, publish_n: 0, publish_us: 0, n_busy: 0, n_deferred: 0, kv_used: 0,
    kv_max: 4096, moe_t_us: null, moe_hit: null, moe_miss: null, gpu_util: null, pcie_rx: null, proc_cpu_us: 0,
    host_busy_us: null, host_total_us: null, host_iowait_us: null,
    ...over,
  };
}

const nums = (s) => Object.entries(s).filter(([, v]) => typeof v === "number");

test("rowsOf turns columns into rows", () => {
  const rows = rowsOf({ cols: { t_us: [1, 2], gen_tokens: [3, null] } });
  assert.deepEqual(rows, [{ t_us: 1, gen_tokens: 3 }, { t_us: 2, gen_tokens: null }]);
  assert.deepEqual(rowsOf({}), []);
});

test("decode rate over the snapshot clock, per-step figures from counters", () => {
  const p = row();
  const c = row({ t_us: 11e6, snap_t_us: 10.9e6, gen_tokens: 45, step_n: 20, step_us: 1e6, decode_n: 20, decode_us: 9e5,
    phase_cpu_us: 5e5, draft_tokens: 60, draft_accepted: 30, n_busy: 1, publish_n: 4, publish_us: 20 });
  const s = derive(p, c);
  assert.equal(s.decode_tps, 50);
  assert.equal(s.step_ms, 50);
  assert.equal(s.decode_ms, 45);
  assert.equal(s.tokens_per_step, 2.25);
  assert.equal(s.accept, 0.5);
  assert.equal(s.cpu_ms, 25);
  assert.ok(Math.abs(s.cpu_share - 5 / 9) < 1e-12);
  assert.equal(s.publish_us, 5);
  assert.equal(s.in_flight, 1);
});

test("a snapshot that did not move: zero when idle, unknown when busy", () => {
  const p = row();
  assert.equal(derive(p, row({ t_us: 11e6 })).decode_tps, 0);
  assert.equal(derive(p, row({ t_us: 11e6, n_busy: 1 })).decode_tps, null);
});

test("no rate across a restart or a counter that went back", () => {
  const p = row({ gen_tokens: 100, step_n: 10, step_us: 1e5 });
  const restarted = derive(p, row({ t_us: 11e6, snap_t_us: 11e6, t_start_us: 9e6, gen_tokens: 5 }));
  assert.equal(restarted.decode_tps, null);
  assert.equal(restarted.step_ms, null);
  const back = derive(p, row({ t_us: 11e6, snap_t_us: 11e6, gen_tokens: 50 }));
  assert.equal(back.decode_tps, null);
  for (const [k, v] of nums(restarted)) assert.ok(Number.isFinite(v), k);
});

test("nulls never turn into zeros", () => {
  const s = derive(row(), row({ t_us: 11e6 }));
  for (const k of ["cpu_pct", "iowait_pct", "server_share_pct", "prefill_tps", "moe_hit_rate", "pcie_rx_mbs", "gpu_sm"]) {
    assert.equal(s[k], null, k);
  }
});

test("the expert cache on its own clock", () => {
  const p = row({ moe_t_us: 10e6, moe_hit: 100, moe_miss: 100, moe_upload: 0, moe_evict: 0, moe_readmit: 0,
    moe_upload_bytes: 0, moe_upload_bytes_warm: 0 });
  const c = row({ t_us: 11e6, moe_t_us: 10.5e6, moe_hit: 130, moe_miss: 170, moe_upload: 50, moe_evict: 40, moe_readmit: 10,
    moe_upload_bytes: 2e6, moe_upload_bytes_warm: 5e5 });
  const s = derive(p, c);
  assert.equal(s.moe_hit_rate, 0.3);
  assert.equal(s.moe_upload_s, 100);
  assert.equal(s.moe_evict_s, 80);
  assert.equal(s.moe_upload_mbs, 4);
  assert.equal(s.moe_warm_mbs, 1);
  assert.equal(s.moe_fill_mbs, 3);
  // an idle cache publishes nothing new: zero, not unknown
  assert.equal(derive(c, { ...c, t_us: 12e6 }).moe_upload_s, 0);
});

test("the machine on the sampler's clock", () => {
  const p = row({ proc_cpu_us: 0, host_busy_us: 0, host_total_us: 0, host_iowait_us: 0, gpu_util: 70, pcie_rx: 2e9 });
  const c = row({ t_us: 12e6, proc_cpu_us: 6e6, host_busy_us: 6e6, host_total_us: 12e6, host_iowait_us: 6e5, gpu_util: 35, pcie_rx: 1e9 });
  const s = derive(p, c);
  assert.equal(s.server_cpu_pct, 300); // 6 s of CPU in 2 s: three cores
  assert.equal(s.server_share_pct, 50);
  assert.equal(s.cpu_pct, 50);
  assert.equal(s.iowait_pct, 5);
  assert.equal(s.gpu_sm, 35);
  assert.equal(s.pcie_rx_mbs, 1000);
  assert.equal(derive(null, c).gpu_sm, 35);
});

test("deriveAll chains from the previous row", () => {
  const rows = [row(), row({ t_us: 11e6, snap_t_us: 11e6, gen_tokens: 10 }), row({ t_us: 12e6, snap_t_us: 12e6, gen_tokens: 30 })];
  const s = deriveAll(rows.slice(1), rows[0]);
  assert.deepEqual(s.map((x) => x.decode_tps), [10, 20]);
  assert.equal(deriveAll(rows)[0].decode_tps, null);
});

test("tasks merge by instance and id, oldest first, capped", () => {
  let list = mergeTasks([], [{ id: 1, t_end_us: 5 }, { id: 2, t_end_us: 6 }], 100);
  list = mergeTasks(list, [{ id: 2, t_end_us: 6 }, { id: 3, t_end_us: 7 }], 100);
  assert.deepEqual(list.map((t) => t.id), [1, 2, 3]);
  list = mergeTasks(list, [{ id: 1, t_end_us: 9 }], 200); // same id, new server
  assert.deepEqual(list.map((t) => t.instance + ":" + t.id), ["100:1", "100:2", "100:3", "200:1"]);
  assert.equal(mergeTasks(list, [], 200, 2).length, 2);
});

test("per-layer hit rate window resets when counters go back", () => {
  const w = new LayerWindow(60);
  assert.deepEqual(w.rates(), []);
  w.push(0, [0, 0], [0, 0]);
  w.push(1, [3, 1], [1, 3]);
  assert.deepEqual(w.rates(), [0.75, 0.25]);
  w.push(2, [0, 0], [0, 0]);
  assert.deepEqual(w.rates(), []);
});

test("phase window: per decode figures over the window", () => {
  const w = new PhaseWindow(60);
  w.push(0, { steps: 10, step_us: 5e5, cpu_us: 2e5, cpu_us_layer: [1e5, 1e5], il: [0, 1] });
  assert.equal(w.view(), null);
  w.push(5, { steps: 30, step_us: 1.5e6, cpu_us: 8e5, cpu_us_layer: [4e5, 4e5], il: [0, 1] });
  assert.deepEqual(w.view(), { steps: 20, step_ms: 50, cpu_ms: 30, layers_ms: [15, 15], il: [0, 1], window_s: 60 });
});

test("PCIe ceiling from the link's generation and width", () => {
  assert.ok(Math.abs(pcieCeilingGBs({ pcie_gen_max: 3, pcie_width_max: 16 }) - 15.76) < 1e-9);
  assert.equal(pcieCeilingGBs({ pcie_gen_max: null, pcie_width_max: 16 }), null);
  assert.equal(pcieCeilingGBs(null), null);
});

test("the prompt cache: disk hit rate and prompt reuse per interval, KV fill as a gauge", () => {
  const p = row({ disk_find: 10, disk_find_hit: 4, prompt_cached: 100, prompt_tokens: 50, kv_used: 100, kv_max: 400 });
  const c = row({ t_us: 11e6, snap_t_us: 10.9e6, disk_find: 20, disk_find_hit: 9, prompt_cached: 160, prompt_tokens: 90, kv_used: 100, kv_max: 400 });
  const s = derive(p, c);
  assert.equal(s.disk_hit_rate, 0.5);
  assert.equal(s.prompt_reuse, 0.6);
  assert.equal(s.kv_fill_pct, 25);
  const idle = derive(p, row({ t_us: 11e6, snap_t_us: 10.9e6, disk_find: 10, disk_find_hit: 4, prompt_cached: 100, prompt_tokens: 50 }));
  assert.equal(idle.disk_hit_rate, null);
  assert.equal(idle.prompt_reuse, null);
  assert.equal(derive(null, row({ kv_used: 1, kv_max: 0 })).kv_fill_pct, null);
});
