// tests for the /stats plane of llama-server (tools/server/server-stats.h): histogram buckets, the completed-task ring,
// the publish period and the immutability of a published snapshot

#include "../tools/server/server-stats.h"

#include "testing.h"

#include <cmath>
#include <string>

static server_stats_task make_task(int id, double t_queue_ms, double t_prompt_ms, double t_gen_ms, int n_gen,
                                   int n_draft, int n_draft_accepted) {
    server_stats_task t;
    t.id                  = id;
    t.id_slot             = 0;
    t.t_queue_ms          = t_queue_ms;
    t.t_prompt_ms         = t_prompt_ms;
    t.t_gen_ms            = t_gen_ms;
    t.n_prompt            = 100;
    t.n_prompt_processed  = 100;
    t.n_gen               = n_gen;
    t.n_gen_steps         = n_gen > 0 ? n_gen - 1 : 0;
    t.n_draft             = n_draft;
    t.n_draft_accepted    = n_draft_accepted;
    t.n_draft_verif_steps = n_draft > 0 ? n_draft / 3 : 0;
    t.stop                = "eos";
    return t;
}

static void test_hist(testing & t) {
    server_stats_hist h({ 1.0, 10.0, 100.0 });
    t.assert_equal("one bucket per edge plus overflow", (size_t) 4, h.counts.size());

    h.add(0.5);   // <= 1
    h.add(1.0);   // <= 1 (edges are inclusive upper bounds)
    h.add(5.0);   // <= 10
    h.add(100.0); // <= 100
    h.add(1e6);   // overflow
    h.add(NAN);   // ignored: a value that was never measured must not land in a bucket

    t.assert_equal((uint64_t) 2, h.counts[0]);
    t.assert_equal((uint64_t) 1, h.counts[1]);
    t.assert_equal((uint64_t) 1, h.counts[2]);
    t.assert_equal((uint64_t) 1, h.counts[3]);
    t.assert_equal((uint64_t) 5, h.n);
    t.assert_true("sum", std::fabs(h.sum - (0.5 + 1.0 + 5.0 + 100.0 + 1e6)) < 1e-6);

    const json j = h.to_json();
    t.assert_equal((size_t) 3, j.at("edges").size());
    t.assert_equal((size_t) 4, j.at("counts").size());
    t.assert_equal((uint64_t) 5, j.at("n").get<uint64_t>());
}

static void test_timer(testing & t) {
    server_stats_timer tm;
    tm.add(1000);
    tm.add(3000);
    t.assert_equal((uint64_t) 2, tm.n);
    t.assert_equal((uint64_t) 4000, tm.sum_us);
    t.assert_equal((uint64_t) 3000, tm.last_us);
    t.assert_equal((uint64_t) 3000, tm.max_us);
    t.assert_true("ema starts at the first sample and moves toward the next", tm.ema_us > 1000.0 && tm.ema_us < 3000.0);
}

static void test_ring(testing & t) {
    server_stats st(/*n_ring =*/ 3, /*period_us =*/ 1);

    for (int i = 0; i < 5; ++i) {
        st.add_task(make_task(i, 1.0, 20.0, 100.0, 11, 0, 0));
    }
    st.publish(10, nullptr);

    auto s = st.latest();
    t.assert_true("published", s != nullptr);
    t.assert_equal("the ring keeps the last n_ring tasks", (size_t) 3, s->tasks->size());
    t.assert_equal("oldest first", 2, s->tasks->front().id);
    t.assert_equal(4, s->tasks->back().id);
    t.assert_equal("counters cover every task, not only the ring", (uint64_t) 5, s->n_tasks);
    t.assert_equal((uint64_t) 5, s->h_ttft_ms.n);
    t.assert_equal("stop reasons", (uint64_t) 5, s->n_stop.at("eos"));
}

static void test_task_derived(testing & t) {
    const server_stats_task a = make_task(1, 5.0, 45.0, 300.0, 11, 30, 21);
    t.assert_true("ttft = queue wait + prompt eval", std::fabs(a.ttft_ms() - 50.0) < 1e-9);
    t.assert_true("ms per generated token over the decode steps", std::fabs(a.gen_ms_per_token() - 30.0) < 1e-9);
    t.assert_true("acceptance", std::fabs(a.accept_rate() - 0.7) < 1e-9);

    server_stats_task none = make_task(3, 5.0, 45.0, 0.0, 0, 0, 0);
    t.assert_true("no token (cancelled, failed, embedding): no ttft", std::isnan(none.ttft_ms()));
    none.n_gen      = 1;
    none.t_queue_ms = NAN;
    t.assert_true("unknown queue wait: ttft from the prompt alone", std::fabs(none.ttft_ms() - 45.0) < 1e-9);

    const server_stats_task b = make_task(2, 0.0, 10.0, 0.0, 1, 0, 0);
    t.assert_true("one token: no decode step, no rate", std::isnan(b.gen_ms_per_token()));
    t.assert_true("no draft: no acceptance", std::isnan(b.accept_rate()));

    server_stats st(4, 1);
    st.add_task(server_stats_task(b));
    st.publish(10, nullptr);
    t.assert_equal("an unmeasured rate stays out of its histogram", (uint64_t) 0, st.latest()->h_gen_ms_per_tok.n);
    t.assert_equal((uint64_t) 0, st.latest()->h_accept.n);
}

static void test_publish_period(testing & t) {
    server_stats st(4, /*period_us =*/ 1000);
    t.assert_true("nothing published yet", st.latest() == nullptr);
    t.assert_true("due before the first publish", st.due(0));

    st.publish(5000, nullptr);
    t.assert_equal((uint64_t) 1, st.latest()->seq);
    t.assert_equal((int64_t) 5000, st.latest()->t_us);
    t.assert_true("not due inside the period", !st.due(5999));
    t.assert_true("due once the period elapsed", st.due(6000));

    server_stats off(4, /*period_us =*/ 0);
    t.assert_true("period 0 disables publishing", !off.due(0) && !off.due(INT64_MAX / 2));
}

static void test_snapshot_immutable(testing & t) {
    server_stats st(4, 1);
    st.add_task(make_task(1, 1.0, 10.0, 100.0, 11, 0, 0));
    st.publish(10, [](server_stats_snapshot & s) {
        s.n_slots = 2;
        s.slots.resize(2);
        s.slots[0].id    = 0;
        s.slots[0].state = "generating";
    });
    auto first = st.latest();

    st.add_task(make_task(2, 1.0, 10.0, 100.0, 11, 0, 0));
    st.publish(20, [](server_stats_snapshot & s) {
        s.slots[0].state = "idle";
    });
    auto second = st.latest();

    t.assert_true("a new snapshot", first.get() != second.get());
    t.assert_equal("a reader's snapshot never changes under it", std::string("generating"), first->slots[0].state);
    t.assert_equal((size_t) 1, first->tasks->size());
    t.assert_equal(std::string("idle"), second->slots[0].state);
    t.assert_equal((size_t) 2, second->tasks->size());
    t.assert_equal((uint64_t) 2, second->seq);
    t.assert_equal("the publish cost of the first snapshot is in the second", (uint64_t) 1, second->publish.n);
}

static void test_json_shape(testing & t) {
    server_stats st(4, 1);
    server_stats_task task = make_task(7, 2.0, 30.0, 200.0, 21, 30, 24);
    task.accepted_per_pos = { 9, 8, 7 };
    task.sel.how  = "lcp";
    task.sel.f_sim = 0.9f;
    st.add_task(std::move(task));
    st.cur().n_sel["lcp"]++;
    st.publish(10, nullptr);

    const json j = st.latest()->to_json();
    for (const char * k : { "seq", "t_us", "t_start_us", "uptime_s", "counters", "decode", "hist", "slots", "tasks", "selection",
                            "publish" }) {
        t.assert_true(std::string("key ") + k, j.contains(k));
    }
    t.assert_true("no disk tier: null", j.at("disk_cache").is_null());
    const json & row = j.at("tasks").at(0);
    t.assert_equal(7, row.at("id").get<int>());
    t.assert_equal((size_t) 3, row.at("accepted_per_pos").size());
    t.assert_equal(std::string("lcp"), row.at("sel").at("how").get<std::string>());
    t.assert_true("derived ttft", std::fabs(row.at("ttft_ms").get<double>() - 32.0) < 1e-9);
    t.assert_equal((uint64_t) 1, j.at("selection").at("lcp").get<uint64_t>());
}

int main(int argc, char ** argv) {
    testing t;

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("hist",               test_hist);
    t.test("timer",              test_timer);
    t.test("ring",               test_ring);
    t.test("task_derived",       test_task_derived);
    t.test("publish_period",     test_publish_period);
    t.test("snapshot_immutable", test_snapshot_immutable);
    t.test("json_shape",         test_json_shape);

    return t.summary();
}
