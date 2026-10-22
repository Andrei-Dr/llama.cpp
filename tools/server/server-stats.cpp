#include "server-stats.h"

#include <algorithm>
#include <atomic>
#include <cmath>

// histogram edges: TTFT runs from a cached prompt (tens of ms) to a cold 128k-token prompt (minutes); decode per token
// and per step around 20-60 ms; acceptance in tenths
static const std::vector<double> k_edges_ttft_ms   = { 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000, 120000, 300000 };
static const std::vector<double> k_edges_ms_per_tk = { 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 150, 250, 500 };
static const std::vector<double> k_edges_step_ms   = { 10, 20, 30, 40, 50, 60, 70, 80, 100, 125, 150, 200, 300, 500, 1000 };
static const std::vector<double> k_edges_accept    = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };

// JSON: nlohmann writes NaN (a rate that was never measured) as null

static json counts_to_json(const std::map<std::string, uint64_t> & m) {
    json j = json::object();
    for (const auto & [k, v] : m) {
        j[k] = v;
    }
    return j;
}

server_stats_hist::server_stats_hist(std::vector<double> edges) : edges(std::move(edges)) {
    counts.assign(this->edges.size() + 1, 0);
}

void server_stats_hist::add(double v) {
    if (std::isnan(v)) {
        return;
    }
    const size_t i = std::lower_bound(edges.begin(), edges.end(), v) - edges.begin();
    counts[i]++;
    n++;
    sum += v;
}

json server_stats_hist::to_json() const {
    return json {
        { "edges",  edges  },
        { "counts", counts },
        { "n",      n      },
        { "sum",    sum    },
    };
}

void server_stats_timer::add(uint64_t us) {
    ema_us  = n == 0 ? (double) us : ema_us + ((double) us - ema_us) / 16.0;
    n      += 1;
    sum_us += us;
    last_us = us;
    max_us  = std::max(max_us, us);
}

json server_stats_timer::to_json() const {
    return json {
        { "n",       n       },
        { "sum_us",  sum_us  },
        { "last_us", last_us },
        { "max_us",  max_us  },
        { "ema_us",  ema_us  },
    };
}

json server_stats_selection::to_json() const {
    return json {
        { "id_task",       id_task       },
        { "how",           how           },
        { "f_sim",         f_sim         },
        { "f_keep",        f_keep        },
        { "cache_update",  cache_update  },
        { "disk_restored", disk_restored },
        { "n_disk_tokens", n_disk_tokens },
        { "f_sim_disk",    f_sim_disk    },
        { "t_cache_ms",    t_cache_ms    },
    };
}

double server_stats_task::ttft_ms() const {
    if (n_gen <= 0) {
        return NAN; // cancelled or failed before the first token, or an embedding / rerank task
    }
    return (std::isnan(t_queue_ms) ? 0.0 : t_queue_ms) + t_prompt_ms;
}

double server_stats_task::gen_ms_per_token() const {
    return n_gen_steps > 0 ? t_gen_ms / n_gen_steps : NAN;
}

double server_stats_task::accept_rate() const {
    return n_draft > 0 ? (double) n_draft_accepted / n_draft : NAN;
}

json server_stats_task::to_json() const {
    return json {
        { "id",                  id                                  },
        { "id_slot",             id_slot                             },
        { "t_end_us",            t_end_us                            },
        { "t_queue_ms",          t_queue_ms                          },
        { "t_prompt_ms",         t_prompt_ms                         },
        { "t_gen_ms",            t_gen_ms                            },
        { "ttft_ms",             ttft_ms()                           },
        { "gen_ms_per_token",    gen_ms_per_token()                  },
        { "n_prompt",            n_prompt                            },
        { "n_prompt_cached",     n_prompt_cached                     },
        { "n_prompt_processed",  n_prompt_processed                  },
        { "n_gen",               n_gen                               },
        { "n_gen_steps",         n_gen_steps                         },
        { "n_ctx",               n_ctx                               },
        { "truncated",           truncated                           },
        { "n_draft",             n_draft                             },
        { "n_draft_accepted",    n_draft_accepted                    },
        { "n_draft_verif_steps", n_draft_verif_steps                 },
        { "accept_rate",         accept_rate()                       },
        { "accepted_per_pos",    accepted_per_pos                    },
        { "n_graphs_reused",     n_graphs_reused                     },
        { "stop",                stop                                },
        { "sel",                 sel.to_json()                       },
    };
}

json server_stats_slot::to_json() const {
    return json {
        { "id",                 id                 },
        { "id_task",            id_task            },
        { "state",              state              },
        { "n_prompt",           n_prompt           },
        { "n_prompt_cached",    n_prompt_cached    },
        { "n_prompt_processed", n_prompt_processed },
        { "n_gen",              n_gen              },
        { "n_ctx",              n_ctx              },
        { "n_ctx_max",          n_ctx_max          },
        { "n_draft",            n_draft            },
        { "n_draft_accepted",   n_draft_accepted   },
        { "t_queue_ms",         t_queue_ms         },
        { "t_age_ms",           t_age_ms           },
        { "t_prompt_ms",        t_prompt_ms        },
        { "t_gen_ms",           t_gen_ms           },
    };
}

json server_stats_snapshot::disk_json() const {
    const server_prompt_disk_stats & d = disk;
    return json {
        { "entries",             disk_entries          },
        { "bytes",               disk_bytes            },
        { "limit_bytes",         disk_limit            },
        { "find",                d.n_find              },
        { "find_hit",            d.n_find_hit          },
        { "f_keep_hit",          d.f_keep_hit          },
        { "f_sim_hit",           d.f_sim_hit           },
        { "save",                d.n_save              },
        { "save_skipped",        d.n_save_skipped      },
        { "save_refused",        d.n_save_refused      },
        { "save_cancelled",      d.n_save_cancelled    },
        { "save_failed",         d.n_save_failed       },
        { "save_tokens",         d.save_tokens         },
        { "save_bytes",          d.save_bytes          },
        { "save_us",             d.save_us             },
        { "restore",             d.n_restore           },
        { "restore_failed",      d.n_restore_failed    },
        { "restore_tokens",      d.restore_tokens      },
        { "restore_bytes",       d.restore_bytes       },
        { "restore_us",          d.restore_us          },
        { "evict_expired",       d.n_evict_expired     },
        { "evict_space",         d.n_evict_space       },
        { "superseded",          d.n_superseded        },
        { "evict_bytes",         d.evict_bytes         },
    };
}

json server_stats_snapshot::to_json(int max_tasks) const {
    const server_metrics & m = metrics;

    json slots_j = json::array();
    for (const auto & s : slots) {
        slots_j.push_back(s.to_json());
    }
    json tasks_j = json::array();
    if (tasks) {
        const size_t n     = tasks->size();
        const size_t first = max_tasks < 0 || (size_t) max_tasks >= n ? 0 : n - (size_t) max_tasks;
        for (size_t i = first; i < n; ++i) {
            tasks_j.push_back((*tasks)[i].to_json());
        }
    }

    return json {
        { "schema",   1                                    },
        { "seq",      seq                                  },
        { "t_us",     t_us                                 },
        { "t_start_us", t_start_us                         },
        { "uptime_s", t_start_us > 0 ? (t_us - t_start_us) / 1e6 : 0.0 },
        { "n_slots",  n_slots                              },
        { "n_deferred", n_deferred                         },
        { "counters", {
            { "prompt_tokens",        m.prompt.count        },
            { "prompt_us",            m.prompt.time         },
            { "prompt_tokens_cached", m.n_prompt_cached     },
            { "gen_tokens",           m.predict.count       },
            { "gen_steps",            m.predict.steps       },
            { "gen_us",               m.predict.time        },
            { "n_decode",             m.n_decode            },
            { "n_busy_slots",         m.n_busy_slots        },
            { "n_tokens_max",         m.n_tokens_max        },
            { "draft_tokens",         m.n_draft_tokens      },
            { "draft_accepted",       m.n_draft_accepted    },
            { "draft_verif_steps",    m.n_draft_verif_steps },
            { "accepted_per_pos",     m.n_accepted_per_pos  },
            { "tasks",                n_tasks               },
            { "truncated",            n_truncated           },
            { "stop",                 counts_to_json(n_stop) },
        }},
        { "decode", {
            { "gen",      decode_gen.to_json()    },
            { "prompt",   decode_prompt.to_json() },
            { "step_gen", step_gen.to_json()      },
        }},
        { "hist", {
            { "ttft_ms",          h_ttft_ms.to_json()        },
            { "gen_ms_per_token", h_gen_ms_per_tok.to_json() },
            { "accept",           h_accept.to_json()         },
            { "step_ms",          h_step_ms.to_json()        },
        }},
        { "selection",  counts_to_json(n_sel) },
        { "slots",      slots_j           },
        { "tasks",      tasks_j           },
        { "disk_cache", has_disk ? disk_json() : json(nullptr) },
        { "publish",    publish.to_json() },
    };
}

json server_stats_moe_cache_json() {
    llama_moe_cache_stats h = {};
    if (!llama_moe_cache_get_stats(&h, nullptr, 0, nullptr, 0)) {
        return nullptr;
    }
    // the sizes are fixed once the cache is built, so a second read fills arrays of the first one's size
    std::vector<llama_moe_cache_layer_stats> layers((size_t) h.n_layer);
    std::vector<int32_t>                     slot_expert((size_t) h.n_layer*h.n_slots);
    if (!llama_moe_cache_get_stats(&h, layers.data(), layers.size(), slot_expert.data(), slot_expert.size())) {
        return nullptr;
    }

    // per layer, columnar
    json il = json::array(), resident = json::array(), hit = json::array(), miss = json::array(), upload = json::array(),
         evict = json::array(), slots = json::array();
    for (int32_t li = 0; li < h.n_layer; ++li) {
        const llama_moe_cache_layer_stats & l = layers[li];
        il.push_back(l.il);
        resident.push_back(l.n_resident);
        hit.push_back(l.n_hit);
        miss.push_back(l.n_miss);
        upload.push_back(l.n_upload);
        evict.push_back(l.n_evict);
        slots.push_back(std::vector<int32_t>(slot_expert.begin() + (size_t) li*h.n_slots, slot_expert.begin() + (size_t) (li + 1)*h.n_slots));
    }

    return json {
        { "seq",          h.seq          },
        { "t_us",         h.t_us         },
        { "n_layer",      h.n_layer      },
        { "n_expert",     h.n_expert     },
        { "n_slots",      h.n_slots      },
        { "n_live",       h.n_live       },
        { "n_in_flight",  h.n_in_flight  },
        { "suspended",    h.suspended    },
        { "pressure",     h.pressure     },
        { "sync_publish", h.sync_publish },
        { "slot_bytes",   h.slot_bytes   },
        { "counters", {
            { "steps",             h.n_steps           },
            { "hit",               h.n_hit             },
            { "miss",              h.n_miss            },
            { "miss_in_flight",    h.n_miss_in_flight  },
            { "refused",           h.n_refused         },
            { "upload",            h.n_upload          },
            { "upload_warm",       h.n_upload_warm     },
            { "readmit",           h.n_readmit         },
            { "evict",             h.n_evict           },
            { "no_slot",           h.n_no_slot         },
            { "upload_bytes",      h.upload_bytes      },
            { "upload_bytes_warm", h.upload_bytes_warm },
            { "suspend",           h.n_suspend         },
            { "resume",            h.n_resume          },
            { "resume_partial",    h.n_resume_partial  },
            { "bytes_released",    h.bytes_released    },
        }},
        { "layers", {
            { "il",       il       },
            { "resident", resident },
            { "hit",      hit      },
            { "miss",     miss     },
            { "upload",   upload   },
            { "evict",    evict    },
        }},
        { "slot_expert", slots },
        { "publish", {
            { "n",      h.publish_n      },
            { "sum_us", h.publish_us_sum },
            { "max_us", h.publish_us_max },
        }},
    };
}

server_stats::server_stats(size_t n_ring, int64_t period_us) : n_ring(std::max<size_t>(1, n_ring)), period_us(period_us) {
    w.h_ttft_ms        = server_stats_hist(k_edges_ttft_ms);
    w.h_gen_ms_per_tok = server_stats_hist(k_edges_ms_per_tk);
    w.h_accept         = server_stats_hist(k_edges_accept);
    w.h_step_ms        = server_stats_hist(k_edges_step_ms);
}

void server_stats::add_task(server_stats_task && t) {
    w.n_tasks++;
    w.n_truncated += t.truncated ? 1 : 0;
    w.n_stop[t.stop]++;
    w.h_ttft_ms.add(t.ttft_ms());
    w.h_gen_ms_per_tok.add(t.gen_ms_per_token());
    w.h_accept.add(t.accept_rate());

    ring.push_back(std::move(t));
    while (ring.size() > n_ring) {
        ring.pop_front();
    }
    ring_dirty = true;
}

bool server_stats::due(int64_t t_now_us) const {
    if (period_us <= 0) {
        return false;
    }
    return !published || t_now_us - t_last_publish_us >= period_us;
}

void server_stats::publish(int64_t t_now_us, const std::function<void(server_stats_snapshot &)> & fill) {
    const int64_t t0 = ggml_time_us();

    if (fill) {
        fill(w);
    }
    if (ring_dirty || !w.tasks) {
        w.tasks    = std::make_shared<const std::vector<server_stats_task>>(ring.begin(), ring.end());
        ring_dirty = false;
    }
    w.seq++;
    w.t_us = t_now_us;

    std::shared_ptr<const server_stats_snapshot> s = std::make_shared<const server_stats_snapshot>(w);
    std::atomic_store(&snap, std::move(s));

    t_last_publish_us = t_now_us;
    published         = true;

    w.publish.add((uint64_t) std::max<int64_t>(0, ggml_time_us() - t0));
}

std::shared_ptr<const server_stats_snapshot> server_stats::latest() const {
    return std::atomic_load(&snap);
}
