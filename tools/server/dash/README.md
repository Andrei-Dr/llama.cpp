# llama-server statistics dashboard

Static files served by `llama-server --dash-path tools/server/dash` (env `LLAMA_ARG_DASH_PATH`) at `/dash/`. No build
step, no dependencies: `index.html`, `css/`, and ES modules in `js/`. The server reads the files on request, so an edit
is live on the next page load (each file is revalidated by its ETag).

Data (all answered by the HTTP thread from published snapshots; none of them posts a task to the main loop):

- `GET /stats` (needs `--metrics`): counters, histograms, slots, completed tasks, the expert cache, the prompt disk tier,
  the CPU-expert phase, the machine (`machine`: NVML GPUs, process and host CPU, memory) and the tasks posted by type.
- `GET /stats/history?tier=s|m&since_us=T&cols=a,b`: one row per sampler period (`LLAMA_SERVER_SAMPLE_MS`, default
  1000; `0` turns the sampler off), 15 minutes of rows and 24 hours of minute rows. Counters are cumulative; the page
  derives every rate from two rows (`js/derive.js`).
- `GET /props` every 15 s for the model name and the sleep state.

The files are public like the chat UI's; when the server has an API key, open the page as `/dash/#key=KEY` (the key is
kept in the tab's sessionStorage and sent as a bearer token).

GPU data needs `libnvidia-ml.so.1` (the NVIDIA driver); `LLAMA_SERVER_NVML_LIB` names another file and
`LLAMA_SERVER_NVML=0` skips it. Without it the GPU panels say so and the rest works.

Tests: `tools/server/tests/unit/test_dash*.py` (serving), `test_stats_history.py`, `test_stats_no_task.py`, and
`node --test tools/server/tests/dash/*.test.mjs` (the arithmetic).
