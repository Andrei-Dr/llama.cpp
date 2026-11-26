# MoE cache

When the experts of MoE layers stay in host memory (`-cmoe`, `-ncmoe`, `-ot`), the MoE cache keeps the most used ones in device memory for small batches (decode, speculative decoding). Large batches still upload the experts they use.

| Argument | |
|---|---|
| `--moe-cache-mib N` | cache size in MiB, split among the GPUs like the layers (default: 0, disabled) |
| `--moe-cache-cpu-share F` | share of the cache misses computed on the CPU from the host experts instead of uploaded, once the cache is full (default: 0) |

The cache hits run on the device. With `--moe-cache-cpu-share F` the CPU computes a share `F` of the misses while the device computes the hits and uploads the other misses into the cache. This is faster when the CPU reads the host experts faster than the device can upload them, for example with fast RAM, a PCIe 4 link or experts that are not in pinned memory. The host experts are pinned with `-lm none`, `-lm mlock` and `-lm dio`, unless `--no-host`, but not when they are mmap'd (the default). The device uploads pinned experts at the full speed of the link, so the CPU share often does not help then. Until the cache is full every miss is uploaded, so the cache fills as fast as without the CPU share. The best share depends on the machine, the model and the cache size.

## Mid-size batches

A prompt batch that is too large for the cache (over 32 tokens) and under the offload threshold of the device (`GGML_OP_OFFLOAD_MIN_BATCH`) computes all of its experts on the CPU. With the environment variable `LLAMA_MOE_HYBRID=1` such a batch splits the experts of each layer: the ones routed few tokens stay on the CPU, the others run on the device at the same time (the ones in the cache are copied inside the device, the rest are uploaded). Off by default.

| Variable | |
|---|---|
| `LLAMA_MOE_HYBRID=1` | enable the split |
| `LLAMA_MOE_HYBRID_MIN`, `LLAMA_MOE_HYBRID_MAX` | batch sizes that are split (default: 33 and 127 tokens) |
| `LLAMA_MOE_HYBRID_COSTS=a,c,g,h` | cost model in microseconds per layer: `a` per routed token on the CPU, `c` per expert on the CPU, `g` per uploaded expert, `h` per expert copied from the cache (default: 27,40,95,10) |

The rows of the experts on the CPU use the arithmetic of the CPU backend, the others the arithmetic of the device, as for a cache hit and a miss on the CPU. The split is not used while the cache is suspended (prefill mode), with expert biases, LoRA adapters, or an eval callback.

## Picking the arguments

Put the script in front of your server command:

```bash
python3 scripts/moe-cache-tune.py -- build/bin/llama-server -m model.gguf -cmoe -t 8 -c 32768
```

It starts the server 6 times, at two cache sizes × shares 0, 0.5 and 0.95, and times the generation of a few thousand tokens. That takes a few minutes, mostly model loading.

From these runs it fits the upload and CPU speeds of the machine, then replays the routing of all the runs through a copy of the cache logic for other sizes and shares. It prints the measured speeds, the predicted t/s and the recommended arguments: the lowest share within 5% of the fastest.

- **Server command:** keep your own arguments (`-t`, `-c`, expert placement, load mode); the speeds and the free device memory depend on them.
- **`--max-mib N`:** tunes a smaller cache than what fits on the device. The runs in `--out` are reused while the command stays the same, so trying another value is instant.
- **`--markdown`:** prints Markdown tables to paste in a PR.
