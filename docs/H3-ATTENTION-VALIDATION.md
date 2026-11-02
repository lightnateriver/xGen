# H3 attention qualification — 50 actual Base steps

Measured on eight Ascend 910B1 devices with 64 GiB/device.
Torch 2.10.0+cpu, TorchNPU 2.10.0.post4 and CANN 9.1.0 were used.
This is a T2VA smoke/performance qualification, not production quality certification.
The default remains `legacy`. Spark and Veda are opt-in and experimental.

## Configuration and accounting

- Original Base weights, INT8 matmul, fused RoPE, matching 51-point AdaLN table, SP8, text TP8, PHAA2.
- 51 schedule points mean **50 actual Base DiT evaluations**. Refiner uses three Laser steps.
- 768P: 1344×768, 124 native frames at 24fps (approximately 5.17s); existing optimized INT8 VAE decoder.
- 2K: 2560×1440, 120 exported frames at 24fps (5s); existing Refiner prefetch/BF16 VAE profile.
- Only Base attention changes. Refiner, VAE and codec source files are unchanged from v0.2.
- Same frozen prompt and seeds 42/43. Twelve primary backend/resolution/seed cells, four queue-1 Laser controls and two repeated seed-42 Laser controls: **18 completed requests**.
- One model initialization per two/three-request group; requests reuse the resident workers and CPU masters.
- Request time is the existing manager wall time, excluding initialization, HTTP queue/transport, post-request memory sampling and CPU audit. New-backend telemetry is included.
- Model initialization excludes interpreter/distributed launch and offline native compilation. A one-shot CLI pays initialization plus process-first request cost; later-request values require resident reuse.
- Process-first requests can reuse the already compiled disk cache. They are not fresh-system installation timings.

## Measured request time

Each subsequent value below is one seed-43 request, not an average or a statistically established latency. Initialization is rank 0; peak allocated is the maximum across all eight ranks and group requests.

| Backend | Queue | Output | Initialization (s) | First seed42 (s) | Subsequent seed43 (s) | Peak allocated (GiB/card) |
|---|---:|---|---:|---:|---:|---:|
| legacy | 2 | 768P | 33.57 | 119.95 | 117.85 | 27.744 |
| legacy | 2 | 2K | 54.18 | 180.31 | 177.50 | 28.901 |
| veda | 2 | 768P | 45.12 | 169.38 | 98.33 | 27.756 |
| veda | 2 | 2K | 71.41 | 186.05 | 175.67 | 29.426 |
| legacy | 1 | 768P | 37.31 | 123.96 | 122.49 | 27.744 |
| legacy | 1 | 2K | 62.01 | 193.09 | 186.19 | 28.901 |
| spark | 1 | 768P | 36.52 | 166.95 | 132.42 | 27.875 |
| spark | 1 | 2K | 62.51 | 213.76 | 193.71 | 29.032 |

Controlled comparisons use the same queue mode. Spark requires queue 1/0
for NPU graph capture; queue 2 is rejected before model loading. A queue-1
Spark versus queue-2 Laser difference is a complete-configuration comparison,
not an isolated attention-backend effect.

- veda 768P, queue 2: subsequent request 117.85 → 98.33s; latency reduction 16.6% (negative means slower).
- veda 2K, queue 2: subsequent request 177.50 → 175.67s; latency reduction 1.0% (negative means slower).
- spark 768P, queue 1: subsequent request 122.49 → 132.42s; latency reduction -8.1% (negative means slower).
- spark 2K, queue 1: subsequent request 186.19 → 193.71s; latency reduction -4.0% (negative means slower).

Veda showed a useful 768P subsequent-request improvement in this sample.
Its 2K E2E improvement was small and does not establish a stable gain.
Spark is not promoted as a performance winner for this SP8/PHAA2 profile.
These results do not establish the best tuning for other head-shard sizes or workloads.

## 2K subsequent-request stage spans

Rank-0 host wall spans, seed43. Base includes the full forward path and
operator hooks; these are not pure kernel times. Other stages and overlaps
are omitted, so these columns must not be summed as the complete request.

| Backend | Queue | Base DiT ×50 (s) | Refiner DiT ×3 (s) | VAE decode (s) | Encode/audio-mux wait (s) |
|---|---:|---:|---:|---:|---:|
| legacy | 2 | 96.23 | 50.25 | 13.34 | 6.07 |
| veda | 2 | 79.44 | 50.74 | 16.48 | 6.81 |
| legacy | 1 | 98.48 | 50.75 | 11.69 | 7.08 |
| spark | 1 | 112.07 | 50.53 | 10.72 | 6.69 |

## Correctness and memory evidence

- 33 distinct CPU/compatibility tests passed, including six real-Torch CPU runtime tests. Default-path tests preserve the original CLI/service behavior when disabled. Explicit legacy override removes the sparse-only warmup restriction; existing Refiner warmup restrictions still apply.
- Twelve NPU parity cases passed: both backends, layers 0/24/49, head shards 0–3 and 53–55. Maximum absolute error was zero against the corresponding pinned upstream sparse algorithm. This is **not equivalence to dense Laser attention**.
- All 18 completed clips passed full audiovisual decode, resolution/frame-rate/frame-count and AV-duration technical checks. All eight ranks completed 50 Base steps; new backends recorded 5,000 PHAA operator calls per rank and correct global-head coverage. Every 2K clip recorded three Refiner steps.
- Each completed clip had 12 sampled frames inspected without obvious black screens, corruption or scene collapse. Screenshots do not establish complete temporal/audio quality; full human playback/listening remains pending.
- Same-seed Laser controls differed in some poses/details. End-to-end pixel equality was not assumed; sparse-generated output differences are not the operator-port parity measurement.
- Post-request allocated memory and model/CPU-master identity were stable within each tested group. Veda predictor load count remained one per process. Two/three requests are not a long-running leak test.
- Veda 2K rank0 after-snapshot PSS rose 3.68 GiB while RSS rose only 24.95 MiB and allocated did not change. Nearby parent-plus-encoder samples rose about 0.35 GiB for the entire group. Forked encoder sharing and non-atomic sampling make single-rank PSS deltas misleading; this alone is not a leak finding.
- Driver HBM, PyTorch allocated/reserved, process RSS/PSS and file cache are separate measurements. CPU master residency is substantial: the 2K chain observed roughly 720 GiB aggregate process PSS in the queue-2 run.
- All tested workers exited. A final live check found no owned workers and no compute processes on all eight NPUs; driver occupancy returned to its existing baseline. File cache remains reclaimable host memory.

## Scope and provenance

The numerical paths tested were `c4a922d` (queue 2) and `7f7507e` (queue 1),
based on published v0.2 `4868728`. The latter adds a pre-load queue-mode
guard without changing operator mathematics. Subsequent source review only
tightens integer parameter rejection, fixes service warmup after an explicit
legacy override, and updates tests/documentation; the
tested integer configurations and numerical paths are retained.

Native source snapshot: `e02a57325029e67eafaf67d64b3d62a330f1e8d4`.
Predictor revision and SHA256 are listed in [H3-ATTENTION.md](H3-ATTENTION.md).
No eight-step Base LoRA was loaded. The predictor was trained for an
eight-NFE recipe, so broader 50-step quality remains to be qualified.

Actual GPU validation used the existing InferenceManager. The existing HTTP
service received parameter/compatibility tests, but full HTTP GPU load
testing, diverse prompts, other durations/resolutions and long-running memory
stability remain unqualified. Raw videos, prompts, rank identities and
deployment traces are private review artifacts and are not redistributed.
See [integration and rollback instructions](H3-ATTENTION.md).
