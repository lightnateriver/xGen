# Experimental CUDA input Token Refiner

The default input Token Refiner remains replicated BF16 with native SDPA.
These options apply only to explicit CUDA FP8 Ref2VA profiles with eight-way
SP. They do not change the main DiT's attention, FP8 Linear, PHAA or wire dtype.

| Option | Input Token Refiner |
|---|---|
| `--h3_cuda_input_refiner none` | Original path; default and rollback |
| `--h3_cuda_input_refiner cudnn-sp` | Ulysses SP8 with forced BF16 cuDNN attention; BF16 FFN |
| `--h3_cuda_input_refiner cudnn-sp-fp8` | Same SP/cuDNN path with four W8A8 FP8 FFN projections across the two blocks |

For the experimental combined route, add the following to the existing
`fp8-sage` SP8 Ref2VA command with local output (SAP):

```bash
--h3_cuda_input_refiner cudnn-sp-fp8
```

The original two-block forward loop is retained. Its full-context input is
split across the SP group; native xGen All-to-All exchanges distribute heads
for each attention. Padding is removed before softmax and restored before
the return exchange. Final normalized outputs are gathered to retain the
original packed-input interface. A separate padding key avoids touching the
main DiT's layout. This input-refiner communication is BF16, independently
of the main DiT's optional FP8 communication.

FFN outputs, nonlinearities, residuals, attention projections and normalization
stay BF16. FP8 FFN activation amax is reduced across the SP group so each
projection uses a full-context scale. Four scalar MAX reductions are added
per two-block forward. No cross-request or latent cache is introduced.

Only the sequence/activation work is sharded; attention weights remain
replicated. FP8 FFN weights ideally save about 0.43 GiB per GPU without a GPU
BF16 duplicate, excluding scale metadata. Peak memory and speed still depend
on context size, collective overhead and temporary buffers; a speedup is not
guaranteed for short contexts. This does not make BF16 intermediate FFN
activations FP8.

The feature is inference-only. `tp_cp`, NPU and other tasks are rejected when
explicitly enabled. Restart resident workers to change the profile; disable
the option and restart to restore the original path. cuDNN is forced rather
than silently falling back. The synthetic gate in
`tools/validate_h3_input_refiner.py` checks padding, BF16 SP equivalence, FP8
error, identical-input FP8 FFN SP equivalence and actual cuDNN operator dispatch
before video validation. The first whole-module FP8 cross-path auxiliary
0.5% gate failed at three tokens (0.5735%). Diagnostics retained at 3/33/1003/8195
tokens show that upstream BF16 arithmetic differs before quantization; compared
to a common FP32 reference, three-token full/SP FP8 errors were about 0.99142%
and 0.99151%. Whole-module FP8 cross-path difference is therefore reported as a
diagnostic, while SP/FFN equivalence is tested on identical inputs. The original
BF16 SP 0.5% and combined FP8 3% primary limits remain unchanged. These random
weight gates do not establish trained model/video quality; full video and audio
review is required separately.

## Single-image R2VA measurement (2026-11-07)

Eight RTX 5090 GPUs, 1344x768, native 124 frames at 24 fps, seed 42, 20 actual
DiT steps. Each arm used one full 20-step same-shape warmup followed by one
resident 20-step measured request, without rebuilding or requantizing models.
Main DiT SP8/Sage/FP8 Linear/PHAA2 FP8 transport, text FP8 TP8, SAP local output,
conditioning CPU staging and SP8 FP16 tiled VAE were held constant.

| Metric | Original scheme B | Input refiner cuDNN/SP8/FP8 FFN |
|---|---:|---:|
| Initialization, separate from measured request | 134.27 s | 141.84 s |
| Measured generate + save | 172.03 s | 177.05 s |
| Steady DiT step | 2.3845 s | 2.3350 s |
| Steady nested input Token Refiner | 73.427 ms | 21.944 ms |
| Peak PyTorch allocated per GPU | 28.356 GiB | 27.925 GiB |

Steady values take the slowest rank for each step, then the median of steps
2-20. The refiner is nested in DiT and must not be added to it. Refiner speedup
was 3.35x and steady DiT latency fell 2.07%, but measured E2E was 2.92% slower.
The latest rank reached the first DiT at 117.68 vs 122.53 seconds relative to
the request timing origin, while video decode envelopes were 4.42 vs 5.58 s.
These front-end/readiness and decode differences outweighed about one second
of steady DiT savings. One sample per arm does not establish a sustained E2E
benefit or regression. Early ranks wait inside the new refiner collectives;
its first-step envelope is not pure compute time.

All ranks completed 20 actual DiT calls and 40 forced cuDNN refiner attention
calls per measured request. Four FP8 FFN projections were installed. Input
shape was [1, 7831, 5376], with 979 positions per rank and one padded token.
Technical checks and full media decode passed; 12 sampled frames per video
showed the expected people, cuts and movement without obvious frame-wide
corruption. Encoded-video decoded-frame SSIM was 0.814938, not pixel equivalence.
Full human motion, lip-sync and audio listening approval remains pending.
Both eight-rank worker groups exited and released GPU resources.

The first real load exposed a lifecycle error: native SP groups are created
after CPU weight preparation. The adapter now packs FFN weights on CPU first
and binds the existing native SP group on its first forward. The completed
baseline was retained; only the corrected experimental arm was rerun. Source
commits were `02992ff` (baseline) and `0273b19` (experimental); their only changes
were the optional adapter's late binding and its lifecycle validation tool.
The default-off baseline path did not change.

## Optional reuse within a managed request

`--h3_cuda_precompute_conditioning` is a separate, default-off option for CUDA
FP8 Ref2VA SP8. It also works with the original BF16 input Token Refiner.
On the first DiT call, condition projection and the refiner run before video
and audio input projections. Later steps reuse that result if the same
condition tensor is unchanged. Positive/negative conditions have separate
entries, bounded to two; changed conditions are recomputed. Inference tensors
without version counters use a snapshot comparison to detect in-place changes.
The existing managed inference call owns the scope. No request tensor is kept
on the model after it ends; both success and exception paths clear it. Outputs
are released before the existing VAE decode trim, then the request scope is
cleared again on exit. Only scalar counts and byte metrics remain. Direct model
calls without this managed scope reject the enabled feature. Restart resident
workers to change this option. NPU and the original default path are unchanged.

### 15-second one-step memory check (2026-11-07)

One single-image Ref2VA request completed on eight initially idle RTX 5090s:
1344x768, requested 362 frames, 361 muxed frames, one actual DiT step, no warmup
or retry. The original BF16 input refiner was retained; only request-scoped
condition precomputation was enabled. Main DiT used scheme B with SP8/Sage/FP8,
50 resident blocks, conditioning staging, SAP output and SP8 FP16 tiled VAE.

| Per-GPU maximum | Value |
|---|---:|
| PyTorch allocated during inference | 29.742 GiB |
| PyTorch reserved during inference | 30.402 GiB |
| Sampled driver usage, including initialization | 32031 MiB |

DiT, VAE, audio/video save and full decode completed without OOM. Each rank
computed the condition once; its BF16 output occupied about 80.30 MiB and was
released before VAE and cleared at request end. All workers then exited and
GPU driver usage returned to zero. Driver sampling was every ten seconds and
can miss brief peaks; PyTorch peaks excluded initialization. This is only a
single-step capacity/chain check, not a 20-step memory, speed or quality result.
