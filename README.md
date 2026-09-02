# AIDC W3D1 — Profile Inference on a Real GPU

This lab profiles `Qwen/Qwen2.5-1.5B-Instruct` on a Google Colab Tesla T4. It compares FP16 and INT8 across three context lengths, then tests batch 1 against batch 8 to show why GPU utilisation is not the same as useful throughput.

## Objective

- Measure resident VRAM at context lengths 512, 2048, and 4096.
- Compare FP16 and INT8 memory usage and generation speed.
- Measure mean GPU utilisation during single-request decoding.
- Compare batch-1 and batch-8 throughput.
- Produce `profile.json` and `batch_check.json` as evidence.

## Predictions

Predictions recorded before opening Colab:

| Measurement | Prediction |
| --- | ---: |
| FP16 weight memory | 3 GB |
| INT8 weight memory | 1.5 GB |
| FP16 resident VRAM at context 512 | 4 GB |
| FP16 resident VRAM at context 4096 | 8 GB |
| Single-request decode utilisation | 1% |

The weight-memory predictions used:

```text
memory = parameters * bytes per parameter

FP16: 1.5B * 2 bytes = 3 GB
INT8: 1.5B * 1 byte  = 1.5 GB
```

The resident-VRAM prediction assumed that the longer context would add about 4 GB because the KV cache grows with context length.

## Environment

| Component | Value |
| --- | --- |
| Runtime | Google Colab |
| GPU | Tesla T4, 15360 MiB |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Transformers | `4.46.*` |
| Accelerate | `1.1.*` |
| BitsAndBytes | `0.49.2` |

The lab deliberately uses Transformers directly and does not install vLLM. Colab's existing Torch remains in place, avoiding the Torch and NumPy compatibility changes that a vLLM installation would introduce.

## Method

For each dtype, the model was loaded once and profiled at three context lengths:

```python
rows = []
for dtype in ["fp16", "int8"]:
    model = load(dtype)
    for context in [512, 2048, 4096]:
        row = profile(model, dtype, context)
        print(row)
        rows.append(row)
    del model
    free_vram()
```

Each run:

1. Created a prompt at the requested context length.
2. Performed an eight-token warm-up.
3. Recorded resident PyTorch VRAM.
4. Sampled `nvidia-smi` utilisation every two seconds.
5. Generated 128 new tokens.
6. Calculated tokens per second.

## Profiling results

| Dtype | Context | Resident VRAM | Mean utilisation | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 512 | 3.113 GB | 46.3% | 30.0 |
| FP16 | 2048 | 3.295 GB | 67.7% | 25.2 |
| FP16 | 4096 | 3.568 GB | 86.3% | 26.7 |
| INT8 | 512 | 1.805 GB | 23.2% | 5.8 |
| INT8 | 2048 | 2.035 GB | 27.7% | 5.6 |
| INT8 | 4096 | 2.309 GB | 31.5% | 5.2 |

## Predictions versus measurements

| Measurement | Predicted | Measured | Difference |
| --- | ---: | ---: | ---: |
| FP16 VRAM at context 512 | 4.000 GB | 3.113 GB | -0.887 GB |
| FP16 VRAM at context 4096 | 8.000 GB | 3.568 GB | -4.432 GB |
| FP16 increase from 512 to 4096 | 4.000 GB | 0.455 GB | -3.545 GB |
| Single-request utilisation | 1% | 46.3%-86.3% | Higher than predicted |

The longer context increased resident VRAM as expected, but by much less than predicted. FP16 remained above INT8 at every shared context length.

## Batch experiment

The same FP16 model and 512-token context were profiled at batch 1 and batch 8:

| Batch | Resident VRAM | Mean utilisation | Tokens/s |
| --- | ---: | ---: | ---: |
| 1 | 3.113 GB | 43.8% | 19.3 |
| 8 | 3.527 GB | 78.7% | 206.1 |

Calculated comparison:

```text
throughput ratio = 206.1 / 19.3 = 10.68x
utilisation ratio = 78.7 / 43.8 = 1.80x
ratio of ratios = 10.68 / 1.80 = 5.94x
utilisation delta = 78.7 - 43.8 = 34.9 percentage points
```

Batch 8 completed over ten times as much token work per second, while the utilisation reading increased by less than two times. This demonstrates that `nvidia-smi` utilisation indicates that work was queued, but does not measure how efficiently the GPU was used.

## Findings

- Resident VRAM increased with context length for both dtypes.
- FP16 used more VRAM than INT8 at every context length.
- INT8 reduced memory use but was substantially slower in this Transformers and BitsAndBytes configuration.
- Single-request utilisation rose with context, but utilisation alone did not describe throughput.
- Batching greatly improved total throughput by giving the GPU more work to process in parallel.
- A GPU can appear busy while still having substantial unused productive capacity.

## Artifacts

The experiment produced:

```text
profile.json
batch_check.json
```

`profile.json` contains the six dtype/context measurements. `batch_check.json` contains the batch-1 and batch-8 throughput values used by the verifier.

## Verification

The supplied `verify_cell.py` checked:

- Six rows with the required schema.
- FP16 and INT8 measurements.
- Three context lengths.
- VRAM rising with context within each dtype.
- FP16 VRAM above INT8 at a shared context.
- Batch-8 throughput above batch-1 throughput.

Final result:

```text
rows: 6, dtypes: ['fp16', 'int8'], contexts: [512, 2048, 4096]
batch-1 tokens/s: 19.3, batch-8 tokens/s: 206.1
GREEN CHECK: PASS
```

## Key takeaway

GPU utilisation answers whether the GPU had work during a sampling interval. It does not answer how much useful work was completed. Throughput, latency, memory use, batch size, and context length must be considered together when evaluating inference performance.


# AIDC W3D2 — Inference Anatomy by Hand

This lab measures the two phases of inference on a Google Colab Tesla T4: prefill, represented by time to first token (TTFT), and decode, represented by time per output token (TPOT). It also validates the KV-cache formula and measures the efficiency ceiling of static batching.

The model is loaded directly with Transformers. No vLLM server is used, because these measurements form the hand-rolled baseline for the W3D3 engine comparison.

## Objective

- Separate TTFT from TPOT using streamed generation.
- Measure how prompt length affects prefill and decode.
- Calculate and directly measure KV-cache bytes per token.
- Distinguish KV-cache memory from peak request memory.
- Measure static batching at batch sizes 1, 4, and 8.
- Export `baselines.json` for the next lab's A/B comparison.

## Predictions

Predictions recorded before opening Colab:

| Question | Prediction |
| --- | --- |
| Effect of longer prompts on TTFT | TTFT goes up |
| Main influence on TPOT | Model size and memory bandwidth |
| KV cache per token | 28 KB |
| KV cache at 4096 tokens | Approximately 0.11 GB |
| Static batch completion | The batch waits for the longest request |

### KV-cache arithmetic

Qwen2.5-1.5B has 28 layers, 2 KV heads, a head dimension of 128, and FP16 values of 2 bytes each:

```text
2 (K and V) * 28 layers * 2 KV heads * 128 head_dim * 2 bytes
= 28,672 bytes per token
= 28 KB per token

28 KB * 4096 tokens = 114,688 KB = 0.109375 GB
```

## Environment

| Component | Value |
| --- | --- |
| Runtime | Google Colab |
| GPU | Tesla T4 |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Dtype | FP16 |
| Transformers | `4.46.*` |
| Accelerate | `1.1.*` |

The model was loaded directly with Transformers. vLLM was intentionally excluded so the measurements describe the hand-written baseline rather than a serving engine.

## TTFT and TPOT measurement

`TextIteratorStreamer` streamed generated output while the main thread timestamped each yield. The first timestamp measured TTFT, and the mean gap between later timestamps measured TPOT.

A warm-up generation was discarded so CUDA initialization and kernel autotuning did not distort the shortest-prompt result.

| Prompt tokens | TTFT | TPOT | Total time | Streamed outputs |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.0331 s | 0.0311 s | 4.0134 s | 129 |
| 512 | 0.0615 s | 0.0320 s | 4.1593 s | 129 |
| 2048 | 0.2911 s | 0.0350 s | 4.7765 s | 129 |

TTFT increased from 0.0331 seconds to 0.2911 seconds, an approximately 8.8x increase. Prefill must process the complete prompt before producing the first token.

TPOT stayed comparatively stable at 0.0311-0.0350 seconds because decode repeatedly performs a similar memory-bound model step after prefill.

The exported baseline remeasured the 512-token TPOT as:

```text
0.0408 seconds per streamed output
```

## KV-cache measurement

The experiment measured both the cache tensors and the peak memory increase during generation:

| Context | Total tokens | Peak KB/token | KV KB/token | Formula |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 768 | 63.4 | 28.0 | 28.0 |
| 2048 | 2304 | 84.0 | 28.0 | 28.0 |
| 4096 | 4352 | 87.6 | 28.0 | 28.0 |

The KV cache matched the formula exactly at every context length. Peak memory was higher because it also included activations and allocator workspace during prefill.

These measurements answer different capacity questions:

- **KV cache:** memory that must be budgeted per concurrent sequence.
- **Peak memory:** whether a complete request may exceed available GPU memory.

## Static batching experiment

The queue contained 24 requests: 18 requested 32 output tokens and 6 requested 256. A static batch could not release completed short requests, so every batch waited for its longest member.

| Batch size | Wall time | Useful tokens/s | Slot efficiency |
| ---: | ---: | ---: | ---: |
| 1 | 62.69 s | 33.7 | 1.000 |
| 4 | 40.37 s | 52.3 | 0.344 |
| 8 | 20.21 s | 104.5 | 0.344 |

Batch-8 throughput relative to batch 1:

```text
104.5 / 33.7 = 3.10x
```

Batching increased total throughput, but slot efficiency fell to 34.4%. Approximately 65.6% of token slots were wasted because short requests remained in the static batch while the 256-token stragglers continued decoding.

This is the static-batching ceiling that continuous batching is designed to remove.

## Exported baseline

`baselines.json` contains:

```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "dtype": "fp16",
  "ttft_s": {
    "128": 0.0331,
    "512": 0.0615,
    "2048": 0.2911
  },
  "tpot_s": 0.0408,
  "batch": {
    "1": 33.7,
    "4": 52.3,
    "8": 104.5
  }
}
```

This file was downloaded immediately because W3D3 compares the vLLM engine against these exact hand-rolled measurements, potentially in a fresh Colab runtime.

## Artifacts

| File | Purpose |
| --- | --- |
| `README.md` | Documents predictions, measurements, and interpretation |
| `baselines.json` | TTFT, TPOT, and static-batch throughput for W3D3 |
| `kv_check.json` | Formula, measured KV bytes, and peak-memory evidence |

## Verification

The supplied verifier checked:

- Required `baselines.json` fields.
- TTFT increasing from 128 to 2048 prompt tokens.
- Positive TPOT.
- Batch-8 throughput above batch-1.
- Measured KV within a factor of two of the 28 KB/token formula.

Final result:

```text
ttft lengths: ['128', '2048', '512'], tpot_s: 0.0408
batch tokens/s 1/4/8: 33.7/52.3/104.5
KV measured 28.0 KB/token vs formula 28.0 KB/token
GREEN CHECK: PASS
```

## Key takeaways

- Prompt length mainly affects prefill and therefore TTFT.
- Decode TPOT depends mainly on model size and memory bandwidth.
- KV-cache size can be predicted exactly from model architecture and dtype.
- Peak request memory is not the same measurement as KV-cache memory.
- Static batching improves throughput but wastes capacity when output lengths differ.
- Downloaded baselines are essential because reproducible comparisons must survive a lost Colab runtime.

# AIDC W3D3 — The Engine Swap

This lab replaces the hand-rolled Transformers inference path from W3D2 with vLLM while keeping the same OpenAI-compatible `/v1` interface. The goal is to test whether a better serving engine can improve concurrent throughput without requiring client changes.

## Objective

- Launch vLLM with `Qwen/Qwen2.5-1.5B-Instruct` on a Colab Tesla T4.
- Confirm that the existing OpenAI client works by changing only its `base_url`.
- Restore the static-batching measurements from W3D2.
- Measure vLLM throughput at concurrency 1, 4, and 8.
- Compare each engine's scaling from 1 to 8.
- Export `ab_report.json` and pass the supplied verifier.

## Prediction

The prediction card was intentionally skipped, so `predicted_speedup` remains `null` in the report. No prediction was added after seeing the results.

## Environment

| Component | Value |
| --- | --- |
| Runtime | Google Colab past runtime 2026.07 |
| Python | 3.12 |
| GPU | Tesla T4 |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| vLLM | `0.6.*` |
| Transformers | `4.46.*` |
| Accelerate | `1.1.*` |
| HTTPX | `0.27.*` |
| OpenAI client | `1.54.*` |

The current Colab Python 3.13 runtime could not resolve the pinned vLLM 0.6 dependency set. The lab was therefore run on Colab's 2026.07 past runtime with Python 3.12, while keeping every package pin from the lab unchanged.

The vLLM server became healthy after approximately 167 seconds.

## Interface seam test

The existing OpenAI client connected to:

```text
http://localhost:8000/v1
```

It successfully returned a completion without changing the client request format. Only the serving engine and `base_url` changed.

This demonstrates the value of a stable API contract: applications above the `/v1` interface do not need to know whether Transformers or vLLM performs inference underneath it.

## A/B results

The W3D2 baseline used hand-rolled static batching. W3D3 sent the same 24-request workload to vLLM with asynchronous concurrency.

| Concurrency | Static baseline | vLLM | vLLM speedup |
| ---: | ---: | ---: | ---: |
| 1 | 33.7 tokens/s | 56.5 tokens/s | 1.68x |
| 4 | 52.3 tokens/s | 161.6 tokens/s | 3.09x |
| 8 | 104.5 tokens/s | 206.5 tokens/s | 1.98x |

vLLM wall-clock measurements:

| Concurrency | Requests | Wall time | Throughput |
| ---: | ---: | ---: | ---: |
| 1 | 24 | 24.578 s | 56.5 tokens/s |
| 4 | 24 | 8.595 s | 161.6 tokens/s |
| 8 | 24 | 6.728 s | 206.5 tokens/s |

## Scaling comparison

Each engine's concurrency-8 throughput was divided by its own concurrency-1 throughput:

```text
Static batching: 104.5 / 33.7 = 3.10x
vLLM:            206.5 / 56.5 = 3.65x
Scaling gain:      3.65 / 3.10 = 1.18x
```

The most useful comparison is the scaling gain, not a single per-level speedup. The per-level ratios do not rise smoothly because they compare two different throughput curves using a queue of only 24 requests.

## Why continuous batching matters

W3D2's mixed-length static batches reached only 34.4% slot efficiency. Short requests stayed in their slots after finishing and waited for the 256-token straggler.

vLLM uses continuous batching: when a sequence finishes, it can leave and another waiting request can join on a later decoding step. This reduces wasted slots and allows throughput to keep scaling under concurrency.

## Exported report

`ab_report.json` contains the measured comparison:

```json
{
  "baseline": {
    "1": 33.7,
    "4": 52.3,
    "8": 104.5
  },
  "vllm": {
    "1": 56.5,
    "4": 161.6,
    "8": 206.5
  },
  "speedup_by_concurrency": {
    "1": 1.68,
    "4": 3.09,
    "8": 1.98
  },
  "predicted_speedup": null
}
```

## Repository files

| File | Purpose |
| --- | --- |
| `README.md` | Documents the experiment, measurements, and conclusions |
| `W3D3_Engine_Swap_vLLM_T4.ipynb` | Reproducible Colab notebook for the lab |
| `ab_report.json` | Verified A/B measurements from the T4 run |

The lab-supplied `ab_client.py`, `verify_cell.py`, and `baselines.sample.json` do not need to be committed unless the instructor explicitly requests the supplied materials. The actual `baselines.json` belongs with W3D2; W3D3 consumes it but does not need a duplicate copy.

## Verification

The supplied verifier confirmed that the report contains both engines' measurements, that vLLM concurrency-8 throughput exceeds the static batch-8 baseline, and that the speedup values were calculated.

```text
baseline batch-8: 104.5, vllm concurrency-8: 206.5
speedup at 8: 1.98x
GREEN CHECK: PASS
```

## Key takeaways

- A stable OpenAI-compatible API lets the serving engine change without changing the client.
- vLLM nearly doubled throughput at concurrency 8 for this workload.
- Continuous batching scaled 3.65x from concurrency 1 to 8, compared with 3.10x for static batching.
- Continuous batching's advantage comes from reusing finished sequence slots instead of waiting for a static batch's longest request.
- Ratios between complete scaling curves are more reliable than expecting per-level speedups to increase monotonically.

# AIDC W3D4 — Quantise and Lock the Model

This lab compares an AWQ-quantised model with the FP16 model served in W3D3, checks whether quantisation affects response quality and function calling, and locks the model configuration the team will operate for the rest of the course.

The final decision is gated by function-calling reliability: a model must score at least 8/10 and avoid unnecessary tool calls on the distractor prompt.

## Objective

- Serve `Qwen/Qwen2.5-1.5B-Instruct-AWQ` with vLLM on a Tesla T4.
- Measure AWQ GPU memory usage, KV-cache capacity, and generation speed.
- Compare five qualitative responses between AWQ and FP16.
- Run the official function-calling smoke test against both models.
- Lock the candidate that satisfies the lab's smoke-test rule.
- Export the locked configuration and smoke-test evidence.

## Environment

| Component | Value |
| --- | --- |
| Runtime | Google Colab past runtime 2026.07 |
| Python | 3.12 |
| GPU | Tesla T4, 15 GB |
| Serving engine | vLLM `0.6.*` |
| Transformers | `4.46.*` |
| Accelerate | `1.1.*` |
| AutoAWQ | `0.2.*` |
| HTTPX | `0.27.*` |
| OpenAI client | `1.54.*` |

The course pins were installed unchanged. The past Colab runtime was used because the pinned vLLM 0.6 dependency set is compatible with Python 3.12.

## AWQ server configuration

The quantised model was launched with:

```text
--model Qwen/Qwen2.5-1.5B-Instruct-AWQ
--dtype half
--max-model-len 4096
--gpu-memory-utilization 0.85
--quantization awq
--enable-auto-tool-choice
--tool-call-parser hermes
```

The `hermes` parser is required for the Qwen2.5 family. It converts generated tool requests into structured OpenAI-compatible `tool_calls` rather than leaving them as ordinary prose.

The server became healthy after approximately 168 seconds, and `/v1/models` confirmed that the served model was `Qwen/Qwen2.5-1.5B-Instruct-AWQ`.

## Memory and capacity

| Measurement | AWQ result |
| --- | ---: |
| GPU memory used | 11,723 MiB |
| GPU KV-cache blocks | 22,955 |
| CPU blocks | 9,362 |

AWQ reduces the model's weight storage, but `nvidia-smi` still showed approximately 11.7 GB in use. This is expected because vLLM allocates memory up to the configured `0.85` utilization target and uses the space freed by smaller weights for additional KV-cache blocks.

The benefit therefore appears as serving capacity rather than a proportionally lower `memory.used` value.

## Throughput

The AWQ server was measured through its OpenAI-compatible endpoint after a short warm-up:

| Model | Observed throughput |
| --- | ---: |
| FP16 baseline from W3D3 | 56.5 tokens/s |
| AWQ | 4.5 tokens/s |

The measured AWQ request generated 128 tokens in 28.44 seconds. In this environment AWQ was approximately 12.6 times slower than the FP16 baseline.

The vLLM startup log warned that AWQ quantisation was not fully optimized in this version. This experiment demonstrates that reducing weight precision does not guarantee higher throughput; performance also depends on the serving engine and its available kernels.

## Five-prompt quality comparison

Both models were tested using the same five prompts covering summarisation, tool selection, rewriting, rollback instructions, and a non-technical explanation of quantisation.

FP16 was more coherent and complete overall. AWQ showed noticeable degradation on the tool-selection prompt by inventing questionable API URLs, and its quantisation explanation used an unhelpful analogy. Some responses from both models did not follow the requested sentence count, and the longer outputs could reach the 200-token limit.

Recorded judgment:

> FP16 was more coherent and complete overall. AWQ showed degradation on the tool-choice and quantisation prompts, but it passed the formal function-calling gate with valid tool behaviour and full distractor compliance.

## Function-calling smoke test

The official smoke test made 10 attempts across three prompt types:

- Four two-tool requests.
- Four single-tool requests.
- Two distractor requests that must remain call-free.

| Model | Score | Required calls valid | Distractor call-free | Passed |
| --- | ---: | ---: | ---: | --- |
| AWQ | 10/10 | 8/8 | 2/2 | Yes |
| FP16 | 10/10 | 8/8 | 2/2 | Yes |

Both models passed the formal gate. AWQ was locked because the lab specifies FP16 as the fallback when the AWQ candidate scores below 8/10; the AWQ candidate instead achieved 10/10 with full distractor compliance.

## Locked model

| Field | Locked value |
| --- | --- |
| Model | `Qwen/Qwen2.5-1.5B-Instruct-AWQ` |
| Quantisation | AWQ |
| Tool-call parser | `hermes` |
| Smoke score | 10/10 |
| Distractor compliance | 2/2 call-free |
| Gate result | Passed |

This configuration is recorded in `model-lock.md`, while `smoke_result.json` contains the machine-readable smoke-test evidence.

## Verification

The supplied verifier checked the smoke-test schema and gate, distractor compliance, and completion of the model-lock record.

```text
smoke score: 10/10, distractor clean: True
model-lock.md: all fields filled
GREEN CHECK: PASS
```

## Repository files

| File | Purpose |
| --- | --- |
| `W3D4_Quantise_and_Lock_T4.ipynb` | Complete Colab experiment and outputs |
| `model-lock.md` | Locked model, flags, smoke score, and quality decision |
| `smoke_result.json` | Machine-readable evidence for the green check |
| `README.md` | Combined W3D1–W3D4 documentation |

## Key takeaways

- Quantised weights do not necessarily make `nvidia-smi memory.used` much lower because vLLM can reinvest freed memory in KV-cache capacity.
- Quantisation does not automatically improve speed; kernel support and engine optimization matter.
- A short quality comparison can reveal degradation that a throughput benchmark cannot.
- Tool-use restraint matters as much as tool-use obedience: a model that always calls tools is unsafe for real consumers.
- Stable model and parser flags must be recorded because this locked configuration becomes an operational dependency for later course work.
