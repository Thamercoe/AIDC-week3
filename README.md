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
