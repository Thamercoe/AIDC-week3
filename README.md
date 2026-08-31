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
