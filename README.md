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
