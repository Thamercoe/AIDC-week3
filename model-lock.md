# Model lock (team record)

## The locked model

- Model id: `Qwen/Qwen2.5-1.5B-Instruct-AWQ`
- Quantisation: `awq`
- Why this one: The AWQ candidate passed the formal function-calling gate with 10/10, kept both distractor attempts call-free, and provides additional KV-cache capacity.

## The launch flags

- Flags: `--model Qwen/Qwen2.5-1.5B-Instruct-AWQ --dtype half --max-model-len 4096 --gpu-memory-utilization 0.85 --quantization awq --enable-auto-tool-choice --tool-call-parser hermes`
- Tool-call parser: `hermes`

## The smoke score

- Score (valid behaviours out of 10): `10`
- Distractor stayed call-free in the majority: `yes`
- Passed the gate (>= 8/10 and distractor majority clean): `yes`
- Measured against: AWQ `10/10`; FP16 `10/10`

## Quality spot check note

- FP16 was more coherent and complete overall. AWQ showed degradation on the tool-choice and quantisation prompts, but it passed the formal function-calling gate with valid tool behaviour and full distractor compliance.
