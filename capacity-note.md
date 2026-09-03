# Capacity note (team, one page)

## The numbers

- Locked model: `Qwen/Qwen2.5-1.5B-Instruct-AWQ`
- Target p95 end-to-end latency (your SLO today): `3.0` seconds
- Knee concurrency (highest concurrency whose p95 is still under target): `16` (sweep-bounded)
- Tokens per second at the knee: `679.42`
- Max sustainable request rate at the target p95: `6.54 req/s`

## The limiting family

- Memory-bound: doubling concurrency from 8 to 16 increased throughput by only 43% while p95 rose from 1.957 to 2.413 seconds, indicating that decode is approaching the memory-bandwidth ceiling.

## Why the knee, not the peak

- The knee represents capacity that still meets the 3-second p95 SLO, while throughput beyond that latency target would not be a reliable service commitment.
