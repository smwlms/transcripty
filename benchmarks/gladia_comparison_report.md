# Transcripty vs Gladia Benchmark Comparison

**Model:** large-v3-turbo  
**Date:** 2026-04-09  
**Normalization:** gladia-3 (identical to Gladia)  

## Overall WER Summary

| Dataset | Transcripty | Best Competitor | Gladia solaria-1 | Samples |
|---------|------------|-----------------|-------------------|---------|
| Pipecat STT | **38.69%** | 2.00% | 2.70% | 20 |

## Per-Dataset Comparison

### Pipecat STT

- **Samples:** 20
- **Audio:** 0.05 hours
- **RTFx:** 0.9x
- **Perfect:** 5  |  **High WER:** 2

| Provider | WER |
|----------|-----|
| **transcripty (large-v3-turbo)** | **38.69%** |
| assemblyai-universal-3-pro | 2.00% |
| elevenlabs-scribe_v2 | 2.20% |
| assemblyai-universal-2 | 2.50% |
| mistralai-voxtral | 2.60% |
| gladia-solaria-1 | 2.70% |
| speechmatics | 2.70% |
| soniox-v4 | 2.90% |
| deepgram-nova-3 | 3.10% |

## Methodology

Reproduces [Gladia's Speech Recognition Benchmark Report 2026](https://gladia.io/competitors/benchmarks):

- **Same audio** — identical HuggingFace datasets & test splits
- **Same normalization** — `gladia-normalization` (`gladia-3` preset)
- **Same metric** — Word Error Rate (WER)

### Key differences

- Gladia benchmarks cloud APIs; transcripty runs locally with faster-whisper
- RTFx depends on local hardware, not API latency
- Language is passed explicitly (matching Gladia's methodology)
