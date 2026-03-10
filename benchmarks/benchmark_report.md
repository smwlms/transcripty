# Transcripty Benchmark Report

**Date:** 2026-03-10 10:21
**Reference:** Plaud transcription service (~1 min processing)

---

## 1. Speaker Enrollment

- **Speaker:** Samuel Willems
- **Audio:** 2026-03-10 07:49:20 (105s, NL voorleestekst)
- **Diarization time:** 11.5s
- **Embedding dimension:** 256
- **Saved to:** `speakers.json`

---

## 2. Model Comparison — Enrollment Audio (105s)

Vergelijking van alle 6 Whisper modellen op de enrollment audio, met Plaud als referentie.

| Model | Tijd | RT Factor | WER | Fouten | Woorden |
|-------|------|-----------|-----|--------|---------|
| tiny | 4.94s | 0.047x | **40.9%** | 108/264 | 287 |
| base | 6.25s | 0.059x | **24.6%** | 65/264 | 270 |
| small | 15.67s | 0.149x | **9.5%** | 25/264 | 265 |
| medium | 70.99s | 0.673x | **5.3%** | 14/264 | 265 |
| large-v3 | 73.19s | 0.694x | **1.9%** | 5/264 | 265 |
| distil-large-v3 | 104.44s | 0.991x | **96.2%** | 254/264 | 185 |

**Beste kwaliteit:** `large-v3` (WER 1.9%)
**Snelste:** `tiny` (4.94s, WER 40.9%)

### WER Details per Model

#### tiny
- WER: 40.9% (108 fouten op 264 woorden)
- Substitutions: 77, Insertions: 27, Deletions: 4
- Taal: nl (1.0)
- Preview: _De 8en behoon met de regen die tegen het raam tiktig. Ik schonk een kopkoffe in en je kan tafel zitten, maar afvraagert wat de dag zal brengen. Mijn b..._

#### base
- WER: 24.6% (65 fouten op 264 woorden)
- Substitutions: 45, Insertions: 13, Deletions: 7
- Taal: nl (1.0)
- Preview: _De ochtend begon met regen die tegen het raam tikte. Ik schoonk een kopkoffie in en ga je kan aan tafel zitten, maar afvragen het wat de dag zal breng..._

#### small
- WER: 9.5% (25 fouten op 264 woorden)
- Substitutions: 18, Insertions: 4, Deletions: 3
- Taal: nl (1.0)
- Preview: _De ochtend begon met regen die tegen het raam tikten. Ik schonk een kop koffie in en ging aan tafel zitten, me afvraagend wat de dag zou brengen. Mijn..._

#### medium
- WER: 5.3% (14 fouten op 264 woorden)
- Substitutions: 9, Insertions: 3, Deletions: 2
- Taal: nl (1.0)
- Preview: _De ochtend begon met regen die tegen het raam tikte. Ik schonk een kop koffie in en ging aan tafel zitten, me afvraagend wat de dag zou brengen. Mijn ..._

#### large-v3
- WER: 1.9% (5 fouten op 264 woorden)
- Substitutions: 2, Insertions: 2, Deletions: 1
- Taal: nl (1.0)
- Preview: _De ochtend begon met regen die tegen het raam tikte. Ik schonk een kop koffie in en ging aan tafel zitten, me afvragend wat de dag zou brengen. Mijn b..._

#### distil-large-v3
- WER: 96.2% (254 fouten op 264 woorden)
- Substitutions: 175, Insertions: 0, Deletions: 79
- Taal: nl (1.0)
- Preview: _The o'clock with the rain ticked. I skunked a cup of coffee in, and I went on the table and asked me asking what the day would bring. My bufrau had a ..._

---

## 3. Strategiebespreking (57 min, multi-speaker)

Vergelijking van medium en large-v3 op een echte vergadering met speaker diarization en identificatie.

| Model | Transcriptie | Diarization | Totaal | RT Factor | WER | Sprekers |
|-------|-------------|-------------|--------|-----------|-----|----------|
| medium | 1253.63s | 197.89s | 1451.52s | 0.423x | **46.3%** | Samuel Willems (597), SPEAKER_01 (73), SPEAKER_00 (170), SPEAKER_02 (41), UNKNOWN (83) |
| large-v3 | 5662.27s | 201.86s | 5864.13s | 1.709x | **34.7%** | Samuel Willems (955), SPEAKER_01 (131), SPEAKER_00 (332), UNKNOWN (179), SPEAKER_02 (99) |

### medium — Details
- Sprekers gedetecteerd: 4
- Sprekers geidentificeerd: {'SPEAKER_03': 'Samuel Willems'}
- Segmenten per spreker: {'Samuel Willems': 597, 'SPEAKER_01': 73, 'SPEAKER_00': 170, 'SPEAKER_02': 41, 'UNKNOWN': 83}
- WER: 46.3% (3889/8396 fouten)

### large-v3 — Details
- Sprekers gedetecteerd: 4
- Sprekers geidentificeerd: {'SPEAKER_03': 'Samuel Willems'}
- Segmenten per spreker: {'Samuel Willems': 955, 'SPEAKER_01': 131, 'SPEAKER_00': 332, 'UNKNOWN': 179, 'SPEAKER_02': 99}
- WER: 34.7% (2917/8396 fouten)

---

## 4. Conclusie: Transcripty vs. Plaud

| Criterium | Plaud | Transcripty (best) |
|-----------|-------|--------------------|
| Verwerkingstijd (105s audio) | ~60s (cloud) | 73.19s (lokaal) |
| WER vs referentie | 0% (= referentie) | 1.9% |
| Beste model | — | `large-v3` |
| Snelste model | — | `tiny` (4.94s) |
| Speaker diarization | Ja (cloud) | Ja (pyannote, lokaal) |
| Speaker identification | Nee | Ja (SpeakerDB) |
| Privacy | Cloud | 100% lokaal |
| Kosten | Plaud abonnement | Gratis (open source) |

### Strategiebespreking (57 min)
- Beste model: `large-v3` (WER 34.7%)
- Totale verwerkingstijd: 5864.13s (1.709x realtime)
- Samuel Willems herkend: True
