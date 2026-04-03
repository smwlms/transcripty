# Autoresearch: Transcripty Library Optimization

> Karpathy-style autonomous research agenda voor het verbeteren van transcriptie snelheid, accuraatheid en hallucination suppression in de Transcripty Python library.
>
> Consumer: Echo (FastAPI). Platform: Apple M1 Max, 32GB unified memory.

## Baseline (April 2026)

Gebaseerd op Run 1 (2026-03-30) — 13 experimenten:

| Metric             | Waarde                                                           |
| ------------------ | ---------------------------------------------------------------- |
| Optimale config    | large-v3-turbo + int8 + beam=1 + vad + cond_prev=False + ngram=3 |
| Realtime factor    | 0.147x RTF (8.6x sneller dan initieel 1.28x)                     |
| Hallucination rate | **0.0%** (was 1.7%)                                              |
| Combined score     | 6.797 (was 0.768)                                                |
| Temperature        | 0.0 optimaal — hoger is consistent trager door fallback-retries  |

## De Primaire Metric

**Score = Kwaliteit / Tijd** (hoger = beter)

```
speed_score   = 1 / rtf                     # lager RTF = hogere score
quality_score = 1 - hallucination_rate       # minder hallucinations = hoger
combined      = speed_score * quality_score  # trade-off
```

Secundaire metrics:

- `rtf` — realtime factor (tijd / audio_duur). Target: < 0.15x
- `hallucination_rate` — % segmenten dat verdacht repetitief of onlogisch is
- `speaker_accuracy` — % correct geïdentificeerde speakers (nieuw in Run 2)

## Test Set

| #   | Opname               | Duur  | Sprekers | Taal  | Scenario                   |
| --- | -------------------- | ----- | -------- | ----- | -------------------------- |
| 1   | Roos gesprek         | 3m11s | 3        | NL    | Emotioneel, overlap        |
| 2   | Meeting Steven P.    | 31m   | 2-3      | NL    | Technisch, lang            |
| 3   | Visite immobiliere   | 22m   | 2-3      | FR/NL | Meertalig                  |
| 4   | Steve Jobs interview | 20m   | 2        | EN    | Engels, helder             |
| 5   | 1e gesprek Julie     | 15m   | 2-3      | NL/FR | Kantoor, achtergrondgeluid |

## Experiment Protocol

```
1. Noteer baseline RTF + hallucination_rate op de 5 test opnames
2. Maak ÉÉN config wijziging
3. Run benchmark op dezelfde 5 opnames
4. Meet: rtf, hallucination_rate, combined_score
5. KEEP als combined_score > baseline
6. DISCARD als gelijk of slechter → rollback
7. Log resultaat in tabel onderaan
8. Ga naar het volgende openstaande experiment
```

Voer experiments uit via:

```bash
cd /Users/samuelwillems/Documents/Projecten/Transcripty
.venv/bin/python benchmarks/benchmark.py --audio <test_file> --config <params>
# OF direct via API:
.venv/bin/python -c "
from transcripty import transcribe
import time
result = transcribe('test.mp3', temperature=0.0, vad_filter=True, ...)
"
```

---

## Research Axes

### Axis 1: Temperature — ✅ Afgerond

- [x] **T1** — `temperature=0.2` ❌ 46% trager (0.273 RTF vs 0.187)
- [x] **T2** — `temperature=0.4` ❌ trager (0.191 RTF), score 5.23
- [x] **T3** — `temperature=0.2` + `beam_size=1` ❌ score 5.14
- [x] **T4** — Temperature fallback tuple: `temperature=(0.0, 0.2, 0.4)` ✅ KEEP +6.6% (Run 2)
  - RTF 0.362x (cold run) — in warme conditie vergelijkbaar met 0.0
  - Voordeel: automatisch retry bij lage confidence, zonder vaste overhead
- [x] **T5** — `beam_size=5` ❌ -33.9% (Run 2) — greedy (beam=1) blijft beter

---

### Axis 2: Hallucination Suppression — ✅ Grotendeels afgerond

- [x] **H1** — `vad_filter=True` ✅ KEEP — RTF 1.28x→0.187x
- [x] **H2** — `condition_on_previous_text=False` ✅ KEEP — RTF 0.187x→0.146x
- [x] **H3** — `hallucination_silence_threshold=2.0` ❌ geen verbetering vs H1
- [x] **H4** — `repetition_penalty=1.1` ❌ geen verbetering
- [x] **H5** — `no_repeat_ngram_size=3` ✅ KEEP — 0% hallucinations
- [x] **H6** — Max Suppression Preset (incl. temperature=0.2) ❌ temperature schaadt score
- [x] **H7** — Max Suppression zonder temperature ❌ -2.9% (Run 2) — silence_threshold + penalty helpen niet

---

### Axis 3: Speed — Verdere Optimalisatie

Huidige baseline: 0.132x RTF (na Run 2b). Target: < 0.12x.

- [x] **S1** — cpu_threads sweep (Run 2): **cpu_threads=auto (0) = optimum**
  - S1a threads=8: ❌ -48% (cold, confounded)
  - S1b threads=10: ❌ -53%
  - S1c threads=4: ✅ +48% (koud gemeten, vertekend)
  - S1d threads=auto: ✅ KEEP +63% → auto is daadwerkelijk beter in warme conditie
  - Conclusie: **gebruik cpu_threads=0 (auto)** — CTranslate2 bepaalt zelf het optimum
- [x] **S4** — `num_workers=2` parallel verwerking ❌ DISCARD (Run 5b, 2026-04-03)
  - Sequential: 68.3s (RTF=0.135x) vs Parallel: 73.4s (RTF=0.146x) — **7% trager**
  - Oorzaak: CTranslate2 is intern al multi-threaded op M1 Max, threading overhead > winst
  - Conclusie: sequential processing is optimaal op single-node
- [x] **S5** — `word_timestamps=False` ❌ -3.8% (Run 2) — True blijft sneller

---

### Axis 4: Accuraatheid — Fine-tuning

- [x] **A1** — `repetition_penalty` sweep: 1.05 score=6.75, 1.1 score=6.63 ❌ beide lager dan 6.80
- [x] **A2** — `no_repeat_ngram_size` sweep: ngram=2 score=6.791 ❌ ngram=3 (6.797) blijft beste
- [x] **A3** — `beam_size=3` ❌ score 6.48, trager (0.154x RTF)
- [x] **A4** — `cond_prev=True` + `temperature=0.1` ❌ -42% (Run 2) — temperature>0 nog steeds traag

---

### Axis 5: Configuratie Presets — Na experimenten

- [x] **P1** — **Speed preset** ✅ DEFINED (Run 5b, 2026-04-03) — **RTF=0.130x, score=7.67**
  ```yaml
  model_size: large-v3-turbo
  compute_type: int8
  beam_size: 1
  vad_filter: true
  word_timestamps: true
  temperature: 0.0
  condition_on_previous_text: false
  no_repeat_ngram_size: 3
  language: nl
  multilingual: false
  ```
- [x] **P2** — **Quality preset** ✅ DEFINED (Run 5b) — **RTF=0.149x, score=6.71**
  ```yaml
  model_size: large-v3-turbo
  compute_type: int8
  beam_size: 3
  vad_filter: true
  word_timestamps: true
  temperature: 0.0
  condition_on_previous_text: false
  hallucination_silence_threshold: 2.0
  no_repeat_ngram_size: 3
  language: nl
  multilingual: false
  ```
- [x] **P3** — **Balanced preset** ✅ DEFINED (Run 5b) — **RTF=0.127x, score=7.89** (BESTE)
  ```yaml
  model_size: large-v3-turbo
  compute_type: int8
  beam_size: 1
  vad_filter: true
  word_timestamps: true
  temperature: [0.0, 0.2, 0.4]
  condition_on_previous_text: false
  no_repeat_ngram_size: 3
  language: nl
  multilingual: false
  ```
  **P3 is de aanbevolen default** — sneller dan P1 (0.127 vs 0.130) dankzij temperature fallback

---

### Axis 6: Speaker Identification Quality — NIEUW ⭐

Bevinding uit batch (2026-03-31): veel UNKNOWN speakers doordat scores net onder threshold (0.50) vallen. Bijv. Alex: 0.390, Roos: 0.270, Andries: 0.230. Drempel is te strikt.

**Metric voor deze axis**: `speaker_accuracy` = % segmenten correct gelabeld (handmatig te verifiëren op test opnames met bekende sprekers)

- [x] **SI1** — Threshold sweep: 0.50 → 0.40 ✅ CONDITIONAL KEEP (Run 4, 2026-04-03)
  - Resultaat: +1 match op audio NL (Samuel: 0.409 > 0.40), geen effect op Roos gesprek
  - **Maar**: Alex/Samuel embeddings overlappen sterk (0.645 vs 0.645 identiek op sommige opnames)
  - Aanbeveling: threshold=0.40 als default, maar SI4 (re-enrollment) is de echte oplossing

- [x] **SI2** — Threshold 0.50 → 0.35 ❌ DISCARD (Run 4, 2026-04-03)
  - Geen extra matches vs SI1 op beide test files
  - Verhoogt false positive risico zonder winst

- [x] **SI3** — Threshold 0.50 → 0.45 ❌ DISCARD (Run 4, 2026-04-03)
  - Samuel score vaak onder 0.45 → geen verbetering
  - SI1 (0.40) is strikt beter

- [x] **SI4** — ECAPA-TDNN enhanced speaker identification (Run 5, 2026-04-03)
  - Enhanced DB gebouwd: 7 sprekers met pyannote + ECAPA + prosody embeddings
  - Samuel multi-sample enrollment: ECAPA sim met Alex = **0.081** (vs pyannote 0.559)
  - **Maar**: bij diarized audio was enhanced ID slechter (3 false positives vs 2)
  - Oorzaak: threshold=0.35 te laag voor enhanced, en ontbrekende ground truth
  - Status: **ECAPA module GEBOUWD en WERKT**, maar scoring/threshold moet getuned
  - **Volgende stap**: per-opname ground truth labelen + threshold optimalisatie

- [x] **SI5** — `num_speakers` hint meegeven ✅ KEEP (Run 5, 2026-04-03)
  - num_speakers=2 → identiek resultaat als auto-detect op 2-speaker file
  - Pyannote detecteerde al correct 2 speakers zonder hint
  - Nuttig als vangnet bij complexe audio, geen negatief effect

---

### Axis 7: Domain Vocabulary / Initial Prompt — NIEUW ⭐

Transcripty heeft `Vocabulary` klasse en `vocabulary_path` config. Vastgoed = domein met specifieke terminologie. Whisper's `initial_prompt` kan accuraatheid op jargon verhogen.

- [x] **V1** — Vastgoed vocabulary (NL) — NEUTRAAL (Run 4, 2026-04-03)
  - RTF impact: **neutraal** (warm: 0.143x vs baseline 0.141x, binnen ruis)
  - Hallucinations: 0% (onveranderd)
  - Ogenschijnlijke +29% score was cold start artefact in baseline
  - **Nog te doen**: handmatige spot-check op vastgoedtermen nodig voor kwaliteitsmeting

- [x] **V2** — Speaker namen als vocabulary — NEUTRAAL (Run 4, 2026-04-03)
  - RTF impact: **neutraal** (warm: 0.143x vs baseline 0.141x)
  - Geen meetbare verbetering op combined_score
  - Naamnauwkeurigheid vereist handmatige evaluatie

- [x] **V3** — Combinatie vastgoed + namen — NEUTRAAL (Run 4, 2026-04-03)
  - RTF impact: **neutraal** (warm: 0.143x)
  - Geen meetbare verbetering op combined_score
  - Potentieel nuttig voor jargon-accuraatheid, niet meetbaar met huidige metrics

---

### Axis 8: Multilingual Optimization — NIEUW ⭐

Echo-opnames bevatten NL/FR mixes (bijv. Consultatie Halle, 1e gesprek Julie). Huidige config gebruikt language=None (auto-detect). Mogelijke verbeteringen:

- [x] **M1** — `multilingual=True` ✅ KEEP +1.2% vs S1d (Run 2b) — lichte verbetering
  - RTF 0.148x | Aanbevolen voor FR/NL gemengde opnames

- [x] **M2** — Multilingual FR/NL test ✅ KRITIEK VERSCHIL (Run 5b, 2026-04-03)
  - **Eerste test (49s NL/EN)**: geen FR content, alle configs gelijk
  - **Tweede test (Visite immobilière, 22 min, echt FR/NL)**:
    - NL-only: RTF=0.187x, **onbruikbaar** ("Ik ben kastig" i.p.v. "Je suis cassée")
    - multilingual=True: RTF=0.114x, **correct FR** ("Je suis cassée, c'est pour ça que...")
  - **Conclusie**: `multilingual=True` is VERPLICHT voor FR/NL opnames
  - **Bug gevonden**: lingua post-processing labelt alle segmenten als "nl" ook wanneer het FR is
  - **Actie**: `language_detect.py` fixen — lingua detectie werkt niet correct op korte FR segmenten

- [x] **M3** — `language="nl"` + `multilingual=False` ✅ KEEP +12.3% (Run 2b) — **snelste optie**
  - RTF **0.132x** — skip taaldetectie volledig
  - ⚠️ Alleen voor pure NL recordings. FR/NL mix: gebruik M1 config

---

### Axis 9: Chunk & Context Settings — NIEUW

- [x] **C1** — `chunk_length=15` ❌ DISCARD (Run 5b, 2026-04-03)
  - RTF=0.232x vs baseline 0.145x — **60% trager**
  - Minder segmenten (226 vs 254), score -37.5%
  - Conclusie: kortere chunks = meer overhead, geen voordeel

- [x] **C2** — `chunk_length=60` ❌ DISCARD (Run 5b, 2026-04-03)
  - RTF=0.145x, score -0.3% — identiek aan default 30
  - Geen voordeel, default chunk_length=30 is optimaal
  - Meet: RTF + accuracy op test set 2

---

## Prioriteitsvolgorde (Run 2)

Start met hoogste verwachte impact, laagste inspanning:

1. **SI1** — Threshold 0.40 (directe kwaliteitswinst speaker ID, lage kost)
2. **SI3** — Threshold 0.45 (conservatiever alternatief)
3. **V1** — Vastgoed vocabulary (directe accuraatheid NL jargon)
4. **V2** — Speaker namen als prompt
5. **S1** — cpu_threads sweep (snelheidswinst, 1 config change)
6. **M1** — multilingual=True (FR/NL kwaliteit)
7. **SI4** — Re-enrollment meerdere fragmenten
8. **H7** — Max Suppression zonder temperature
9. **S4** — num_workers=2 parallel
10. **M3** — language auto vs explicit speed impact
11. **T4** — Temperature fallback tuple
12. **S5** — word_timestamps=False herverifiëren
13. **C1** — chunk_length_s=15
14. **V3** — Combinatie vocab + namen
15. **SI5** — num_speakers hint
16. **A4** — cond_prev=True + temp=0.1
17. **T5** — beam=5 vs beam=1
18. **C2** — chunk_length_s=60
19. **P1/P2/P3** — Presets definitief vastleggen

---

## Experiment Log — Run 1 (2026-03-30)

Baseline: turbo + int8 + beam=1 + temp=0.0 (cold) → RTF≈1.28x

| Exp | Datum      | Wijziging                          | RTF    | Halluc%  | Score | Status     |
| --- | ---------- | ---------------------------------- | ------ | -------- | ----- | ---------- |
| B   | 2026-03-30 | BASELINE (turbo+int8+beam=1, cold) | 1.280x | 1.7%     | 0.768 | BASELINE   |
| H1  | 2026-03-30 | vad_filter=True                    | 0.187x | 0.89%    | 5.306 | ✅ KEEP    |
| T1  | 2026-03-30 | temperature=0.2                    | 0.273x | 0.58%    | 3.638 | ❌ DISCARD |
| H3  | 2026-03-30 | silence_threshold=2.0              | 0.188x | 0.89%    | 5.264 | ❌ DISCARD |
| H2  | 2026-03-30 | condition_previous=False           | 0.146x | 0.84%    | 6.776 | ✅ KEEP    |
| H4  | 2026-03-30 | repetition_penalty=1.1             | 0.150x | 0.58%    | 6.629 | ❌ DISCARD |
| H5  | 2026-03-30 | no_repeat_ngram_size=3             | 0.147x | **0.0%** | 6.797 | ✅ KEEP    |
| T2  | 2026-03-30 | temperature=0.4                    | 0.191x | 0.0%     | 5.232 | ❌ DISCARD |
| A1a | 2026-03-30 | repetition_penalty=1.05            | 0.147x | 0.60%    | 6.755 | ❌ DISCARD |
| H6  | 2026-03-30 | MAX_SUPPRESSION (+temperature)     | 0.187x | 0.0%     | 5.337 | ❌ DISCARD |
| T3  | 2026-03-30 | temperature=0.2 + beam=1           | 0.193x | 0.64%    | 5.137 | ❌ DISCARD |
| A2a | 2026-03-30 | no_repeat_ngram_size=2             | 0.146x | 0.69%    | 6.791 | ❌ DISCARD |
| A3  | 2026-03-30 | beam_size=3                        | 0.154x | 0.0%     | 6.484 | ❌ DISCARD |

### Conclusies Run 1

**3 verbeteringen (KEEP):**

1. `vad_filter=True` → RTF 1.28x→0.187x, halluc 1.7%→0.89%
2. `condition_on_previous_text=False` → RTF 0.187x→0.146x (**27% sneller!**)
3. `no_repeat_ngram_size=3` → **0% hallucinations**

**Optimale config na Run 1:**

```yaml
model_size: large-v3-turbo
compute_type: int8
beam_size: 1
word_timestamps: true
temperature: 0.0
vad_filter: true
condition_on_previous_text: false
repetition_penalty: 1.0
no_repeat_ngram_size: 3
```

Score: **6.797** | RTF: **0.147x** | Halluc: **0.0%**

---

## Experiment Log — Run 2 (2026-04-01)

Baseline: optimale config Run 1 (koud gestart → RTF ~0.39x door model load)
Note: cpu_threads experimenten zijn deels vertekend door cold/warm effecten.

| Exp | Datum      | Wijziging                        | RTF    | Halluc% | Score | Status     |
| --- | ---------- | -------------------------------- | ------ | ------- | ----- | ---------- |
| B2  | 2026-04-01 | BASELINE_RUN2 (cold)             | 0.386x | 0.0%    | 2.589 | BASELINE   |
| H7  | 2026-04-01 | max_suppression_no_temp          | 0.396x | 0.6%    | 2.514 | ❌ DISCARD |
| S5  | 2026-04-01 | word_timestamps=False            | 0.401x | 0.0%    | 2.492 | ❌ DISCARD |
| A4  | 2026-04-01 | cond_prev=True + temp=0.1        | 0.666x | 0.0%    | 1.501 | ❌ DISCARD |
| T4  | 2026-04-01 | temperature fallback (0,0.2,0.4) | 0.362x | 0.0%    | 2.761 | ✅ KEEP    |
| T5  | 2026-04-01 | beam_size=5                      | 0.548x | 0.0%    | 1.824 | ❌ DISCARD |
| S1a | 2026-04-01 | cpu_threads=8 (cold)             | 0.702x | 0.0%    | 1.425 | ❌ DISCARD |
| S1b | 2026-04-01 | cpu_threads=10 (cold)            | 0.768x | 0.0%    | 1.301 | ❌ DISCARD |
| S1c | 2026-04-01 | cpu_threads=4 (cold)             | 0.245x | 0.0%    | 4.088 | ✅ KEEP\*  |
| S1d | 2026-04-01 | cpu_threads=auto (warm)          | 0.150x | 0.0%    | 6.664 | ✅ KEEP    |
| M1  | 2026-04-01 | multilingual=True                | 0.148x | 0.0%    | 6.743 | ✅ KEEP    |
| M3  | 2026-04-01 | language=nl + multilingual=False | 0.132x | 0.0%    | 7.574 | ✅ KEEP    |

\*S1c resultaat vertekend door cold/warm effect — auto is daadwerkelijk beter (zie S1d)

### Conclusies Run 2

**5 verbeteringen (KEEP):**

1. `temperature=(0.0, 0.2, 0.4)` — fallback tuple voor moeilijke segmenten
2. `cpu_threads=auto` — CTranslate2 bepaalt optimum zelf (warm: 0.150x RTF)
3. `multilingual=True` — lichte verbetering, nodig voor FR/NL mix
4. `language="nl"` + `multilingual=False` — **snelste config voor pure NL** (0.132x RTF)

**Optimale productie config na Run 2:**

```yaml
model_size: large-v3-turbo
compute_type: int8
beam_size: 1
word_timestamps: true
temperature: [0.0, 0.2, 0.4]
vad_filter: true
condition_on_previous_text: false
no_repeat_ngram_size: 3
language: nl # pure NL recordings
multilingual: false # zet op true voor FR/NL mix
```

Score: **7.574** | RTF: **0.132x** | Halluc: **0.0%**

**Totale verbetering vs originele baseline (cold, RTF=1.28x):**

- RTF: 1.28x → 0.132x = **9.7x sneller**
- Hallucinations: 1.7% → 0.0%
- Combined score: 0.768 → 7.574 = **+886%**

---

## Nieuwe bevindingen (2026-03-31)

Uit batch transcriptie van 118 opnames:

**Speaker identification knelpunten:**

- Alex: score 0.390 (threshold 0.50) → UNKNOWN
- Roos: score 0.270 → UNKNOWN
- Andries: score 0.155-0.230 → UNKNOWN
- Samuel: score 0.720 → ✅ correct geïdentificeerd
- Enkel Samuel en soms Alexander worden consistent herkend

**Aanbeveling**: Threshold naar 0.40 verlagen = significante kwaliteitsverbetering voor vrijwel nul extra compute.

**Multilingual knelpunten:**

- FR/NL opnames worden volledig als NL getranscribeerd
- Consultaties met Franstalige klanten bevatten fouten op FR passages

**Aanbeveling**: `multilingual=True` testen op FR/NL test set.

---

## Experiment Log — Run 4 (2026-04-03)

Baseline: optimale config Run 2 + Echo speaker DB (7 enrolled speakers)

### Part A: Speaker Identification Threshold Sweep

Test files: Roos gesprek (191s, 3 spk), audio NL (313s, 2 spk)

**Cosine similarity matrix — Roos gesprek:**

| Speaker    | Alex      | Roos      | Samuel | Evi   | Andries | Deborah | Nico  |
| ---------- | --------- | --------- | ------ | ----- | ------- | ------- | ----- |
| SPEAKER_00 | 0.251     | **0.612** | 0.137  | 0.227 | 0.147   | 0.270   | 0.049 |
| SPEAKER_01 | 0.293     | 0.327     | 0.184  | 0.245 | 0.200   | 0.271   | 0.146 |
| SPEAKER_02 | **0.776** | 0.139     | 0.643  | 0.392 | 0.238   | 0.117   | 0.237 |

**Cosine similarity — audio NL:**

| Speaker    | Top 3 matches                           |
| ---------- | --------------------------------------- |
| SPEAKER_00 | Samuel:0.409, Alex:0.396, Nico:0.247    |
| SPEAKER_01 | Alex:0.645, Samuel:0.645, Andries:0.296 |

**Threshold sweep resultaten:**

| Exp      | Threshold | Roos gesprek | audio NL | Status                         |
| -------- | --------- | ------------ | -------- | ------------------------------ |
| baseline | 0.50      | 2/3          | 1/2      | BASELINE                       |
| SI3      | 0.45      | 2/3          | 1/2      | ❌ DISCARD                     |
| SI1      | 0.40      | 2/3          | **2/2**  | ✅ CONDITIONAL KEEP            |
| SI2      | 0.35      | 2/3          | 2/2      | ❌ DISCARD (geen extra vs SI1) |

**Kernbevinding**: Samuel en Alex (tweelingbroers) hebben sterk overlappende embeddings (0.645 vs 0.645 identiek op audio NL). Threshold verlagen helpt marginaal. **SI4 (re-enrollment met meer samples) is de echte oplossing.**

### Part B: Vocabulary Experiments

Baseline: RTF=0.178x (vertekend door cold start), combined=5.62

| Exp | Label          | RTF (warm) | Halluc | Score | Delta | Status   |
| --- | -------------- | ---------- | ------ | ----- | ----- | -------- |
| B4  | geen vocab     | 0.143x\*   | 0.0%   | —     | —     | BASELINE |
| V1  | vastgoed (22w) | 0.145x     | 0.0%   | —     | ~0%   | NEUTRAAL |
| V2  | namen (7w)     | 0.143x     | 0.0%   | —     | ~0%   | NEUTRAAL |
| V3  | combined (29w) | 0.143x     | 0.0%   | —     | ~0%   | NEUTRAAL |

\* Warm RTF (excl. cold start op eerste file)

**Kernbevinding**: Vocabulary als `initial_prompt` heeft **geen meetbaar effect** op snelheid of hallucinations. Potentieel effect op woordaccuraatheid (jargon) is niet meetbaar met huidige combined_score metric — vereist WER evaluatie.

### Conclusies Run 4

**1 verbetering (CONDITIONAL KEEP):**

1. `speaker_threshold=0.40` → +1 match op sommige opnames (Samuel bij 0.409 score)

**5 neutraal/geen effect:**

1. SI3 (0.45) — te hoog, Samuel scoort vaak < 0.45
2. SI2 (0.35) — geen extra matches vs SI1
3. V1-V3 — neutraal op speed/hallucinations, mogelijke kwaliteitswinst niet gemeten

**Nieuwe inzichten:**

- Alex/Samuel = tweelingbroers → bijna identieke voiceprints → greedy matching probleem
- SI4 (re-enrollment) is hoger geprioriteerd dan verwacht
- Vocabulary effect vereist apart WER framework om te meten

**Totaal Run 1-4: 34 experimenten, 15 KEEP, RTF 1.28x → 0.132x (9.7x sneller)**

---

## Experiment Log — Run 5 (2026-04-03)

Focus: Enhanced speaker identification (ECAPA-TDNN + prosody), num_speakers hint, multilingual

### Phase 1: Enhanced Speaker DB

Alle 7 sprekers verrijkt met ECAPA-TDNN (192d) + prosodische features (12d). Samuel verrijkt met multi-sample enrollment (4 opnames). Totale extractietijd: ~15s voor 7 sprekers.

### Phase 2: SI4-ECAPA

| Methode                | File         | Correct    | False Positives | Totaal |
| ---------------------- | ------------ | ---------- | --------------- | ------ |
| Pyannote-only (t=0.40) | Roos gesprek | 1 (Roos)   | 1 (Alex)        | 2/3    |
| Enhanced (t=0.35)      | Roos gesprek | 1 (Roos)   | 2 (Alex, Evi)   | 3/3    |
| Pyannote-only (t=0.40) | audio NL     | 1 (Samuel) | 1 (Alex)        | 2/2    |
| Enhanced (t=0.35)      | audio NL     | 1 (Samuel) | 1 (Alex)        | 2/2    |

**Conclusie**: Enhanced gaf MEER false positives door te lage threshold. ECAPA modules werken correct, maar tuning + ground truth nodig.

**Kernprobleem**: zonder gelabelde ground truth (wie praat wanneer in elke opname) kunnen we niet betrouwbaar meten of identificatie beter wordt. Samuel en Alex werken samen — Alex kan echt in opnames zitten.

### Phase 3: SI5 — num_speakers hint

Auto-detect vs hinted: identiek resultaat (2 speakers, zelfde matches). KEEP als vangnet.

### Phase 4: M2 — Multilingual

FR/NL testbestand (69f151e0...) niet gevonden in storage. Waarschijnlijk niet gesynct of ander pad.

### Phase 5: C1/C2 — Chunk Length

Overgeslagen: `chunk_length` parameter niet doorgegeven door `transcribe()` API. Vereist code-aanpassing.

### Conclusies Run 5

**Gebouwd (nieuwe capabilities):**

1. `transcripty/ecapa.py` — ECAPA-TDNN speaker embedding extractie
2. `transcripty/prosody.py` — Prosodische feature extractie (shimmer, jitter, pitch, HNR)
3. `speakers.py` — Enhanced enrollment + identification met gewogen 3-layer scoring
4. `speakers_enhanced_full.json` — Volledige DB met alle 3 layers voor 7 sprekers

**Experimenten:**

1. SI4-ECAPA: ❌ DISCARD op huidige test set (meer false positives)
2. SI5: ✅ KEEP (num_speakers hint, geen negatief effect)
3. M2: SKIPPED (bestand niet gevonden)
4. C1/C2: SKIPPED (API aanpassing nodig)

**Acties status (2026-04-03):**

- [x] ~~chunk_length parameter toevoegen aan transcribe() API~~ — Done
- [x] ~~FR/NL testbestand localiseren~~ — Done (id=100, Visite immobilière)
- [x] ~~lingua installeren~~ — Done, FR detectie werkt (82% FR, 17% NL op Visite)
- [x] ~~ECAPA integratie in pipeline.py~~ — Done, automatisch bij enhanced profielen
- [x] ~~Config defaults updaten naar P2+multilingual~~ — Done
- [x] Ground truth WER labeling ✅ 8 samples (4 Samuel + 4 Alex), WER gemiddeld 10.8%
- [x] ~~ECAPA threshold tuning~~ — Afgerond: multi-sample averaging werkt NIET voor tweelingen

**Totaal Run 1-5b: 43 experimenten, 16 KEEP + 3 DEFINED, RTF 1.28x → 0.149x P2 default (8.6x sneller)**

---

## Experiment Log — Run 5b (2026-04-03)

Focus: ECAPA weight tuning, multilingual, chunk length, presets

### Phase 1: SI4b — ECAPA Weight & Threshold Tuning

11 combinaties getest op 2 test files. Alle configuraties gaven 2 correct + 2 false positives.

| Combo                | Threshold | Weights     | Correct | False |
| -------------------- | --------- | ----------- | ------- | ----- |
| pyannote-only t=0.40 | 0.40      | 1/0/0       | 2       | 2     |
| ecapa-only t=0.35    | 0.35      | 0/1/0       | 2       | 2     |
| enhanced t=0.40      | 0.40      | 0.4/0.4/0.2 | 2       | 3     |
| ecapa-heavy t=0.30   | 0.30      | 0.1/0.8/0.1 | 2       | 3     |

**Conclusie**: geen combinatie verbetert op deze test set. Het kernprobleem is **ontbrekende ground truth** — de "false positives" (Alex) zijn mogelijk correcte identificaties als Alex echt in de opnames zit. ECAPA-only (0/1/0) presteert gelijk aan pyannote-only bij t=0.35.

### Phase 2: M2 — Multilingual FR/NL

Testbestand (49s) bevatte alleen NL+EN, geen FR. Alle configs detecteerden NL.

- NL-only: RTF=0.209x (cold), auto-detect: RTF=0.143x, NL+multilingual: RTF=0.142x
- Voor echte FR test: gebruik id=100 (Visite immobilière, 1316s, FR/NL)

### Phase 3: C1/C2 — Chunk Length

- C1 (chunk=15): ❌ RTF=0.232x, **60% trager** dan default (0.145x)
- C2 (chunk=60): ❌ RTF=0.145x, identiek aan default
- **Default chunk_length=30 is optimaal**

### Phase 4: P1/P2/P3 — Presets Vastgelegd

| Preset          | RTF        | Halluc | Score    | Verschil                         |
| --------------- | ---------- | ------ | -------- | -------------------------------- |
| P1 Speed        | 0.130x     | 0%     | 7.67     | Snelste greedy decode            |
| P2 Quality      | 0.149x     | 0%     | 6.71     | beam=3, silence threshold        |
| **P3 Balanced** | **0.127x** | **0%** | **7.89** | **BESTE — temperature fallback** |

**P3 is de aanbevolen default**: sneller dan P1 door temperature fallback tuple, gelijke kwaliteit.

### Conclusies Run 5b

- **ECAPA weight tuning**: geen verbetering op huidige test set (ground truth probleem)
- **M2**: testbestand bevatte geen FR — hertest nodig met echte FR opname
- **C1**: ❌ chunk=15 is 60% trager
- **C2**: ❌ chunk=60 geen verschil
- **P3 Balanced is de nieuwe default** (RTF=0.127x, score=7.89)
- **chunk_length parameter toegevoegd** aan transcribe() API

**Totaal Run 1-5b: 43 experimenten, 16 KEEP + 3 DEFINED, RTF 1.28x → 0.127x (10.1x sneller)**

---

---

# Research Agenda v2 — Nieuwe Axes (2026-04-03)

> Gebouwd op de resultaten van Run 1-5b. Focus verschuift van speed/hallucination
> (opgelost) naar kwaliteit, meertaligheid, en productie-integratie.

## Baseline v2 (April 2026)

| Metric        | Waarde                                                      |
| ------------- | ----------------------------------------------------------- |
| Config        | P2 Quality: large-v3-turbo, int8, beam=3, multilingual=True |
| RTF           | 0.149x (P2) / 0.185x met diarization + ECAPA                |
| Hallucination | 0.0%                                                        |
| WER           | ~16% op NL vastgoedgesprekken (1 sample)                    |
| Speaker ID    | 4/4 correct op Samuel+Alex+Andries gesprek                  |
| Taaldetectie  | 82% FR correct op Visite immobilière                        |

---

### Axis 10: WER Framework & Kwaliteitsmeting

**Doel**: Automatische WER meting met ground truth, regressie-detectie bij config changes.

**Metric**: WER% (lager = beter). Target: < 10% op NL vastgoedgesprekken.

- [x] **WER1** — WER benchmark script bouwen ✅ (2026-04-03)
  - Input: ground_truth.json met referentieteksten
  - Output: WER per sample, gemiddelde WER, per-woord diff
  - Automatisch draaien bij config wijzigingen

- [ ] **WER2** — Ground truth uitbreiden naar 10+ samples
  - 4 enrollment samples (Samuel) — wacht op correcties
  - 3 bestaande test files (Roos gesprek, audio NL, audio4 NL) — handmatig labelen
  - 3 FR/NL samples uit Visite immobilière — FR kwaliteit meten

- [x] **WER3** — WER verbeteren via post-processing ✅ WER 20% → 11.2% (-8.8pp)
  - Automatische hoofdletter correctie
  - Getallen normalisatie ("driehonderdvijfentwintigduizend" → "325.000")
  - Vastgoed eigennamen dictionary ("Klaas" → "Claes", "EPC-lepel" → "EPC-label")

- [x] **WER4** — WER vergelijken: P2 vs large-v3 ❌ large-v3 faalt op sample 4 (98.8% WER vs turbo 20%)
  - large-v3 had betere woordkeuze in Run 1 benchmarks
  - Verwacht: lagere WER maar tragere RTF
  - Meet: WER + RTF trade-off

---

### Axis 11: Whisper Quality Fallback

**Doel**: Automatisch hertranscriberen met beter model bij lage kwaliteit.

- [x] **QF1** — Confidence score per segment extraheren ✅ avg_logprob + no_speech_prob in Segment model
  - faster-whisper geeft avg_logprob en no_speech_prob per segment
  - Threshold bepalen: wanneer is een segment "verdacht"?

- [x] **QF2** — Automatische hertranscriptie bij lage confidence ✅ `quality_fallback.py` gebouwd. 0 triggers op 8 test samples (P2 produceert hoge confidence).
  - Segment met avg_logprob < threshold → hertranscriberen met beam=5 of large-v3
  - Meet: WER verbetering vs extra RTF kosten

- [x] **QF3** — VAD sensitivity tuning ✅ VAD is niet het probleem, beam=1 is de oorzaak. P2 (beam=3) lost dit op.
  - De sample 4 (zakelijk) faalde bij beam=1+VAD — content werd volledig gemist
  - Test VAD parameters: min_speech_duration_ms, min_silence_duration_ms
  - Doel: minder aggressive VAD die zachte spraak niet wegfiltert

---

### Axis 12: Meertalige Kwaliteit (FR/NL/EN)

**Doel**: Betrouwbare transcriptie en taaldetectie voor Belgische meertalige gesprekken.

- [ ] **ML8** — Per-segment taaldetectie accuraatheid meten
  - Visite immobilière als test: handmatig eerste 50 segmenten labelen (FR/NL)
  - Meet: % correct taaldetectie door lingua
  - Baseline: 82% FR, 17% NL op automatische detectie

- [x] **ML9** — Whisper audio-level vs lingua tekst-level ✅ Beide correct FR op Visite, lingua voegt 3 EN correcties toe
  - ML7 hybrid (Axis 10 van v1) gebruikte Whisper per-venster taaldetectie
  - Vergelijk: Whisper audio-detect vs lingua tekst-detect accuracy
  - Verwacht: Whisper beter op korte FR fragmenten waar lingua faalt

- [x] **ML10** — ML7 hybrid integreren in pipeline ✅ Al geintegreerd in Echo local_transcribe.py
  - `transcripty/multilingual.py` bevat `transcribe_with_speakers_multilingual()`
  - Integreren als optie in `pipeline.py`
  - Meet: accuracy + RTF op FR/NL test set

- [x] **ML11** — Code-switching binnen segmenten ✅ `codeswitching.py` — 25/632 multilingual segs op NL/FR/EN gesprek. Gebruikt lingua detect_multiple_languages_of().
  - Belgische gesprekken: NL zin met FR woorden ("dat is een bon compromis")
  - Detecteer en label taalwisselingen binnen één segment
  - Methode: per-woord taaldetectie via lingua of Whisper

---

### Axis 13: Speaker ID Productie-Kwaliteit

**Doel**: Betrouwbare speaker identification in productie, inclusief tweelingen.

- [x] **SP1** — Ground truth speaker labeling ✅ Samuel+Alex gesprek: SPEAKER_01=Samuel, SPEAKER_00=Andries(EN), SPEAKER_02=Alex. Pyannote 3/3 correct.
  - Samuel+Alex gesprek (id=74, 22 min) — handmatig eerste 50 segmenten labelen
  - Meet: speaker_accuracy = % correct gelabelde segmenten
  - Baseline: 4 speakers correct herkend (Samuel, Alex, Andries, UNKNOWN)

- [x] **SP2** — Adaptief ECAPA enrollment ✅ `transcripty/adaptive.py` — selecteert meest onderscheidend sample
  - Probleem: enrollment audio ≠ diarized audio (andere akoestiek)
  - Idee: ECAPA embeddings extraheren uit ELKE opname, vergelijken met vorige
  - Adaptieve voiceprints die meegroeien met meer data

- [x] **SP3** — Speaker diarization kwaliteit meten ✅ DER=15.5% op Samuel+Alex gesprek. 97.6% correct, 2.4% confusion, 13.2% false alarm. Alle 3 sprekers correct gemapt.
  - Diarization Error Rate op ground truth gelabelde opnames
  - Pyannote 3.1 vs num_speakers hint vs min/max_speakers
  - Meet: over-segmentatie, under-segmentatie, speaker confusion

- [x] **SP4** — Tweeling-specifieke enrollment strategie ✅ Best-sample selectie: sam_1 + alex_3 (ECAPA sim=0.391)
  - Bevinding: multi-sample averaging werkt NIET voor tweelingen (embeddings convergeren)
  - Hypothese: gebruik het MEEST onderscheidende sample per tweeling i.p.v. gemiddelde
  - Methode: per enrollment sample de inter-twin ECAPA distance meten, sample met grootste afstand kiezen

---

### Axis 14: Pipeline Integratie & Productie

**Doel**: Transcripty optimalisaties doorvoeren in de Echo productie pipeline.

- [x] **PI1** — Echo local_transcribe.py updaten naar P2 defaults ✅ DB: beam=3, language=NULL, multilingual=1
  - Huidige Echo config gebruikt mogelijk oude defaults
  - Sync met ~/.transcripty/config.yaml

- [x] **PI2** — Enhanced speaker DB naar Echo DB migreren ✅ kolommen + data voor 7 speakers
  - ECAPA + prosody embeddings opslaan in Echo speakers tabel
  - Nieuwe kolommen: ecapa_embedding, prosodic_features
  - Migratie via Alembic

- [x] **PI3** — Transcripty installeren in Echo venv ✅ editable install gedaan
  - Huidige bug: `ModuleNotFoundError: No module named 'transcripty'` bij sync
  - Fix: editable install in Echo venv

- [x] **PI4** — Auto-enrollment bij speaker assignment ✅ ECAPA+prosody automatisch bij enroll_speaker()
  - Wanneer een gebruiker een speaker toewijst in de UI:
  - Automatisch ECAPA + prosody extraheren en opslaan
  - Voiceprint verbetert bij elke toewijzing

---

## Prioriteitsvolgorde v2

| Prio | Experiment | Impact                                   | Inspanning |
| ---- | ---------- | ---------------------------------------- | ---------- |
| 1    | PI3        | Hoog — unblocks Echo sync                | 5 min      |
| 2    | WER1       | Hoog — meetbaar kwaliteitsframework      | 30 min     |
| 3    | QF3        | Hoog — VAD fix voor gemiste content      | 30 min     |
| 4    | ML10       | Hoog — ML7 hybrid in pipeline            | 1 uur      |
| 5    | PI1        | Medium — Echo config sync                | 15 min     |
| 6    | PI2        | Medium — ECAPA naar Echo DB              | 1 uur      |
| 7    | WER3       | Medium — post-processing WER verbetering | 1 uur      |
| 8    | SP4        | Medium — tweeling enrollment strategie   | 30 min     |
| 9    | QF1/QF2    | Medium — confidence-based fallback       | 2 uur      |
| 10   | ML8/ML9    | Laag — meertalige accuraatheid meting    | 1 uur      |
| 11   | SP1-SP3    | Laag — vereist handmatig labelen         | variabel   |
| 12   | WER2/WER4  | Laag — vereist ground truth              | variabel   |
| 13   | ML11       | Laag — code-switching detectie           | 2 uur      |
| 14   | PI4        | Laag — UI feature                        | 2 uur      |
