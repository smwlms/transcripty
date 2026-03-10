# Speaker Enrollment Guide

To identify speakers by name, you first need to enroll their voice by recording a reference audio sample. This page explains how to create a good enrollment recording and provides a ready-to-use read-aloud text.

## How enrollment works

1. Record a **30-60 second** audio clip of the speaker reading aloud
2. The audio should contain **only one speaker** (no background voices)
3. Transcripty extracts a voice embedding (a numerical fingerprint of the voice)
4. The embedding is stored in a `SpeakerDB` for future identification

```python
from transcripty import SpeakerDB

db = SpeakerDB()
db.enroll("Alice", "alice_reference.wav", hf_token="hf_...")
db.enroll("Bob", "bob_reference.wav")
db.save("speakers.json")
```

## Tips for a good recording

- **Quiet environment** — avoid background noise, music, or other speakers
- **Natural pace** — speak normally, don't rush or whisper
- **Consistent volume** — hold the microphone at a steady distance
- **30-60 seconds** — shorter is unreliable, longer doesn't help much
- **Varied intonation** — the text below includes questions, lists, and statements to capture your full vocal range

## Read-aloud text for enrollment

Read the following text at your normal speaking pace. It takes approximately 45-60 seconds.

---

> The morning began with rain drumming against the kitchen window. I poured myself a cup of coffee and sat down at the table, wondering what the day would bring.
>
> My neighbor had left a note on the door. It said: "Could you water the plants on Thursday? I'll be in Copenhagen until the weekend." I made a mental note to check if I still had her spare key somewhere in the drawer.
>
> Later that afternoon, I went to the market. The list was short: four tomatoes, a loaf of sourdough bread, olive oil, and if they had it, fresh basil. The vendor at the corner stall recognized me. "Back again?" he asked with a grin. I nodded and picked out the ripest tomatoes I could find.
>
> On the walk home, I passed the old bookshop on Elm Street. There was a new display in the window — something about polar expeditions. I thought about going in, but my bag was already full. Maybe next week.
>
> That evening, I called my brother. We talked about the usual things: work, the weather, whether our parents had figured out the new remote control yet. He told me a story about his cat climbing onto the roof and refusing to come down for three hours. I laughed so hard I nearly spilled my tea.
>
> Before bed, I read a few pages of the novel on my nightstand. The plot was getting interesting — a detective in Lisbon, a missing painting, and a suspect who kept changing her story. I marked the page and turned off the light.

---

## After enrollment

Once speakers are enrolled, use the database with `transcribe_with_speakers()`:

```python
from transcripty import transcribe_with_speakers, SpeakerDB

db = SpeakerDB.load("speakers.json")

segments = transcribe_with_speakers(
    "meeting.mp3",
    hf_token="hf_...",
    speaker_db=db,
)

for seg in segments:
    print(f"{seg.speaker}: {seg.text}")
# Alice: Good morning, shall we start?
# Bob: Yes, let's go through the agenda.
```

## Updating profiles

To re-enroll a speaker (e.g. with a better recording), simply call `enroll()` again with the same name — it overwrites the previous profile:

```python
db = SpeakerDB.load("speakers.json")
db.enroll("Alice", "alice_better_recording.wav", hf_token="hf_...")
db.save("speakers.json")
```

## Identification threshold

The default similarity threshold is `0.5`. If you get false matches, increase it. If speakers aren't being recognized, lower it:

```python
names = db.identify(diarization_result, threshold=0.6)  # stricter
names = db.identify(diarization_result, threshold=0.4)  # more lenient
```
