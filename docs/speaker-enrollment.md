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

## Which language?

Voice embeddings are largely **language-independent** — they capture vocal characteristics like pitch, timbre, and rhythm, not the words themselves. However, intonation and speaking patterns differ between languages, so enrollment quality is best when the language matches your typical recordings.

**Recommendation:**

- Enroll in the language you speak most often in recordings
- If you regularly speak multiple languages, enroll once per language and use the best match
- A single enrollment in your primary language works well in most cases

## Read-aloud texts for enrollment

Each text takes approximately 45-60 seconds at normal speaking pace. Choose the language that matches your recordings.

### Nederlands

> De ochtend begon met regen die tegen het keukenraam tikte. Ik schonk een kop koffie in en ging aan tafel zitten, me afvragend wat de dag zou brengen.
>
> Mijn buurvrouw had een briefje op de deur geplakt. Er stond: "Kun je donderdag de planten water geven? Ik ben tot het weekend in Kopenhagen." Ik nam me voor om te kijken of ik haar reservesleutel nog ergens in de la had liggen.
>
> Later die middag ging ik naar de markt. Het lijstje was kort: vier tomaten, een brood, olijfolie, en als ze het hadden, verse basilicum. De verkoper op de hoek herkende me meteen. "Alweer hier?" vroeg hij met een grijns. Ik knikte en zocht de rijpste tomaten uit die ik kon vinden.
>
> Op weg naar huis liep ik langs de oude boekhandel in de Kerkstraat. Er stond een nieuwe etalage, iets over poolexpedities. Ik dacht erover om naar binnen te gaan, maar mijn tas zat al vol. Misschien volgende week.
>
> Die avond belde ik mijn broer. We praatten over de gewone dingen: werk, het weer, of onze ouders de nieuwe afstandsbediening al onder de knie hadden. Hij vertelde een verhaal over zijn kat die op het dak was geklommen en drie uur lang weigerde om naar beneden te komen. Ik moest zo hard lachen dat ik bijna mijn thee morste.
>
> Voor het slapengaan las ik nog een paar bladzijden van de roman op mijn nachtkastje. Het verhaal werd spannend: een detective in Lissabon, een verdwenen schilderij, en een verdachte die steeds haar verhaal veranderde. Ik legde een bladwijzer en deed het licht uit.

### English

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

### Francais

> La matinee a commence avec la pluie qui tambourinait contre la fenetre de la cuisine. Je me suis servi une tasse de cafe et je me suis assis a table, me demandant ce que la journee allait apporter.
>
> Ma voisine avait laisse un mot sur la porte. Il disait : "Pourrais-tu arroser les plantes jeudi ? Je serai a Copenhague jusqu'au week-end." Je me suis fait une note mentale pour verifier si j'avais encore sa cle de rechange quelque part dans le tiroir.
>
> Plus tard dans l'apres-midi, je suis alle au marche. La liste etait courte : quatre tomates, une miche de pain au levain, de l'huile d'olive et, s'ils en avaient, du basilic frais. Le vendeur du stand du coin m'a reconnu. "Encore de retour ?" a-t-il demande avec un sourire. J'ai hoche la tete et j'ai choisi les tomates les plus mures que j'ai pu trouver.
>
> En rentrant a pied, je suis passe devant la vieille librairie de la rue de l'Orme. Il y avait une nouvelle vitrine, quelque chose sur les expeditions polaires. J'ai pense a y entrer, mais mon sac etait deja plein. Peut-etre la semaine prochaine.
>
> Ce soir-la, j'ai appele mon frere. Nous avons parle des choses habituelles : le travail, la meteo, et si nos parents avaient enfin compris la nouvelle telecommande. Il m'a raconte une histoire sur son chat qui etait monte sur le toit et avait refuse de descendre pendant trois heures. J'ai tellement ri que j'ai failli renverser mon the.
>
> Avant de me coucher, j'ai lu quelques pages du roman sur ma table de nuit. L'intrigue devenait interessante : un detective a Lisbonne, un tableau disparu et une suspecte qui changeait sans cesse sa version des faits. J'ai marque la page et j'ai eteint la lumiere.

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
