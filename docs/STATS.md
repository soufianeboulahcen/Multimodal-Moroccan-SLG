# MoSL dataset — statistics & implications for the SignLLM-style model

Computed from `data/labels.csv` and `data/video_meta.csv` (ffprobe).

## Counts

| Category              | Clips | Unique signs (NFC) |
|-----------------------|------:|-------------------:|
| Diverse               | 1,941 |              1,508 |
| Numbers               |   130 |                 51 |
| Letters               |    71 |                 39 |
| days_months_seasons   |    59 |                 23 |
| Pronouns              |    15 |                 10 |
| **Total**             | **2,216** | **1,631** |

> The MoSL paper reports 2,199 clips. We extract **2,216** from `vedios-dataset.zip`.
> Discrepancy is small (+17) and unexplained — likely a versioning difference between
> the published Mendeley snapshot and the zip we received. Not addressed at this stage.

## Diacritic-only signs are real

15 clips have a single Arabic diacritic mark as their entire label
(ً ٌ ٍ َ ُ ِ ّ ْ — fatḥatān, ḍammatān, kasratān, fatḥa, ḍamma, kasra, shadda, sukun).
These are **legitimate MoSL signs for the diacritics themselves**, not data bugs.
Implication: do NOT use diacritic-stripped Arabic as the dedup key for signs.
Use NFC-normalized raw Arabic. `word_arabic_stripped` in `labels.csv` is retained
only as a secondary fuzzy-lookup field for user input that omits diacritics.

## Stripped-form collisions = real minimal pairs

17 stripped-Arabic forms each map to ≥2 distinct NFC signs. Examples:

| Stripped | NFC pair                  | English meanings                |
|----------|---------------------------|---------------------------------|
| رجل      | رَجُلٌ ↔ رِجْلٌ          | man ↔ leg                       |
| علم      | عَلَمٌ ↔ عِلْمٌ          | flag ↔ knowledge                |
| ذهب      | ذَهَبٌ ↔ ذَهَبَ          | gold (noun) ↔ he went (verb)    |
| درس      | دَرَسَ ↔ دَرْسٌ          | he studied ↔ lesson             |

Implication: input handling must accept diacritics (or do morphological
disambiguation) — collapsing diacritics turns these into homographs at inference.

## Clips per sign (the long tail)

```
 1 clip:  1,201 signs   ████████████████████████████████████████████████
 2 clips:   318 signs   ████████████
 3 clips:    84 signs   ███
 4 clips:    17 signs   █
 5 clips:     7 signs
 6 clips:     4 signs
```

**74% of signs have only one clip.** This drives the split strategy.

## Video metadata

- **Duration:** median 3.72s, mean 3.96±1.19s, max 9.44s.
- **Frame count:** median 93, mean 100.6±32, max 236.
- **FPS:**
  - 25 fps:   2,061 clips (paper-stated standard)
  - 30 fps:     154 clips (anomalous — different recording session?)
  - 24.75 fps:    1 clip
- **Resolution:** 460×460 per the paper (not re-verified).

> **Required preprocessing**: resample all clips to 25 fps before pose extraction,
> so the compressed pose tensors are frame-rate-consistent (SignLLM/Prompt2Sign
> assumes uniform fps).

## Splits

Strategy in `src/data/split.py`:
- 1-clip signs   → all to train
- 2-clip signs   → 1 train + 1 val
- 3+-clip signs  → 1 val + 1 test, rest to train

| Split | Clips | % |
|-------|------:|--:|
| train | 1,674 | 75.5% |
| val   |   430 | 19.4% |
| test  |   112 |  5.1% |

**Every val/test sign appears in train.** Held-out clips test the realistic
question: *"reproduce a known sign in a new variant"*. We cannot test true
out-of-vocabulary generalization with this data.

## Open issues for later phases

1. **No signer IDs in filenames.** Cannot do speaker-disjoint splits, so val/test
   may include the same signer as train. The MoSL paper says 9 signers contributed
   but the metadata does not propagate to the files. Worth contacting the authors
   if signer-disjoint evaluation is required for the PFE.
2. **Single-clip dominance.** With 1,201 signs having only one example, the model
   has zero variation to learn signer/style invariance for those signs.
3. **30 fps subset.** Need to confirm the resampling target (25 vs 30) matches
   what Prompt2Sign expects in the compressed format.
