# fNIRS High-Frequency Supine Dataset

> **Terms of use:** this dataset may be used only during the Heroes of the Brain Hackathon (13–15 November 2026).
> To use it after the hackathon, please ask us at [info@cortivision.com](mailto:info@cortivision.com). We are happy
> to support you if you would like to continue your project, or start a new one, after the hackathon.

fNIRS recordings of 60 adult volunteers lying supine, made with a sparse prefrontal montage sampled at a high rate
(141.79 Hz, Cortivision Spectrum C23). Each participant performed a 10 min **resting baseline (BS)** followed by a
**breath-holding task (BH)**, recorded as separate SNIRF files.
In total there are 118 files: 59 baseline and 59 BH; 58 participants have both.

## Participants

60 participants (31 female, 29 male), age 26.7 ± 13.0 years (range 18–78). 56 participants are aged 18–42
(23.5 ± 4.7 years) and 4 are aged 68–78.
Participant 41 has no baseline file, and participant 59 has no BH file.

Participant characteristics are in `subjects_description.csv`, one row per participant. `Nr osoby badanej` matches
the participant number in the file names. The column names and some values are in Polish, and decimals use a comma
(e.g. `57,5`):

| Column | Meaning |
|---|---|
| `Nr osoby badanej` | participant number |
| `Płeć` | sex (`Kobieta` = female, `Mężczyzna` = male) |
| `Wiek`, `Wzrost`, `Waga` | age [years], height [cm], weight [kg] |
| `Kolor włosów`, `Typ włosów`, `Tekstura włosów` | hair colour (`brązowe` brown, `czarne` black, `siwe` grey, `brak` none), type and texture |
| `Fitzpatrick` | Fitzpatrick skin type |
| `Rozmiar głowy`, `Rozmiar czepka` | head circumference [cm], cap model |
| `Przyjmowane leki` | medication (`-` = none) |
| `Ocena aktywności fizycznej` | physical activity score |
| `Prędkość bieżni`, `Grupa` | treadmill speed and counterbalancing group for the mental calculation tasks (not part of this dataset) |

Medication that may affect haemodynamics: participants 48, 49, 59 and 60 (aged 68–78) take antihypertensive drugs
(`od nadcisnienia`), and participant 38 takes nebivolol (`nebilet`), a beta-blocker.

## Experiments

Participants lay supine on a mattress with their eyes open. Baseline and BH were recorded one after the other in
the same visit, with the same cap.

### BS – resting baseline (1 per participant)

Participants rested quietly and breathed freely for about 10 min. There are no markers.

```
| rest (579–749 s) |
```

### BH – breath holding (1 per participant)

Participants performed 3 cycles of about 1 min of free breathing followed by a maximal breath hold. Each hold
started at the end of an exhalation and lasted as long as the participant could manage.

```
| free breathing (~70 s) | hold | free breathing (~70 s) | hold | free breathing (~70 s) | hold | until recording end |
```

Holds with a marked end lasted 5–106 s (median 21.5 s), and free-breathing periods lasted 63–89 s. The recording ends 14–90 s
after the onset of the last hold.

## Markers

Each marker gives the onset of the event. In MNE they appear as annotation descriptions `'1'` and `'2'`.

| Task | Event | Code | Per file |
|---|---|---|---|
| BS | – | – | 0 |
| BH | Free breathing starts (= end of the preceding hold) | 1 | 3 |
| BH | Breath hold starts | 2 | 3 |

In 12 BH files (participants 22, 23, 24, 26, 36, 41, 44, 46, 48, 49, 53 and 54), the first `1` marker is missing.
These recordings start during free breathing, and the first hold begins 68–74 s after the start.

## File organization

```
fnirs_high_freq_supine/
  DATASET_DESCRIPTION.md     this file
  subjects_description.csv   participant characteristics
  data/
    baseline/   <ID>_<INI>_BS_<date-time>.snirf                     (59 files)
    BH/         Breath holding_V2_<ID>_<INI>_BH_<date-time>.snirf   (59 files)
```

`<ID>` is the participant number without a leading zero, `<INI>` the initials and `<date-time>` the recording start
(`YYYY-MM-DD-hh-mm-ss`, clock time). Three BH files carry `BS` or `BH2` instead of `BH` in the name
(participants 25, 26 and 47). They are BH recordings like the others.

## Data format and montage

| Property | Value |
|---|---|
| Format | SNIRF 1.1 (HDF5) |
| Device | Cortivision Spectrum C23 (Cortiview software) |
| Signal | raw continuous-wave light intensity, no preprocessing |
| Sampling rate | 141.79 Hz |
| Channels | 20 = 10 source–detector pairs × 2 wavelengths (760 and 850 nm) |
| Optodes | 8 sources, 5 detectors; 2D and 3D positions plus 130 landmarks of the 10-10 system in `probe` |
| Source–detector distance | 8 long channels (23–30 mm) and 2 short channels (10 mm: S7–D4, S8–D5) |
| Coverage | prefrontal cortex (optodes around Fp1, Fpz, Fp2, AF7, AF8 and F1–F4) |
| Auxiliary | 3-axis accelerometer and 3-axis gyroscope (`aux1`–`aux6`), same time base as fNIRS |
| Time | each file starts at t = 0 s; clock time of the start in `metaDataTags/MeasurementDate` and `MeasurementTime` |

## Notes for analysis

* The data are raw intensity. Convert them to optical density and haemoglobin concentration yourself.
* The end of each hold is marked by the next `1`, except for the last hold in every file. Finding the end of the
  last hold is part of the analysis. Look in the signals (fNIRS haemodynamics, accelerometer and gyroscope) and/or
  in the breath-holding literature for a biomarker of the hold's end.
* Hold durations differ widely between participants (5–106 s).
* Every file contains a few samples flagged in `metaDataTags/missing_sample`, usually less than 0.5%. Files above 1%:
  * baseline: `45_AW` (7.7%), `48_DK` (2.6%), `14_KZ` (2.1%), `27_BS` (2.1%), `26_MS` (1.5%), `10_PL` (1.5%),
    `28_AG` (1.4%);
  * BH: `25_AB` (3.7%), `10_PL` (1.5%).
* Marker durations are stored as a 1 s placeholder. Use only the onsets.

## Recording lengths

| Task | Files | Mean ± SD [s] | Range [s] |
|---|---|---|---|
| BS | 59 | 629.2 ± 32.7 | 579.3–749.2 |
| BH | 59 | 301.0 ± 49.8 | 243.4–505.3 |
