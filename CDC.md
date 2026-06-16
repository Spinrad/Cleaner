# CDC — cleaner v0.1.2

**Statut :** Spec (intention) — le code peut diverger, les divergences sont documentées.
**Version :** 0.1.2
**Dernière mise à jour :** 2026-06-16

---

## 1. Vision

Cleaner utilise une architecture **hybride ffmpeg + LSP/LV2**, où :

- Les **étages structurels** (décodage, resample, HP, encodage/décodage M/S,
  sidechain ducking, mesure LUFS, re‑limiteur post‑LUFS) restent en
  **ffmpeg natif**.
- Les **étages de coloration** (expander, notches EQ + air,
  de‑harsher, bus compressor, limiteur musical) utilisent
  des **plugins LSP au format LV2**, chargés via le filtre `lv2` natif
  de ffmpeg.
- La **saturation** est native ffmpeg (`asoftclip=type=tanh`), pilotée en
  drive+makeup avec oversampling 4x.
- Le rendu s'effectue en **single‑pass `filter_complex`** — aucun render
  intermédiaire, aucune dépendance à Carla ou JACK.
- L'installation des plugins LSP est la responsabilité de l'utilisateur
  final. Cleaner détecte leur présence au démarrage et échoue proprement
  si requis, avec `--force-native` comme échappatoire.

## 2. Prérequis

- Python ≥ 3.11
- ffmpeg ≥ 5.0, **compilé avec `--enable-lv2`**
- LSP plugins LV2 installés (`lsp-plugins-lv2` v1.2.12 sur Debian/Ubuntu)

## 3. Architecture cible

### 3.1 Chaîne complète (ordre exact)

```
[0:a]aresample=48000,highpass=f=35:t=o:p=2
│
├─ 1. [LSP] Expander (gentle relief)      expander_stereo
│      Mode=Up. Désactivé par défaut (--expander).
│
├─ 2. [NATIF] M/S encode                  stereotools=mode=lr>ms
│
├─ 3. [NATIF] Sidechain ducking           channelsplit + HP 150 Hz +
│      (Mid → Side)                        sidechaincompress + amerge
│
├─ 4. [NATIF] M/S decode                  stereotools=mode=ms>lr
│
├─ 5. [LSP] De‑harsher (opt‑in)           sc_compressor_stereo
│      Bande 2.5-4.5 kHz. Désactivé par défaut (--deharsher).
│
├─ 6. [LSP] Parametric EQ                 para_equalizer_x16_stereo
│      Notches (Bell, ft=1) + Air (Bell 10 kHz) + Clean‑mediums (Bell 600 Hz)
│
├─ 7. [NATIF] Saturator                   asoftclip=type=tanh
│      Drive ×12 dB max, makeup −drive×0.4, oversample 4x.
│
├─ 8. [LSP] Bus Compressor                compressor_stereo
│      SSL‑style, ratio 2:1, parallèle. Désactivé par défaut (--bus-comp).
│
├─ 9. [LSP] Limiter                       limiter_stereo
│      True‑peak, oversampling, ALR. lk/at/rt = minimums port (0.1/0.25/0.25 s).
│
├─ 10. [NATIF] Mesure LUFS                ebur128 + volume
│      Gain clampé [−6, +14] dB.
│      Note : un gain élevé (+14 dB) suivi du re‑limiteur écrase la dynamique,
│      en tension avec la philosophie restauration. Documenté honnêtement.
│
└─ 11. [NATIF] Re‑limiteur sécurité       alimiter
       Filet post‑LUFS, ceiling = --ceiling. Vérifié (returncode + existence).
```

### 3.2 Règles d'intégrité

1. **Single‑pass** : tout tient dans UN `filter_complex`. Aucun render intermédiaire.
2. **Aucun sidechain externe LSP** : le seul traitement sidechain est le ducking Mid→Side natif.
3. **Suivi de niveau heuristique** : un `GainTracker` estime le niveau (peak/RMS) après chaque
   étage. C'est un registre de gain linéaire — il ne modélise pas les non‑linéarités
   de compression/saturation ni les changements de crest factor. Les niveaux réels
   doivent être mesurés sur le fichier de sortie.
4. **Échappement robuste** : les URIs LSP contiennent `:` et les contrôles
   utilisent `|` — le builder produit un graphe syntaxiquement valide pour ffmpeg.

### 3.3 Deux builders

- **`lsp_chain_builder.py`** : construit le graphe avec nœuds `lv2`.
  Si un plugin LSP requis est absent → **échec propre** avec message d'installation.
- **`ffmpeg_chain.py`** : construit le graphe full‑natif.
  Invoqué via `--force-native`.
- **Pas de fallback natif silencieux par étage.**
- Les deux builders partagent les constantes musicales (`SAT_DRIVE_MULTIPLIER=12.0`,
  `SAT_MAKEUP_RATIO=0.4`) et le filtre air (Bell 10 kHz, Q=2.0).

## 4. Plugins LSP — mapping et contraintes

### 4.1 URIs

Format : `http://lsp-plug.in/plugins/lv2/<slug>`

| Étage | Slug | Rôle |
|-------|------|------|
| Expander | `expander_stereo` | Gentle relief, Mode=Up |
| EQ | `para_equalizer_x16_stereo` | Notches + air + clean‑mediums |
| De‑harsher | `sc_compressor_stereo` | Bande unique, réduction de dureté |
| Compressor | `compressor_stereo` | Glue/bus, modes multiples |
| Limiter | `limiter_stereo` | True‑peak, ALR |

### 4.2 Unités LSP

LSP exprime les gains en **multiplicateur linéaire G** (1.0 = unité = 0 dB), pas en dB.
Les temps sont en **ms** pour expander/compressor, en **secondes** pour le limiter.
Le code applique la conversion via `clamp_to_port()`.

- G = 10^(dB/20)
- dB = 20 × log10(G)

### 4.3 Découverte des ports

Source de vérité : `lv2info <uri>` et/ou `ffmpeg -f lavfi -i anullsrc -af lv2=p='<uri>':c=help -f null -`.
L'introspection est **cachée** (JSON sur disque) et **rafraîchie si la version de LSP change**.
Le cache est mémoïsé en mémoire processus (1 lecture disque, 1 vérification version).

### 4.4 Ports réels — confirmés par introspection LSP 1.2.12

**Types de filtre EQ** (`ft_{i}`, 0-11) :
```
0=Off  1=Bell  2=Hi-pass  3=Hi-shelf  4=Lo-pass  5=Lo-shelf
6=Notch  7=Resonance  8=Allpass  9=Bandpass  10=Ladder-pass  11=Ladder-rej
```

**EQ** (`para_equalizer_x16_stereo`) — 16 bandes :
- `ft_{i}` : filter type (enum, voir ci-dessus)  ← source : TTL LSP 1.2.12
- `f_{i}` : frequency (Hz, 10–24000)
- `w_{i}` : filter width (0–12, paramètre primaire de bande passante)
- `g_{i}` : gain (G linéaire, 0.01585–63.096)
- `q_{i}` : Q factor (0–100, dérivé de w, piloté pour cohérence)
- `fm_{i}` : filter mode (0=RLC BT, 1=RLC MT, 2=BWC BT, 3=BWC MT, 4=LRX BT, 5=LRX MT, 6=APO DR)

**Expander** (`expander_stereo`) :
- `em` : mode (0=Down, 1=Up) → 1
- `al` : attack level / threshold (G linéaire, 0.001–1)
- `er` : ratio (1–100)
- `at` : attack time (**ms**, 0–2000)
- `rt` : release time (**ms**, 0–5000)
- `kn` : knee (G linéaire, 0.063–1)
- `mk` : makeup gain (G linéaire)
- `sla` : sidechain lookahead (**s**, 0–20)

**Compressor** (`compressor_stereo`) :
- `cm` : mode (0=Down, 1=Up, 2=Boost) → 0
- `al`, `at`, `rt`, `kn`, `mk` : idem expander (temps en **ms**)
- `cr` : ratio (1–100)
- `cdr`/`cwt` : dry/wet gain (G linéaire, 0–10)
- `sla` : sidechain lookahead (**s**, 0–20)

**Limiter** (`limiter_stereo`) :
- `th` : threshold (G linéaire) → ceiling
- `lk` : lookahead (**s**, 0.1–20)
- `at`, `rt` : attack/release (**s**, 0.25–20)
- `ovs` : oversampling (0–20)
- `alr` : adaptive release (bool, 0–1)

**De‑harsher** (`sc_compressor_stereo`) :
- Ports du compresseur (temps en **ms**) + `sct` (sidechain type), `shpf`/`slpf` (Hz)

**Règle de résolution d'unité** (implémentée dans `lv2_params.py` + `lv2_introspect.py`) :
1. Table `PLUGIN_UNITS[(slug, symbole)]` — autorité, consommée du présent CDC
2. Table `EXPLICIT_UNITS[symbole]` — symboles non‑ambigus (enum, ratio, Hz...)
3. `_infer_unit()` — fallback heuristique (max_v > 20 → ms, ≤ 20 → s)

## 5. Paramétrage par étage (spécification)

### 5.1 Analyse globale

Mesures effectuées à **16 kHz** (resample scipy) pour les modes/onsets/dynamique.
Le rendu final est à **48 kHz**. Le delta de SR peut sur/sous‑estimer le peak de
quelques dixièmes de dB. Les seuils (expander, ducking, bus‑comp) sont calés sur
ces mesures 16 kHz — cohérents avec le son validé, mais consciemment décalés du
rendu 48 kHz. Une mesure à 48 kHz est documentée comme piste future.

### 5.2 GainTracker (module `gain_tracking.py`)

```python
class GainTracker:
    def __init__(self, initial_peak_dbfs, initial_rms_dbfs)
    def predict_after(stage_gain_db) -> tuple[float, float]
    def commit(stage_name, gain_db)
```

Registre de gain linéaire — ne modélise pas les non‑linéarités ni le crest factor.
Les valeurs commit sont dérivées des paramètres réels (somme des gains notch,
ΔRMS saturateur estimé, réduction compresseur proportionnelle à bus_comp).
Utilisé comme heuristique grossière pour le seuil du bus‑comp.

### 5.3 Expander (gentle relief, off par défaut)

- **Mode** = Up (expansion vers le haut)
- **Seuil** ← peak_db − 3 dB (très proche du sommet, n'expande que les crêtes)
- **Ratio** ← clamp(1.6 − crest×0.03, 1.1, 1.5) × intensity
- **Effet** : ~0.4 dB de boost crête sur captation AGC typique. Volontairement
  conservateur — respect des dynamiques originales. Activé via `--expander`.
- Pénalité d'écrêtage : ratio × 0.6 si `is_heavily_clipped`.

### 5.4 EQ (notches + air + clean‑mediums)

- **Notches** : Bell (ft=1) avec gain négatif aux fréquences des room modes.
  Profondeur = −(proéminence × 0.5), clampée [−9, −2] dB.
  Si proéminence < 3 dB → bande désactivée. Max 3 bandes.
- **Air** : Bell (ft=1) à 10 kHz, Q=2.0, gain = `--air`.
  Positif = boost (plus brillant), négatif = cut (plus sombre).
  Défaut = 0.0 (neutre). Position : dans le même nœud EQ, avant saturation.
- **Clean‑mediums** : Bell (ft=1) à 600 Hz, Q=1.5, gain = `--clean-mediums`.
  Négatif uniquement (cut dans le bas‑médium).

### 5.5 De‑harsher (opt‑in)

- **Bande** : 2.5–4.5 kHz (filtres internes shpf/slpf)
- **Seuil** ← harshness_index (décorrélation HF × ratio d'énergie)
- **Ratio** ← 1.5–3.0, modéré
- Placé **avant** le saturateur. Activé via `--deharsher`.

### 5.6 Saturator

- **Drive (dB)** ← `eff_glue × 12.0` où `eff_glue = glue × (0.3 + intensity × 0.7)`
  - `glue=0` → 0 dB, `glue=0.5` → +4–6 dB, `glue=1.0` → +12 dB
- **Makeup (dB)** ← `−drive × 0.4` (compensation automatique)
- **Seuil tanh** ← `0.92 − eff_glue × 0.35`
- **Oversampling** : 4x
- Pénalité d'écrêtage : seuil relevé de +0.05 si `is_heavily_clipped`.

### 5.7 Bus Compressor

- **Mode** = Down (compression vers le bas)
- **Seuil** ← RMS prédit − crest×0.3 + (1−bus_comp)×12
- **Ratio** ← 2:1 (SSL classique)
- **Attack** = 10 ms, **Release** = 100 ms
- **Mix** ← bus_comp (compression parallèle dry/wet)

### 5.8 Limiter

- **Mode** = true‑peak avec oversampling
- **Seuil** = ceiling (`--ceiling` en G linéaire)
- **lk/at/rt** = 0.1/0.25/0.25 s (minimums du port LSP)
- **ALR** (adaptive release) activé
- **Post‑LUFS re‑limiteur** : `alimiter` natif, returncode vérifié,
  fichier restauré en cas d'échec.

### 5.9 Macro `--intensity`

Appliquée différemment selon l'étage :
- **Saturator drive** : `intensity` via `0.3 + intensity × 0.7`
- **Expander ratio** : `intensity` appliqué linéairement au ratio au‑dessus de 1.0
- **Notch depth** : `intensity` appliqué linéairement à la profondeur

Intensity=0 → effet nul, intensity=1 → effet maximal.

### 5.10 Séparation analyse/pipeline

Les paramètres sont calculés dans deux contextes partageant les constantes communes :
- `compute_ffmpeg_params()` : alimente la chaîne native (`--force-native`)
- `compute_*_lsp_params()` : alimentent la chaîne LSP (défaut)

Les constantes partagées (`SAT_DRIVE_MULTIPLIER`, `SAT_MAKEUP_RATIO`) sont
définies dans `global_analysis.py` et utilisées par les deux chemins.

## 6. Builders

### 6.1 `lsp_chain_builder.py`

Construit le graphe `filter_complex` avec nœuds `lv2=...` pour les étages de coloration.
Fonction `build_lv2_node(uri, params)` → `"lv2=p='<uri>':c=sym1=val1|sym2=val2"`

### 6.2 `ffmpeg_chain.py`

Builder full‑natif, invoqué via `--force-native`.

## 7. Stratégie de build et fallback

```
Au démarrage :
  1. Détection LSP : lsp_available = (lv2ls renvoie les URIs requises)
  2. Si --force-native → utiliser ffmpeg_chain.py (full natif)
  3. Si LSP non disponible → fallback natif avec avertissement
  4. Si LSP disponible → utiliser lsp_chain_builder.py
```

Pas de fallback silencieux par étage.

## 8. Interface CLI

### Flags de coloration

| Flag | Range | Défaut | Description |
|------|-------|--------|-------------|
| `--glue` | 0–1 | 0.15 | Saturation drive (0=off, 1=max) |
| `--air` | −5…+5 | 0.0 | Bell 10 kHz (+=brighter, −=darker) |
| `--width` | −1…+1 | 0.0 | Stereo width (+widens) |
| `--bus-comp` | 0–1 | 0.0 | SSL bus compressor (parallèle) |
| `--intensity` | 0–1 | 0.5 | Global intensity |
| `--expander` | flag | off | Gentle upward expansion |
| `--deharsher` | flag | off | De‑harsher 2.5-4.5 kHz |

### Flags structurels

| Flag | Défaut | Description |
|------|--------|-------------|
| `--target-lufs` | −14 | Output loudness (EBU R128) |
| `--ceiling` | −1.1 | Limiter ceiling dBFS |
| `--force-native` | off | Pure ffmpeg chain |
| `--dry-run` | off | Analyse only, print filtergraph |

### Presets

| Preset | Glue | Air | Width | Bus | LUFS |
|--------|------|-----|-------|-----|------|
| `transparent` | 5% | +0.5 | 0.00 | 0% | −14 |
| `warm` | 50% | 0.0 | −0.15 | 30% | −13 |
| `open` | 10% | +2.5 | +0.35 | 15% | −14 |
| `punchy` | 40% | +2.0 | +0.10 | 50% | −11 |
| `loud` | 60% | +3.0 | +0.20 | 70% | −9 |

## 9. Décisions architecturales

| # | Question | Décision | Justification |
|---|----------|----------|---------------|
| 1 | EQ Stereo vs MidSide | **Stereo** | Plus simple, pas d'artefact de phase M/S |
| 2 | Drive saturateur | **Fixe** (piloté par glue+intensity) | Déterministe, correspond aux saturateurs hardware |
| 3 | Plugin manquant | **Échec propre**, `--force-native` global | Environnement homogène |
| 4 | De‑harsher | **Opt‑in, avant** le saturateur | Corriger la source avant de saturer |
| 5 | Position air shelf | **Dans le même nœud EQ** (avant saturation) | La brillance pré‑saturation est plus musicale |
| 6 | Expander défaut | **Off** par défaut | Effet très léger (~0.4 dB). Activé sur demande. |
| 7 | Air défaut | **0.0** (neutre) | Ne pas colorer sans demande explicite |

## 10. Glossaire

| Terme | Définition |
|-------|-----------|
| **LSP** | Linux Studio Plugins — suite de plugins audio open‑source |
| **LV2** | Format de plugin audio standard sur Linux (successeur de LADSPA) |
| **URI** | Uniform Resource Identifier — identifiant unique d'un plugin LV2 |
| **filter_complex** | Graphe de filtres ffmpeg en une seule chaîne |
| **M/S** | Mid/Side — encodage stéréo alternatif (Mid = L+R, Side = L−R) |
| **Sidechain** | Signal de contrôle externe (ex: Mid compresse le Side) |
| **Crest factor** | Ratio peak / RMS (en dB) — mesure la dynamique |
| **AGC** | Automatic Gain Control — compression lente typique des smartphones |
| **LUFS** | Loudness Units relative to Full Scale — mesure normalisée EBU R128 |
| **G linéaire** | Multiplicateur de gain (1.0 = unité = 0 dB) utilisé par LSP |
| **ALR** | Adaptive Level Release — release variable selon le niveau |
| **GainTracker** | Registre de gain heuristique le long de la chaîne |

## 11. Références

- LSP project : https://lsp-plug.in/
- LV2 specification : https://lv2plug.in/
- FFmpeg LV2 filter : https://ffmpeg.org/ffmpeg-filters.html#lv2
