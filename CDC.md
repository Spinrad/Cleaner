# CDC — cleaner v4.0.0 (cible)

**Statut :** Design / Spécification cible — non implémenté.
**Base :** cleaner v3.1.0 (architecture 100 % native ffmpeg).
**Dernière mise à jour :** 2026-06-11

---

## 1. Vision

Faire évoluer cleaner d'une architecture 100 % native ffmpeg vers une
architecture **hybride ffmpeg + LSP/LV2**, où :

- Les **étages structurels** (décodage, resample, HP, encodage/décodage M/S,
  sidechain ducking, mesure LUFS, re‑limiteur post‑LUFS) restent en
  **ffmpeg natif**.
- Les **étages de coloration** (expander anti‑AGC, notches EQ + air,
  de‑harsher, saturation, bus compressor, limiteur musical) passent sur
  des **plugins LSP au format LV2**, chargés via le filtre `lv2` natif
  de ffmpeg.
- Le rendu reste en **single‑pass `filter_complex`** — aucun render
  intermédiaire, aucune dépendance à Carla ou JACK.
- L'installation des plugins LSP est la responsabilité de l'utilisateur
  final. Cleaner détecte leur présence au démarrage et échoue proprement
  si requis, avec `--force-native` comme échappatoire.

## 2. Prérequis

- Python ≥ 3.11
- ffmpeg ≥ 5.0, **compilé avec `--enable-lv2`**
- LSP plugins LV2 installés (`lsp-plugins-lv2` sur Debian/Ubuntu)

Commande de vérification :
```bash
ffmpeg -version | grep lv2          # doit montrer --enable-lv2
lv2ls | grep lsp                    # doit lister les plugins LSP
```

## 3. Architecture cible

### 3.1 Chaîne complète (ordre exact)

```
[0:a]aresample=48000,highpass=f=35:t=o:p=2
│
├─ 1. [LSP] Expander anti‑AGC          expander_stereo
│      Mode=Up, piloté par crest/RMS/AGC recovery
│      REMPLACE agate=mode=upward natif
│
├─ 2. [NATIF] M/S encode               stereotools=mode=lr>ms
│
├─ 3. [NATIF] Sidechain ducking        channelsplit + HP 150 Hz +
│      (Mid → Side)                     sidechaincompress + amerge
│
├─ 4. [NATIF] M/S decode               stereotools=mode=ms>lr
│
├─ 5. [LSP] De‑harsher (opt‑in)        sc_compressor_stereo (bande unique)
│      Centre ← bande de dureté mesurée ; seuil ← harshness_index
│      Placé AVANT le saturateur
│
├─ 6. [LSP] Parametric EQ              para_equalizer_x16_stereo
│      Par mode : 1 bande Bell/Notch (freq, Q, gain)
│      + 1 bande hi‑shelf pour l'air
│      Mode Stereo (pas MidSide)
│
├─ 7. [LSP] Saturator                  saturator_stereo
│      Piloté en drive + makeup.
│      Mapping glue→drive (0→0 dB, 0.5→+6, 1→+12), modulé par --intensity
│      Le signal entre RÉELLEMENT dans la zone non‑linéaire
│
├─ 8. [LSP] Bus Compressor             compressor_stereo
│      Modes Down/Up/Boost, Ratio, Knee, Makeup, Attack/Release (2 temps),
│      Dry/Wet pour parallèle
│
├─ 9. [LSP] Limiter                    limiter_stereo
│      True‑peak, oversampling, ALR (adaptive release)
│
├─ 10. [NATIF] Mesure LUFS             ebur128 + volume
│      Gain clampé [‑3, +6] dB
│
└─ 11. [NATIF] Re‑limiteur sécurité    alimiter
       Filet post‑LUFS, ceiling = --ceiling
```

### 3.2 Règles d'intégrité

1. **Single‑pass** : tout tient dans UN `filter_complex`. Aucun render
   intermédiaire, aucune mesure audio mid‑graphe.
2. **Aucun sidechain externe dans les nœuds LSP** : tous les LSP sont
   mono/stéréo in → out. Le seul traitement sidechain est le ducking
   Mid→Side en ffmpeg natif.
3. **Suivi de niveau 100 % analytique** : un `GainTracker` prédit le
   niveau (peak/RMS) après chaque étage LSP. Les paramètres dépendants
   du niveau (drive saturator, threshold limiter) utilisent le niveau
   PRÉDIT, pas le RMS d'origine. Interdit : réanalyse STFT/onsets/corrélation
   entre étages.
4. **Échappement robuste** : les URIs LSP contiennent `:` et les contrôles
   utilisent `|` — le builder doit produire un graphe syntaxiquement
   valide pour ffmpeg.

### 3.3 Deux builders, pas de fallback par étage

- **`lsp_chain_builder.py`** : construit le graphe avec nœuds `lv2`.
  Si un plugin LSP requis est absent → **échec propre** avec message
  d'installation.
- **`ffmpeg_chain.py`** (conservé) : construit le graphe full‑natif
  (comportement v3.1.0). Invoqué via `--force-native`.
- **Pas de fallback natif silencieux par étage.** L'environnement est
  déclaré homogène (LSP installé partout ou pas du tout). Si l'utilisateur
  choisit LSP, il a LSP.

## 4. Plugins LSP — mapping et contraintes

### 4.1 URIs

Format : `http://lsp-plug.in/plugins/lv2/<slug>`

Slugs cibles (à confirmer par introspection au runtime) :

| Étage | Slug | Rôle |
|-------|------|------|
| Expander | `expander_stereo` | Anti‑AGC, Mode=Up |
| EQ | `para_equalizer_x16_stereo` | Notches + air shelf |
| De‑harsher | `sc_compressor_stereo` | Bande unique, réduction de dureté |
| Saturator | `saturator_stereo` | Drive + makeup, coloration tape |
| Compressor | `compressor_stereo` | Glue/bus, modes multiples |
| Limiter | `limiter_stereo` | True‑peak, ALR |

### 4.2 Unités LSP

**ATTENTION** : LSP exprime les gains en **multiplicateur linéaire G**
(1.0 = unité = 0 dB), pas en dB. Le cerveau calcule en dB/linéaire/ms
et convertit par port.

- G = 10^(dB/20)
- dB = 20 × log10(G)
- Temps : LSP utilise généralement les **secondes** (s), pas les ms.

### 4.3 Découverte des ports

Source de vérité : `lv2info <uri>` et/ou `ffmpeg -f lavfi -i anullsrc -af
lv2=p='<uri>':c=help -f null -` (ffmpeg imprime les contrôles sur stderr).

L'introspection est **cachée** (JSON sur disque) et **rafraîchie si la
version de LSP change**.

### 4.4 Ports réels — source de vérité

Les symboles de ports ci‑dessous sont **indicatifs**. La source de
vérité est l'introspection runtime (`lv2info <uri>` ou
`lv2=...:c=help`). Chaque port sera confirmé avant d'être utilisé
dans le code.

Exemple pour `saturator_stereo` (à vérifier) :
- `sg` : input gain (G linéaire)
- `drive` : drive (G linéaire ou dB)
- `owg` : output gain / makeup (G linéaire)
- `ld_gain` : post‑filter gain

Exemple pour `expander_stereo` (à vérifier) :
- `at_lvl` : attack level / threshold
- `rat` : ratio
- `at` : attack time (s)
- `rt` : release time (s)
- `kn` : knee
- `mk` : makeup gain (G linéaire)

Note : le LSP `expander_stereo` n'a pas de port "range" (contrairement
à l'agate natif). La quantité d'expansion est pilotée par le Ratio.

## 5. Cerveau refondu

### 5.1 Split structurel / niveau

- **Analyse structurelle** (une fois, sur l'original) : modes, crest,
  onsets, corrélation M/S, clipping, harshness. Inchangeable.
- **Niveau** (mesuré une fois sur l'original, puis SUIVI analytiquement) :
  peak_db, rms_db, LUFS. Le `GainTracker` maintient l'estimation courante.

### 5.2 GainTracker (module `gain_tracking.py`)

```python
class GainTracker:
    def __init__(self, initial_peak_dbfs, initial_rms_dbfs)
    def predict_after(stage_gain_db) -> tuple[float, float]
    def commit(stage_name, gain_db)
```

Chaque fonction de paramétrage LSP reçoit le tracker et utilise
`tracker.current_rms_db` (ou peak) pour calculer ses seuils.

### 5.3 Émission des dicts de contrôle

Par étage LSP, le cerveau émet un `dict[symbole, valeur]` où :
- La valeur est dans l'unité native du port LSP (dB, G linéaire,
  secondes, Hz).
- Elle est clampée à `[min, max]` du port (issu de l'introspection).
- Les symboles sont découverts dynamiquement (pas de mapping statique).

### 5.4 Paramétrage par étage (spécification)

#### Expander (anti‑AGC)
- **Mode** = Up (expansion vers le haut)
- **Ratio** ← `clamp(1.6 − crest × 0.03, 1.1, 1.5)`
  - crest=4 (très compressé) → 1.48 → expansion forte
  - crest=12 (normal) → 1.24 → expansion modérée
  - crest=18 (très dynamique) → 1.1 (clamp) → expansion minimale
- **Attack Level** ← peak_db − 3 dB (seuil proche du peak)
- **Attack Time** ← min(attack_ms × 0.5, 10) ms
- **Release Time** ← AGC recovery × 0.8, clamp [15, 50] ms
- **Knee** + **Makeup** pour adoucir la transition

Note : les symboles exacts (ports) seront confirmés par `lv2info
expander_stereo` lors de la Phase 0 — ne pas coder en dur sans
vérification. Le LSP expander_stereo n'a pas de port "range" ;
la quantité d'expansion est pilotée par le Ratio.

#### EQ (notches + air)
- Par mode détecté (max 3) :
  - **Freq** ← mode mesuré (Hz)
  - **Q** ← mode Q mesuré, clampé [3, 10]
  - **Gain** ← −(proéminence × 0.5), clampé [−9, −2] dB.
    Si proéminence < 3 dB → bande désactivée (gain = 0 dB).
  - **Type** = Bell (ou Notch selon disponibilité LSP)
- **Air shelf** : hi‑shelf à 8 kHz, gain = `--air`, Q = 0.7.
  **Position :** dans le même nœud EQ (avant saturation).
  Alternative : nœud séparé post‑saturation via `--air-post-sat`
  (réservé v4.1).

#### De‑harsher (opt‑in)
- **Centre freq** ← milieu de la bande de dureté mesurée (2.5–4.5 kHz)
- **Seuil** ← harshness_index (échelle normalisée)
- **Ratio** ← modéré (1.5–3.0)
- Placé **avant** le saturateur (la saturation n'amplifie pas la dureté)

#### Saturator
- **Drive (dB)** ← mapping glue → drive :
  - `glue=0` → 0 dB (pas de saturation)
  - `glue=0.5` → +4–6 dB (saturation audible)
  - `glue=1.0` → +12 dB (saturation marquée)
  - Le mapping est modulé par `--intensity`
- **Makeup (dB)** ← −drive × 0.5 (compensation automatique, niveau
  quasi‑constant en sortie)
- **Pas** de modulation par mesure HF en v4.0 (option `--adaptive-drive`
  réservée pour v4.1)

#### Bus Compressor
- **Mode** = Down (Downward compression)
- **Threshold (dB)** ← RMS prédit − crest × 0.3 + (1 − bus_comp) × 12
  (utilise `--bus-comp`, pas `--glue`)
- **Ratio** ← 2:1 (SSL classique)
- **Attack** = 10 ms (laisse passer les transitoires)
- **Release** = 100 ms (relâchement doux)
- **Knee** = 4 dB
- **Mix** ← bus_comp (compression parallèle)
- Utiliser le GainTracker pour le RMS prédit (after EQ + saturator)

#### Limiter
- **Mode** = true‑peak avec oversampling
- **Threshold (dB)** ← niveau peak prédit (after bus comp)
- **Ceiling (dB)** ← `--ceiling` (linéaire)
- **ALR** (adaptive release) activé
- **Post‑LUFS re‑limiteur** : reste un `alimiter` natif (filet de sécurité
  uniquement, ne devrait normalement pas s'activer)

### 5.5 Macro `--intensity`

Scale simultanément :
- **Expander ratio** (déjà fait en v3.1 via le range ; en v4 via le ratio)
- **Notch depth** (déjà fait en v3.1)
- **Saturator drive** (nouveau en v4)

Mapping : `intensity ∈ [0, 1]` → multiplicateur `0.2 + intensity × 0.8`
appliqué au drive, au ratio expander et à la profondeur des notches.
intensity=0 → effet nul, intensity=1 → effet maximal.

### 5.6 Introspection et cache

Module `lv2_introspect.py` :
- `discover_lsp_plugins()` → `dict[slug, uri]` via `lv2ls | grep lsp`
- `introspect_plugin(uri)` → `dict[symbol, PortInfo]` via
  `lv2info <uri>` ou `lv2=...:c=help`
- Cache JSON dans `/tmp/cleaner_lv2_cache.json`
- Invalidation si version LSP ou ffmpeg change

Module `lv2_params.py` :
- `db_to_linear_gain(db)`, `linear_gain_to_db(g)`
- `ms_to_s(ms)`, `s_to_ms(s)`
- `clamp_to_port(value, PortInfo)` → valeur clampée + convertie

## 6. Builders

### 6.1 `lsp_chain_builder.py` (nouveau)

Construit le graphe `filter_complex` avec nœuds `lv2=...` pour les étages
de coloration.

Fonction `build_lv2_node(uri, params)` → `"lv2=p='<uri>':c=sym1=val1|sym2=val2"`

Contraintes d'échappement :
- L'URI contient `:` → wrappée dans des quotes simples : `p='http://...'`
- Les `|` des contrôles cohabitent avec les `;` du graphe → le builder
  assemble correctement
- Tests unitaires : graphe complet accepté par `ffmpeg -filter_complex -f null -`

### 6.2 `ffmpeg_chain.py` (conservé, modifié)

- Reste le builder full‑natif (comportement v3.1.0).
- Invoqué via `--force-native`.
- Peut partager des helpers avec `lsp_chain_builder.py` pour le préambule
  natif (resample, HP35, M/S, sidechain) et la sortie (ebur128, post‑limiter).

## 7. Stratégie de build et fallback

```
Au démarrage :
  1. Détection LSP : lsp_available = (lv2ls renvoie les URIs requises)
  2. Si --force-native → utiliser ffmpeg_chain.py (full natif)
  3. Si LSP non disponible → message d'erreur + suggestion --force-native
  4. Si LSP disponible → utiliser lsp_chain_builder.py
```

**Pas de fallback silencieux par étage.** L'utilisateur sait s'il est
en mode natif ou LSP.

## 8. Interface CLI — changements v3.1 → v4.0

### Conservés
- Tous les flags booléens (`--expander/--no-expander`, etc.)
- Tous les presets et leurs valeurs
- `--dry-run`, `--keep-temp`, `--verbose`, `--timeout`
- `--target-lufs`, `--ceiling`, `--notch-intensity`, `--tame-cymbals`
- `--glue`, `--air`, `--width`, `--bus-comp`, `--intensity`

### Nouveaux
- `--force-native` : utilise le builder full‑natif (v3.1).
  Sans ce flag, le mode LSP est le défaut et l'absence d'un plugin
  requis provoque un échec propre avec message d'installation.

### Modifiés
- `--intensity` : scale maintenant aussi le drive du saturateur (en plus
  de l'expander et des notches)

## 9. Décisions architecturales actées

| # | Question | Décision | Justification |
|---|----------|----------|---------------|
| 1 | EQ Stereo vs MidSide | **Stereo** | Plus simple, pas d'artefact de phase M/S, gain MidSide nul sur des coupes ≤ 9 dB |
| 2 | Drive saturateur | **Fixe** (piloté par glue+intensity) en v4.0 ; `--adaptive-drive` en v4.1 | Déterministe, prédictible, correspond aux saturateurs hardware |
| 3 | Plugin manquant | **Échec propre**, pas de fallback par étage. `--force-native` global. | Environnement homogène ; pas de surprise sur la qualité |
| 4 | De‑harsher | **Opt‑in, avant** le saturateur | Corriger la source avant de saturer |
| 5 | Expander LSP | **Remplace agate**, en tête de chaîne (après HP35, avant M/S encode). Rôle anti‑AGC. Pas de second expander. | Un seul expander, le LSP est plus paramétrable |
| 6 | Position air shelf | **Dans le même nœud EQ** (avant saturation) par défaut. Un flag `--air-post-sat` (v4.1) permettra de le placer après. | La brillance pré‑saturation est plus musicale (la saturation écrête les aigus boostés naturellement). L'option post‑saturation offre un son plus « hifi ». |

## 10. Phases d'implémentation

### Phase 0 — Jalon Saturator (banc d'essai)
1. `lv2_introspect.py` : découverte + introspection + cache
2. `lv2_params.py` : conversions + clamp
3. `gain_tracking.py` : suivi de niveau analytique
4. `lsp_chain_builder.py` : `build_lv2_node()` avec échappement
5. Saturator LSP intégré de bout en bout :
   - Cerveau → dict de contrôle → nœud lv2 → render → test A/B
6. **Test d'audibilité** : rendu sweep → comparaison harmonique entrée/sortie

**STOP après Phase 0 pour validation A/B.**

### Phase 1 — Expansion
7. EQ LSP (notches + air)
8. Compressor LSP (bus/glue)
9. Expander LSP (anti‑AGC, remplace agate)
10. Limiter LSP
11. De‑harsher LSP

### Phase 2 — Robustesse et polish
12. `--force-native` + détection LSP au démarrage
13. Renforcement des tests preset (valeurs réelles)
14. README.md final (archi hybride)
15. Nettoyage `.gitignore`, docstrings

## 11. Glossaire

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
| **GainTracker** | Module de suivi analytique du niveau le long de la chaîne |

## 12. Références

- LSP project : https://lsp-plug.in/
- LV2 specification : https://lv2plug.in/
- FFmpeg LV2 filter : https://ffmpeg.org/ffmpeg-filters.html#lv2
