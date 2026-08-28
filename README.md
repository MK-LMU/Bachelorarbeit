# SpEx vs. IDC — comparing rule-based and gate-based clustering explanations

Code and results for a bachelor's thesis. Two interpretable clustering methods
produce very different explanation objects: **SpEx** an axis-parallel rule tree,
**IDC** learned per-sample feature gates. This repository converts the SpEx tree
into SHAP's custom-tree format so that both methods yield an `(N, D)` importance
matrix — and then scores both with **IDC's own metric code, imported verbatim**,
so no measured difference can originate in the evaluation.

**[English](#english) · [Deutsch](#deutsch)**

---

## English

### Quick start

```
python -m venv venv
./venv/Scripts/pip install -r requirements.txt   # pick the torch CUDA build for your GPU
./venv/Scripts/python.exe reproduce_all.py --dry # print every command, run nothing
```

`reproduce_all.py` is the entry point worth reading first: it lists **every
program invocation** behind the reported numbers, in dependency order, with a
comment on each. Running it without `--dry` executes everything (~13 h, GPU+CPU).

One prerequisite no script creates: `artifacts/data/har_data.npz`, built once
from the UCI HAR dataset (keys `X`, `y`).

### What is in this repository

| Path | Contents |
|---|---|
| `RESULTS_MULTISEED.md` | **The canonical results table** (mean ± std, all metrics and datasets). Generated — do not edit; regenerate with `gen_results_table.py`. |
| `reproduce_all.py` | Every program invocation, in dependency order. Documentation and executable at once. |
| `results/` | All result JSONs: `results_multiseed_*.json` plus the standalone checks. |
| `*.py` | The pipeline (see below). |
| `SpEx/`, `IDC/` | The authors' original repositories, **unchanged** — see `VENDORED.md`. |
| `VENDORED.md` | Provenance, upstream commits and licences of the vendored code. |

The thesis' own documentation — findings log, methodological review, figures and
their reading guide — is kept locally in `notes/` and is not part of this
repository. Every script states in its own docstring why it exists and what it
establishes; there are deliberately no references to the private documents.

### Scripts by role

**The bridge** — the actual contribution of this work:

| Script | Purpose |
|---|---|
| `tree_shap_spex.py` | Converts a trained SpEx tree into SHAP's seven-array custom-tree dict. Contains the sklearn round-trip check. |
| `spex_pipeline.py` | `spex_side()`: spectral reference → CliqueBased tree → Tree SHAP → `(N, D)` gate matrix. |

**Measurement:**

| Script | Purpose |
|---|---|
| `metrics.py` | IDC's metrics imported verbatim, plus labelled additions (drop-faithfulness, row-normalisation, nn-identical, `diversity_fixed`) and a bit-exact fast path. **Run it directly** to execute the equivalence proof against IDC's originals on all nine datasets → `results/equivalence_check.json`. |

**Training:**

| Script | Purpose |
|---|---|
| `run_idc.py` | One IDC training run. `--data <ds> --seed <k> [--lgl <f>] [--epochs <n>] [--tag _tuned]`. Saves gates, labels, weights and the full resolved config. |
| `train_all.py` | Resumable multi-seed campaign: 8 datasets × 5 seeds as subprocesses. |

**Evaluation:**

| Script | Purpose |
|---|---|
| `evaluate.py` | One pass per dataset: SpEx side per seed, all metrics for both methods, k-means baseline → `results/results_multiseed_<ds>.json`. |
| `gen_results_table.py` | Regenerates `RESULTS_MULTISEED.md`. Almost every number is read straight from a JSON; the one exception is IDC's `faithfulness (correlation)` column, which is aggregated here from the training npz in `artifacts/` (not shipped). |
| `significance.py` | One-sample t-tests of IDC's per-seed ARIs against the seed-constant SpEx value, Holm-corrected, plus Welch tests of the architecture check. |

**Label-free model selection:**

| Script | Purpose |
|---|---|
| `tune_idc.py` | Per-dataset grid (gate penalty × epochs). Winner chosen by silhouette; ARI is logged but never used. |
| `reselect_best.py` | Applies the K-constrained rule to the logged grid — a candidate must use all K clusters. No retraining of the grid. |
| `reference_selection.py` | The same protocol for SpEx's only free choice, the reference clusterer (spectral vs. k-means). |

**RQ2 — stability under data perturbation:**

| Script | Purpose |
|---|---|
| `perturbations.py` | The shared perturbation function. Imported by both sides so they see identical perturbed data. |
| `perturbation_stability.py` | Noise and subsampling with repetitions, including the σ = 0 retrain control. |

**Verification:**

| Script | Purpose |
|---|---|
| `test_correctness.py` | The five-part correctness chain (functional equivalence, format round-trip, additivity, model-agnostic cross-check, exact Shapley definition by brute force) plus three edge cases (over-segmentation, empty leaves, high dimensions). |
| `test_correctness_evaluated.py` | Part 3: equivalence, additivity, production wiring and brute force on the trees that are actually evaluated, for all nine datasets. |
| `test_metrics.py` | Property tests of the metric layer (diversity offset and ceiling, empty-set pathology, K pass-through, fast path bit-exact). Data-free, seconds. |
| `compare_kprime.py` | Over-segmentation `k' > k`: do more leaves close the granularity gap? |

Four further check scripts (`check_architecture.py`, `check_gener_scale.py`,
`check_uniq_scale.py`, `check_faithfulness_masking.py`) and their result JSONs
are held back pending a discussion with the supervisor; the calls in
`reproduce_all.py` are commented out accordingly. The shipped
`results/significance.json` still carries the four Welch tests of the
architecture check, because they are a result of this work; only the script
that produced their input is held back. `significance.py` degrades cleanly:
run without `arch_check.json`, its architecture section becomes `null` and
the head-to-head t-tests are unaffected.

**Data and infrastructure:**

| Script | Purpose |
|---|---|
| `extract_cifar.py` | CIFAR-10 ResNet18 features → `artifacts/data/cifar_feats.npz`. |
| `extract_mnist_feats.py` | MNIST ResNet18 features on the *same* 10k samples as the raw-pixel runs. |
| `run_mnist_feats_chain.py` | One-shot driver for the `mnist_feats` dataset (extract → seeds → grid → evaluate). |
| `visualize_explanations.py` | Regenerates the figures into `notes/figures/`. |
| `wpaths.py` | Central paths — the one place that knows the folder layout. |
| `reproduce_all.py` | The list of all program invocations. |

### Folder layout

```
artifacts/idc_out/   idc_out_<ds>[_tag]_seed<k>.npz   IDC training outputs
artifacts/models/    idc_model_*.pt                   weights + config + seed
artifacts/data/      har_data.npz, cifar_feats.npz, mnist_feats.npz, downloads
results/             results_multiseed_*.json plus the standalone check JSONs
logs/                console output of all runs
notes/               documentation, figures        — NOT in the repository
SpEx/, IDC/          the authors' repositories, UNCHANGED
```

`artifacts/`, `logs/` and `notes/` are gitignored. The first two are fully
regenerated by `train_all.py` / `evaluate.py`.

### Reproducibility caveats

Three data files carry the CIFAR and MNIST-features numbers and are **not** in
the repository (87 MB): `har_data.npz` is hand-built from UCI HAR;
`cifar_feats.npz` and `mnist_feats.npz` are extracted with ResNet18 on the GPU
and are **not bit-identically re-extractable** on different hardware or a
different torch version. Their SHA-256 sums are in `DATA_CHECKSUMS.sha256`:

```
sha256sum -c DATA_CHECKSUMS.sha256          # Git Bash
```

MNIST itself is downloaded by IDC's own dataset class.

### Environment

Windows, Python 3.13, `venv\Scripts\python.exe` (GPU: RTX 5070 Ti, torch cu128).
Start all runs from this folder. `requirements.txt` pins the exact versions the
reported numbers were produced with; `scipy < 1.20` is mandatory because
`scipy.spatial.distance_matrix` — used by IDC's original metrics and by the fast
path — is removed in 1.20.

**Note for non-Windows users:** the subprocess drivers (`train_all.py`,
`tune_idc.py`, `reselect_best.py`, `perturbation_stability.py`,
`reproduce_all.py`, `run_mnist_feats_chain.py`) assume the Windows interpreter
path `venv\Scripts\python.exe`. On Linux or macOS, adjust the `PY` variable at
the top of those files to `venv/bin/python`.

### Licence

This repository's own code is MIT (see `LICENSE`). The vendored repositories
keep their own MIT licences and copyright holders: `SpEx/` (talargv), `IDC/`
(Jonathan Svirsky). `VENDORED.md` records the exact upstream commits.

---

## Deutsch

### Schnellstart

```
python -m venv venv
./venv/Scripts/pip install -r requirements.txt   # torch mit CUDA passend zur GPU
./venv/Scripts/python.exe reproduce_all.py --dry # alle Befehle ausgeben, nichts rechnen
```

`reproduce_all.py` ist die Datei, die man zuerst liest: sie listet **jeden
Programmaufruf** hinter den berichteten Zahlen in Abhängigkeitsreihenfolge, jeweils
kommentiert. Ohne `--dry` führt sie alles aus (ca. 13 h, GPU + CPU).

Eine Voraussetzung erzeugt kein Skript: `artifacts/data/har_data.npz`, einmalig
aus dem UCI-HAR-Datensatz gebaut (Keys `X`, `y`).

### Was hier liegt

| Pfad | Inhalt |
|---|---|
| `RESULTS_MULTISEED.md` | **Die kanonische Ergebnistabelle** (mean ± std, alle Metriken und Datensätze). Generiert — nicht editieren, sondern mit `gen_results_table.py` neu erzeugen. |
| `reproduce_all.py` | Alle Programmaufrufe in Abhängigkeitsreihenfolge. Dokumentation und ausführbar zugleich. |
| `results/` | Alle Ergebnis-JSONs: `results_multiseed_*.json` plus die freistehenden Prüfungen. |
| `*.py` | Die Pipeline (siehe unten). |
| `SpEx/`, `IDC/` | Die Original-Repos der Autoren, **unverändert** — siehe `VENDORED.md`. |
| `VENDORED.md` | Herkunft, Upstream-Commits und Lizenzen des Fremdcodes. |

Die ausführliche Dokumentation der Arbeit — Befundprotokoll, methodisches
Gutachten, Abbildungen samt Lese-Anleitung — liegt lokal in `notes/` und ist
nicht Teil dieses Repos. Jedes Skript begründet im eigenen Docstring, warum es
existiert und was es belegt; Verweise auf die privaten Dokumente gibt es
bewusst nicht.

### Skripte nach Rolle

**Die Brücke** — die eigentliche Eigenleistung:

| Skript | Zweck |
|---|---|
| `tree_shap_spex.py` | Überführt einen trainierten SpEx-Baum in SHAPs Sieben-Array-Format. Enthält den sklearn-Round-Trip-Test. |
| `spex_pipeline.py` | `spex_side()`: Spectral-Referenz → CliqueBased-Baum → Tree SHAP → `(N, D)`-Gate-Matrix. |

**Messung:**

| Skript | Zweck |
|---|---|
| `metrics.py` | IDCs Metriken wörtlich importiert, dazu gekennzeichnete Ergänzungen (drop-Faithfulness, Zeilennormierung, nn-identical, `diversity_fixed`) und ein bitgenauer Schnellpfad. **Direkt ausführen** startet den Äquivalenzbeweis gegen IDCs Originale auf allen neun Datensätzen → `results/equivalence_check.json`. |

**Training:**

| Skript | Zweck |
|---|---|
| `run_idc.py` | Ein einzelnes IDC-Training. `--data <ds> --seed <k> [--lgl <f>] [--epochs <n>] [--tag _tuned]`. Speichert Gates, Labels, Gewichte und die vollständig aufgelöste Config. |
| `train_all.py` | Wiederaufnehmbare Mehr-Seed-Kampagne: 8 Datensätze × 5 Seeds als Subprozesse. |

**Auswertung:**

| Skript | Zweck |
|---|---|
| `evaluate.py` | Ein Durchlauf je Datensatz: SpEx-Seite pro Seed, alle Metriken für beide Verfahren, k-Means-Baseline → `results/results_multiseed_<ds>.json`. |
| `gen_results_table.py` | Erzeugt `RESULTS_MULTISEED.md` neu. Fast jede Zahl stammt direkt aus einer JSON; einzige Ausnahme ist IDCs Spalte `faithfulness (correlation)`, die hier aus den Trainings-npz in `artifacts/` aggregiert wird (nicht ausgeliefert). |
| `significance.py` | Ein-Stichproben-t-Tests der IDC-Seeds gegen den seed-konstanten SpEx-Wert, Holm-korrigiert, plus Welch-Tests des Architektur-Spotchecks. |

**Label-freie Modellwahl:**

| Skript | Zweck |
|---|---|
| `tune_idc.py` | Grid je Datensatz (Gate-Strafe × Epochen). Auswahl per Silhouette; ARI wird protokolliert, aber nie verwendet. |
| `reselect_best.py` | Wendet die K-beschränkte Regel auf das protokollierte Grid an — ein Kandidat muss alle K Cluster benutzen. Kein Neutraining des Grids. |
| `reference_selection.py` | Dasselbe Protokoll für SpEx' einzige freie Wahl, den Referenz-Clusterer (Spectral vs. k-Means). |

**RQ2 — Stabilität unter Datenstörung:**

| Skript | Zweck |
|---|---|
| `perturbations.py` | Die gemeinsame Störfunktion. Von beiden Seiten importiert, damit sie exakt dieselben gestörten Daten sehen. |
| `perturbation_stability.py` | Rauschen und Subsampling mit Wiederholungen, inklusive der σ = 0-Retrain-Kontrolle. |

**Absicherung:**

| Skript | Zweck |
|---|---|
| `test_correctness.py` | Die fünfteilige Korrektheitskette (Funktionsgleichheit, Format-Round-Trip, Additivität, modellagnostische Gegenprobe, exakte Shapley-Definition per Brute Force) plus drei Grenzfälle (Übersegmentierung, leere Blätter, hohe Dimension). |
| `test_correctness_evaluated.py` | Teil 3: Funktionsgleichheit, Additivität, Produktionspfad und Brute Force auf den tatsächlich ausgewerteten Bäumen aller neun Datensätze. |
| `test_metrics.py` | Eigenschaftstests der Metrik-Schicht (diversity-Offset und -Decke, Empty-Set-Pathologie, K-Durchreichung, Schnellpfad bitgenau). Datenfrei, Sekunden. |
| `compare_kprime.py` | Übersegmentierung `k' > k`: schließen mehr Blätter die Granularitätslücke? |

Vier weitere Prüfskripte (`check_architecture.py`, `check_gener_scale.py`,
`check_uniq_scale.py`, `check_faithfulness_masking.py`) und ihre Ergebnis-JSONs
sind bis zur Absprache mit dem Betreuer zurückgehalten; die Aufrufe in
`reproduce_all.py` sind entsprechend auskommentiert. Die ausgelieferte
`results/significance.json` enthält die vier Welch-Tests des Architektur-Checks
weiterhin, weil sie ein Ergebnis dieser Arbeit sind; zurückgehalten ist nur das
Skript, das ihre Eingabe erzeugt hat. `significance.py` degradiert sauber: ohne
`arch_check.json` wird sein Architektur-Abschnitt `null`, die
Head-to-head-t-Tests bleiben unberührt.

**Daten und Infrastruktur:**

| Skript | Zweck |
|---|---|
| `extract_cifar.py` | CIFAR-10-ResNet18-Features → `artifacts/data/cifar_feats.npz`. |
| `extract_mnist_feats.py` | MNIST-ResNet18-Features auf *denselben* 10.000 Samples wie die Rohpixel-Läufe. |
| `run_mnist_feats_chain.py` | Einmal-Treiber für den Datensatz `mnist_feats` (Extraktion → Seeds → Grid → Auswertung). |
| `visualize_explanations.py` | Erzeugt die Abbildungen neu, nach `notes/figures/`. |
| `wpaths.py` | Zentrale Pfade — die eine Stelle, die das Ordner-Layout kennt. |
| `reproduce_all.py` | Die Liste aller Programmaufrufe. |

### Ordnerstruktur

```
artifacts/idc_out/   idc_out_<ds>[_tag]_seed<k>.npz   IDC-Trainings-Outputs
artifacts/models/    idc_model_*.pt                   Gewichte + Config + Seed
artifacts/data/      har_data.npz, cifar_feats.npz, mnist_feats.npz, Downloads
results/             results_multiseed_*.json plus die freistehenden Prüf-JSONs
logs/                Konsolen-Outputs aller Läufe
notes/               Dokumentation, Abbildungen     — NICHT im Repo
SpEx/, IDC/          Original-Repos der Autoren, UNVERÄNDERT
```

`artifacts/`, `logs/` und `notes/` stehen in `.gitignore`. Die ersten beiden
werden von `train_all.py` / `evaluate.py` vollständig regeneriert.

### Einschränkungen der Reproduktion

Drei Datendateien tragen die CIFAR- und MNIST-Feature-Zahlen und liegen **nicht**
im Repo (87 MB): `har_data.npz` ist aus UCI HAR handgebaut; `cifar_feats.npz`
und `mnist_feats.npz` entstehen per ResNet18 auf der GPU und sind auf anderer
Hardware oder Torch-Version **nicht bitgleich re-extrahierbar**. Ihre
SHA-256-Summen stehen in `DATA_CHECKSUMS.sha256`:

```
sha256sum -c DATA_CHECKSUMS.sha256          # Git Bash
```

MNIST lädt IDCs eigene Dataset-Klasse selbst herunter.

### Umgebung

Windows, Python 3.13, `venv\Scripts\python.exe` (GPU: RTX 5070 Ti, torch cu128).
Alle Läufe aus diesem Ordner starten. `requirements.txt` pinnt die exakten
Versionen, unter denen die berichteten Zahlen entstanden sind; `scipy < 1.20` ist
zwingend, weil `scipy.spatial.distance_matrix` — von IDCs Original-Metriken und
vom Schnellpfad benutzt — in 1.20 entfernt wird.

**Hinweis für Nicht-Windows-Nutzer:** Die Subprozess-Treiber (`train_all.py`,
`tune_idc.py`, `reselect_best.py`, `perturbation_stability.py`,
`reproduce_all.py`, `run_mnist_feats_chain.py`) nehmen den Windows-Interpreterpfad
`venv\Scripts\python.exe` an. Unter Linux oder macOS ist die Variable `PY` am
Anfang dieser Dateien auf `venv/bin/python` anzupassen.

### Lizenz

Der eigene Code dieses Repos steht unter MIT (siehe `LICENSE`). Der eingebettete
Fremdcode behält seine eigenen MIT-Lizenzen und Rechteinhaber: `SpEx/` (talargv),
`IDC/` (Jonathan Svirsky). `VENDORED.md` hält die genauen Upstream-Commits fest.
