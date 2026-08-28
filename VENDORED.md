# Eingebetteter Fremdcode (vendored)

Die beiden verglichenen Methoden liegen als **unveränderte Kopien** der
Autoren-Repositories in diesem Repo (ohne deren `.git`-Historie, damit das
Repo keine verschachtelten Git-Repos enthält). Herkunft, exakter Stand und
Lizenz:

| Ordner | Quelle | Commit | Lizenz |
|---|---|---|---|
| `SpEx/` | https://github.com/talargv/SpEx | `d2b87795f275133d5e9051a5995e03fe311496bc` | siehe `SpEx/LICENSE` |
| `IDC/`  | https://github.com/jsvir/idc   | `7facf5339161d2c410e2b70f1b0556d3e86cac01` | siehe `IDC/LICENSE` |

**Keine Datei in diesen Ordnern wurde verändert** (vor dem Entfernen der
`.git`-Ordner per `git status` verifiziert: null modifizierte Dateien; am
2026-08-21 zusätzlich per `diff -r --strip-trailing-cr` gegen die von GitHub
geladenen Snapshots der beiden Commits bestätigt — inhaltlich identisch).
Die einzigen Unterschiede zum Upstream-Snapshot sind keine Code-Änderungen:
Zeilenenden liegen lokal als CRLF vor (Windows-Checkout), `__pycache__/`, und
in `IDC/data/` liegen die heruntergeladenen Daten `pbmc_x.npz`/`pbmc_y.npz`,
die das Upstream-Repo nicht mitführt.
Alle Anpassungen dieser Arbeit — Konverter, Metrik-Ergänzungen, Fixes wie die
K-bewusste Faithfulness — leben ausschließlich in eigenen Modulen im
Projekt-Root (siehe `README.md`). Wo publizierte Metriken Defekte haben, sind
diese dort dokumentiert und durch benannte Varianten ersetzt, ohne den
Originalcode anzutasten: `diversity` summiert die Jaccard-Diagonale mit und
teilt zugleich durch die Zahl der Nicht-Diagonalpaare, und leere Merkmalsmengen
erhöhen den Wert statt ihn undefiniert zu lassen (→ `diversity_fixed` in
`metrics.py`); `stability` fällt bei k = 2 mit `uniqueness` zusammen (→ wird
zusätzlich bei k = 5 berichtet).

Zum Reproduzieren gegen die Originale:

```
git clone https://github.com/talargv/SpEx && git -C SpEx checkout d2b8779
git clone https://github.com/jsvir/idc IDC && git -C IDC checkout 7facf53
```
