> ℹ️ **Über diese Datei**
>
> - Sie basiert auf [../README.md](../README.md).
> - Alle Pfade zeigen einen Ordner nach oben, zum Beispiel `../assets/` und `../data/`.
> - Die englische Version mit gleichem Aufbau liegt in [../En/README.md](../En/README.md).
> - Einen Lösungs-Weg auf Deutsch findest du in [../summary/loesungsweg.ipynb](../summary/loesungsweg.ipynb).

# King County House Prices Project

Ein Projekt für einen Tag mit echten Haus-Daten.

- Du sagst den Verkaufs-Preis von Häusern in King County vorher, also in Seattle und Umgebung.
- Du nutzt ein **Regressions**-Modell.
- Du verbindest zwei Tabellen, bereinigst die Daten und untersuchst sie.
- Du baust eine **Pipeline** für die Vorbereitung.
- Du stimmst zwei Modelle ab und vergleichst sie mit einem **Basis-Modell**.

![Karte von King County mit allen 21.597 Haus-Verkäufen aus 2014 und 2015. Jeder Verkauf ist ein Punkt an seinem Ort, gefärbt nach dem Verkaufs-Preis. Die teuersten Häuser liegen um Bellevue, Mercer Island und am Ufer des Lake Washington, nach Süden hin fallen die Preise.](../assets/king_county_price_map.png)

- Jeder Punkt ist ein Haus-Verkauf, gefärbt nach seinem Preis.
- Die Farb-Skala endet bei 1,5 Millionen Dollar.
- Deshalb erscheinen die wenigen teureren Häuser im selben Gelb.

## 📖 Neue Begriffe

| Deutsch | Englisch | In einem Satz |
| :--- | :--- | :--- |
| **Regression** | regression | Eine **Regression** sagt eine Zahl vorher, hier den Verkaufs-Preis eines Hauses. |
| **Pipeline** | pipeline | Eine **Pipeline** verkettet Vorbereitungs-Schritte und ein Modell zu einem Objekt. |
| **Basis-Modell** | baseline | Ein **Basis-Modell** ist ein einfaches Modell, das deine echten Modelle schlagen müssen. |
| **Kreuz-Validierung** | cross-validation | **Kreuz-Validierung** bewertet ein Modell nacheinander auf mehreren Teilen der Trainings-Daten. |
| **Vorlage** | template | Eine **Vorlage** ist ein GitHub-Repository, das du kopierst, um dein eigenes zu beginnen. |
| **Platzhalter** | placeholder | Ein **Platzhalter** ist Text in spitzen Klammern, den du durch deinen eigenen Wert ersetzt. |

## 🎯 Lern-Ziele

Am Ende dieses Projekts kannst du:

1. Zwei zusammen-gehörende Tabellen verbinden, nachdem du den Schlüssel geprüft hast. Danach teilst du die Daten in Trainings-Daten und Test-Daten, bevor du sie untersuchst.
2. Eine **Pipeline** mit einem `ColumnTransformer` bauen. Sie kodiert die Kategorie-Spalten, die als Zahlen gespeichert sind. Danach stimmst du zwei Modelle mit **Kreuz-Validierung** ab.
3. Jedes Modell in einer Tabelle mit Trainings-Werten und Test-Werten mit zwei **Basis-Modellen** vergleichen. Das eine nimmt den Mittelwert, das andere eine Regel. Danach beschreibst du, wo das beste Modell die größten Fehler macht.

## 🗺️ Lern-Weg

| Datei | Beschreibung |
| :--- | :--- |
| [**01 Loading the Data**](01_loading_the_data.ipynb) | Beide Tabellen laden, den Schlüssel prüfen, sie verbinden und das Ergebnis speichern. |
| [**02 King County Project**](02_king_county_project.ipynb) | Das Projekt: die Daten-Beschreibung und die Aufgabe in acht Schritten. Es lädt die Daten, die Notebook 01 gespeichert hat. |
| [**Lösungs-Weg**](../summary/loesungsweg.ipynb) | Ein möglicher Lösungs-Weg für beide Notebooks, Schritt für Schritt. |
| [**Zusammenfassung**](../summary/zusammenfassung.ipynb) | Eine kurze Zusammenfassung zum Wiederholen, mit Glossar. |
| [**Ausführliche Zusammenfassung**](../summary/zusammenfassung_lang.ipynb) | Alle Ideen des Projekts ausführlich erklärt, mit den echten Zahlen. |

### Weitere Ordner und Dateien

| Datei oder Ordner | Beschreibung |
| :--- | :--- |
| [**Assets**](../assets/) | Das Diagramm der Projekt-Schritte und die Karte der Haus-Verkäufe. Die Mermaid-Quelle liegt in `../assets/diagrams/`. |
| [**Data**](../data/) | Die zwei Quell-Tabellen `king_county_house_sales.csv` und `king_county_house_details.csv`. |
| [**pyproject.toml**](../pyproject.toml) | Projekt-Einstellungen und Abhängigkeiten. |
| [**uv.lock**](../uv.lock) | Datei mit den festen Versionen der Abhängigkeiten. |

## Einrichtung

> [!NOTE]
> Text in spitzen Klammern wie `<repo-name>` ist ein **Platzhalter**. Ersetze ihn samt Klammern durch deinen eigenen Wert. Zum Beispiel wird `cd <repo-name>` zu `cd my-king-county-project`.

### 1. Das Repository aus der Vorlage erstellen

Klick auf GitHub auf **Use this template**. Beim Erstellen des Repositorys:

1. Setz dich selbst als **Owner**.
2. Wähle einen Namen für das Repository.
3. Schalte **Include all branches** aus.
4. Klick auf **Create repository**.

> [!IMPORTANT]
> Arbeitet ihr zu zweit oder in Gruppen, macht nur **eine Person** diesen Schritt.

---

### 2. Mitarbeitende hinzufügen, nur für Paare und Gruppen

Wenn du mit anderen arbeitest:

1. Öffne das Repository auf GitHub.
2. Geh zu **Settings → Collaborators**.
3. Füge die anderen als Collaborators hinzu.
4. Teile den Link zum Repository mit deinem Team.

Die anderen nehmen die Einladung an, bevor sie weiter-machen.

---

### 3. Das Repository klonen

Kopiere die SSH-URL über den Knopf **Code** auf GitHub. Führe dann aus:

```bash
git clone <copied-ssh-url>
```

Die kopierte SSH-URL sieht so aus: `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. In den Projekt-Ordner wechseln und die Abhängigkeiten installieren

Das installiert alle Abhängigkeiten. Es legt außerdem eine virtuelle Umgebung in `.venv/` an.

```bash
cd <repo-name>
uv sync
```

---

### 5. Die Notebooks öffnen

> [!NOTE]
> Öffne VS Code im Haupt-Ordner des Projekts. Dann findet es die Umgebung aus `uv sync` automatisch.

Starte VS Code im Haupt-Ordner des Projekts:

```bash
code .
```

Öffne dann ein Notebook. Wähle die Python-Umgebung aus `uv sync` als Kernel.

## Quellen und weiteres Lesen

- [**scikit-learn: Pipelines and composite estimators**](https://scikit-learn.org/stable/modules/compose.html): wie du Vorbereitung und Modelle mit `Pipeline` und `ColumnTransformer` verbindest.
- [**scikit-learn: Tuning the hyperparameters of an estimator**](https://scikit-learn.org/stable/modules/grid_search.html): Gitter-Suche, Zufalls-Suche und wie sie **Kreuz-Validierung** nutzen.
- [**scikit-learn: Polynomial features**](https://scikit-learn.org/stable/modules/preprocessing.html#polynomial-features): wie du Merkmale um Quadrate und Produkte erweiterst.
- [**Common pitfalls in the interpretation of coefficients of linear models**](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html): warum korrelierte Merkmale die Koeffizienten unsicher machen.
- [**House Sales in King County, USA auf Kaggle**](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction): der Original-Datensatz.
