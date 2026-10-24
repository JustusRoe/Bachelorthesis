# Anweisungen für Claude – Bachelorarbeit Justus Röder

## Projekt in einem Satz
Bachelorarbeit (Quantitative Finance): Wie stark werden eingehende EUR/USD-Forwards und -Swaps
beim Pre-Trade-CVA unter- oder überberechnet, wenn sie untertags gegen eine Szenario-Matrix aus
dem Nacht-Batch bepreist werden, die nicht aktualisiert wird? Gemessen auf anonymisierten
Bankdaten, gerechnet mit einem eigenen Drei-Faktor-Modell.

**Maßgeblicher Projektstand:** `docs/wissensstand.md`. Vor jeder inhaltlichen Arbeit lesen.
Neue Entscheidungen, geprüfte Quellen und erledigte Schritte dort eintragen (Datum dazu).

## Sprache
- Mit Justus auf **Deutsch** kommunizieren, direkt und knapp.
- Exposé und Thesis sind auf **Englisch**: akademischer Stil, präzise, keine Floskeln,
  keine Übertreibungen („novel“, „first ever“) ohne Beleg.

## Quellen – oberste Regel
- **Niemals Quellen erfinden.** Keine Autoren, Jahre, Titel, Journals, Seitenzahlen oder
  Zahlen aus dem Gedächtnis als belegt ausgeben.
- Eine Quelle gilt erst als belegt, wenn sie in dieser Arbeit tatsächlich gelesen wurde
  (Volltext oder mindestens Abstract) – und die zitierte Aussage dort steht.
- Status jeder Quelle kennzeichnen: **geprüft (Volltext)**, **geprüft (Abstract)**,
  **ungeprüft**. Ungeprüftes nie ohne Hinweis ins Exposé schreiben.
- Bei konkreten Zahlen oder Zitaten die Stelle angeben (Abschnitt/Seite/Tabelle).
- Vendor-, Blog- und Firmenseiten (Numerix, Murex, OSTTRA, Alef …) nur als Praxisbeleg,
  nie als Forschungsstand.
- Paywall-Quellen (Risk.net, Journal of Credit Risk, Wiley, Springer): offen sagen, dass nur
  Metadaten/Abstract verfügbar sind, und Justus um das PDF bitten.
- Nach jeder Recherche: Quellen mit Links auflisten.

## Forschungslücke – kritisch bleiben
- Die Lücke muss belastbar sein. Aktiv nach Arbeiten suchen, die sie **schließen** oder
  einschränken, und diese offen benennen – lieber jetzt als im Gutachten.
- „Nichts gefunden“ ist kein Beweis. Formulieren als „nach systematischer Suche in … nicht
  gefunden“ und die Suche (Datenbanken, Suchbegriffe, Datum) dokumentieren.
- Schwächen in Argumentation, Design oder Daten ehrlich ansprechen, auch ungefragt.
  Keine Bestätigung um der Bestätigung willen.
- Aussagen, die schon aus der Theorie folgen (z. B. Standalone ≥ inkrementell durch
  Netting-Ungleichung), nicht als empirischen Befund verkaufen.

## Vorgaben der Dozentin (Feedback 11.08.2026)
Bei jeder Arbeit am Exposé/der Thesis einhalten:
1. Analysezeitraum explizit nennen.
2. Data Sources als eigener Punkt (welche Daten, welcher Zeitraum).
3. Forschungsfrage kurz halten.
4. Ein Hauptziel + Teilziele/Teilfragen.
5. Stand der Literatur **vor** Zielsetzung: Motivation → Literatur → Ziele/Fragen.
6. Research Gaps **direkt bei jedem Literaturstrang** formulieren, nicht gesammelt am Ende.
7. Results **inklusive Diskussion**: Ergebnisse mit Literatur verknüpfen, Forschungsfragen
   und Hypothesen explizit beantworten.
Formale Vorgaben (Zitierstil, Länge, Gliederung) aus dem Startpaket der Dozentin haben Vorrang;
solange nicht bekannt: Autor-Jahr-Zitation wie im bisherigen Exposé.

## Daten und Vertraulichkeit
- Bankdaten sind vertraulich: **nie committen, nie pushen, nie an externe Dienste schicken**,
  nie wörtlich in Antworten, Docs oder Plots mit Einzelwerten zitieren.
- Bankdaten nur unter `data/raw/` ablegen (per `.gitignore` ausgeschlossen). Vor jedem Commit
  prüfen, dass keine Bankdaten im Diff sind.
- Bankalgorithmen werden nicht verwendet oder beschrieben; alle Proxies und der wahre CVA
  kommen aus dem eigenen Modell.
- Ergebnisse nur aggregiert/anonymisiert berichten.
- Öffentliche Marktdaten (EZB, FRED, Dukascopy …) mit Quelle und Abrufdatum dokumentieren.

## Modell und Code
- Python; Engines in `cva/`, Orchestrierung in `main.py`, **alle Parameter in `cva/config.py`**
  (mit Kennzeichnung [CONVENTION] / [CHOICE] / [SYNTHETIC] bzw. [DATA]).
- Die Validierungssuite (`cva/validation.py`) läuft vor jedem Ergebnis. Schlägt ein Test fehl,
  werden keine Ergebnisse berichtet. Neue Bausteine bekommen eigene Validierungstests.
- Fester Seed, Common Random Numbers für Differenzen (inkrementeller CVA, Proxy vs. wahr).
- Vorzeichenkonvention: Fehler = Proxy − wahrer Wert; **positiv = Overcharging,
  negativ = Undercharging**. Inkrementeller CVA = neuer Trade wird **zuletzt** hinzugefügt.
- Monte-Carlo-Fehler immer mitberichten (Standardfehler/Konfidenzintervall); Effekte kleiner
  als das MC-Rauschen nicht interpretieren.
- Zahlen in Texten nur aus tatsächlich gelaufenen Ergebnissen (`results/`), nie geschätzt.
- Stil des bestehenden Codes beibehalten; Änderungen vor dem Commit laufen lassen.

## Arbeitsweise
- Bei unklaren Entscheidungen, die die Arbeit inhaltlich festlegen (Thema, Daten, Hypothesen,
  Gliederung), Justus fragen statt annehmen. Kleinigkeiten selbst entscheiden und erwähnen.
- Vorschläge, die Justus noch nicht bestätigt hat, als Vorschlag kennzeichnen.
- Größere Aufgaben: erst kurzer Plan, dann umsetzen.
- Git: Arbeit committen und pushen, klare Commit-Messages; keine Bankdaten, keine großen
  Binärdateien außer Ergebnis-Plots.
- Claude in Chrome ist in Cloud-Sessions nicht verfügbar → Websuche nutzen und Grenzen
  (Paywall) offen sagen.
