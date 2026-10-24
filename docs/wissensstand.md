# Wissensstand Bachelorarbeit – Justus Röder

Stand: 24.10.2026. Sammeldokument für Feedback, Entscheidungen, geprüfte Quellen und offene Punkte.
Rohtranskript der Sprachnachricht: `feedback_dozentin_transkript_roh.txt`

---

## 1. Ausgangslage

**Bisheriger Titel (Exposé):** *Standalone vs. Incremental CVA Proxies for Real-Time Pre-Trade Pricing
of FX Forwards and Swaps – A Simulation-Based Analysis of Under- and Overcharging Relative to the
Full Portfolio CVA*

**Code im Repo (`cva/`, `main.py`):**
- Drei-Faktor-Modell: Hull-White USD, Hull-White EUR (mit Quanto-Drift), EUR/USD-Spot (GBM);
  Nelson-Siegel-Startkurven, umschaltbar deterministisch/stochastisch
- 5 synthetische Netting-Sets, unbesichert; Hazard-Kurve stückweise konstant, R = 40 %
- Drei Größen: wahrer inkrementeller CVA (volle Simulation), Standalone-Proxy, inkrementeller
  Proxy (Szenario-Matrix + Interpolation des neuen Trades)
- Interpolation linear und Chebyshev, anisotrope Gitter (Spot affin → 2 Knoten reichen)
- 5 Validierungstests bestanden (Kurven, FX-Martingal, Quanto, Zero-Vol, Konvergenz)

**Bisherige Ergebnisse (`results/cva_results.csv`):**

| | mittlerer relativer Fehler |
|---|---|
| Standalone-Proxy | ca. 51–59 % (risikoreduzierender Trade > 200 %) |
| Inkrementeller Proxy | ca. 0,002–0,03 % |

- Swap 1y/3y: Exposure vor Settlement unter stochastischen Zinsen ca. 3,9× höher
  (78 % zinsgetrieben), aber inkrementeller CVA unterscheidet sich zwischen den Zinsmodellen nur
  um ca. 4 %, weil die Exposure nach Settlement dominiert.

---

## 2. Feedback der Dozentin (Sprachnachricht 11.08.2026)

**Gesamturteil:** Thema „richtig gut“, vorläufige Gliederung gut, Motivation/Problemstellung in
Ordnung, Framework (Kap. 3) gut zitiert. Nach Überarbeitung → **Registrierung**. Diesen Monat
keine Termine → schriftlich abstimmen.

**To-dos:**
1. **Analysezeitraum** festlegen und nennen.
2. **Data Sources** als eigenen Punkt; welche Daten, welcher Zeitraum.
3. **Allgemeine Forschungsfrage kürzen** (zu lang).
4. **Konkretes Ziel:** ein Hauptziel + Teilziele/Forschungsfragen.
5. **Stand der Literatur vor die Zielsetzung:** Motivation/Problem → Literatur → Ziele/Fragen.
6. **Research Gaps** jeweils direkt bei den einzelnen Literaturpunkten formulieren, nicht gesammelt.
7. **Results inkl. Diskussion:** Ergebnisse mit Literatur verknüpfen, Fragen/Hypothesen beantworten.
8. Startpaket durchsehen.
9. Überarbeitetes Exposé einreichen → registrieren.

---

## 3. Kritik an der bisherigen Forschungslücke

1. **Ergebnisse vorbestimmt.** Standalone ≥ inkrementell folgt aus der Netting-Ungleichung
   (Subadditivität von max(·,0)); die Teilfrage „überberechnet er systematisch?“ ist durch einen
   Beweis beantwortet. Der inkrementelle Proxy hat nur Interpolationsfehler, und der ist ~0.
   Kernaussage wäre nur „Netting ignorieren ist teuer“ – bekannt.
2. **Falsche Literaturaussage (Exposé 2.2):** „Alle Beiträge behandeln Standalone/inkrementellen
   CVA als exakt berechnete Größen.“ Stimmt nicht: Pykhtin & Rosen (2010), Abschnitt 7
   „Analytical CVA Contributions under a Normal Approximation“, schätzen Beiträge schnell
   außerhalb des Simulationssystems (im Volltext geprüft).
3. **Fehlende, sehr nahe Quelle:** Kenyon & Green (2014/15), „Efficient XVA Management“:
   Trade-Level-Regression als Wertproxy, szenarioweise aggregiert; inkrementelle Änderungen laut
   Abstract „exactly and simply“. Prinzipiell der inkrementelle Proxy mit Regression statt
   Interpolation.
4. **Inkonsistenz:** Lücke 2.5 nennt das FX-spezifische Zusammenspiel Spot/Zins, die
   Forschungsfrage enthält es nicht.

---

## 4. Neue Forschungsrichtung (entschieden): Intraday-Verzerrung

**Idee:** Proxies werden nachts aus der Szenario-Matrix des Nacht-Batches berechnet und untertags
nicht aktualisiert. Gemessen wird, wie stark dadurch über einen Beobachtungszeitraum unter- oder
überberechnet wird – auf echten (anonymisierten) Trade-Strömen einer Bank.

**Warum besser:**
- Fehler hängt von echten Abläufen ab (Trades pro Tag/Batch, Richtung, Marktbewegung) und ist
  nicht vorab ausrechenbar → echte empirische Frage.
- Unter-/Überberechnung wird substanziell. **Hypothese:** gleichgerichtete Trade-Batches
  (z. B. Kunde kauft wiederholt EUR) → systematisches **Undercharging**, weil die Matrix die
  heutige Exposure nicht kennt; gegenläufige → **Overcharging**.
- Erfüllt Dozentin-Punkte 1 und 2 (Zeitraum, Data Sources) automatisch.
- Bestehender Code fast vollständig weiterverwendbar.

**Fehlerzerlegung je Trade:**
1. Netting ignoriert (Standalone-Proxy, Referenz)
2. Veraltetes Portfolio (Trades seit dem Nacht-Lauf fehlen in der Matrix)
3. Veralteter Markt (Spot/Zinsen seit Tagesabschluss bewegt)

Hinweis: Der Standalone-Proxy hängt nicht vom Portfolio ab; bei ihm veraltet nur der Markt.

**Auswertung:** Summen über den Zeitraum (EUR unter-/überberechnet), Verteilung, Abhängigkeit von
Anzahl Trades im Batch, Richtung, Größe der Marktbewegung, Produkt (Swap/Forward).

### Entwurf Forschungsfrage (englisch, wie Exposé)

> *How large and in which direction are the CVA pricing errors that arise when incoming FX trades
> are charged against an overnight scenario matrix that is not updated intraday?*

**Hauptziel:** Intraday-Fehler auf echten Trade-Strömen beziffern und in Ursachen zerlegen.

**Teilfragen:**
1. Wie groß ist der Fehler durch fehlende Trades seit dem Nacht-Lauf vs. durch Marktbewegungen?
2. Systematisches Unter-/Überberechnen abhängig von der Richtung der Trade-Batches?
3. Wie schneidet der veraltete inkrementelle Proxy gegen den Standalone-Proxy als Referenz ab?

---

## 5. Daten und Annahmen

**Aus der Bank (anonymisiert; keine Bankalgorithmen):**
- ca. 100 Counterparties; nicht jeder Kunde handelt täglich, oft batchweise (z. B. 1×/Woche
  10 Trades) – für den Effekt reicht: mehrere Trades zwischen zwei Nacht-Läufen
- je Trade: Kunden-ID, EUR/USD-Produkt (Forward/Swap), Laufzeit(en), Notional, Kurs, Zeitstempel
- Spot und Zins zum Trade-Zeitpunkt (live)

**Zeitraum:** 6 Monate Historie, **Monat 7 als Evaluationsmonat**.
- **Wichtig:** Zusätzlich **Snapshot aller offenen Trades** zum letzten Tag von Monat 6
  anfordern. Historie allein reicht nicht – ältere, noch laufende Swaps/Forwards (bis 5 J.)
  würden fehlen, Netting wäre falsch.
- Historie nutzen, um Handelsverhalten zu beschreiben.
- Option: mehr als einen Evaluationsmonat (Robustheit gegen ruhigen/unruhigen Einzelmonat).

**CSA:** Annahme unbesichert. Besser: CSA-Kennzeichen ja/nein je Counterparty → nur unbesicherte
auswerten (saubere Abgrenzung statt Überschätzung bei besicherten Kunden).

**Kreditqualität:** Bevorzugt Rating-Klasse je Counterparty → öffentliche Durchschnittsspreads je
Rating als Hazard. Fallback zufällig → dann primär **relative** Fehler berichten (CVA skaliert
etwa proportional zum Spread). Echte Ratings/Kurven notfalls von der Bank verfügbar.

**Marktdaten:** Nacht-Läufe mit öffentlichen Tageskurven (EZB-Zinsstrukturkurve EUR;
Treasury/SOFR von FRED für USD) oder Bankdaten; zum Trade-Zeitpunkt Spot/Zins der Bank.
Intraday-Marktfehler daher über Spot + Kurzfristzins. FX-Vola historisch aus Spot geschätzt
oder Annahme.

**Datenanfrage an die Bank (Checkliste):**
- [ ] alle Trades der letzten 6–7 Monate inkl. Zeitstempel und Kurs
- [ ] Snapshot offener Trades zum Stichtag (Ende Monat 6)
- [ ] Netting-Set-Zuordnung
- [ ] CSA-Kennzeichen
- [ ] Rating-Klasse
- [ ] Freigabe/Vertraulichkeit (NDA, ggf. Sperrvermerk, Veröffentlichung aggregierter Ergebnisse)
- [ ] (allgemeine) Bestätigung, dass Pre-Trade-CVA untertags gegen Nacht-Stand gerechnet wird

---

## 6. Literatur

### 6.1 Geprüft (Volltext oder Abstract gelesen)

| Quelle | Befund | Status |
|---|---|---|
| Pykhtin & Rosen (2010), J. Credit Risk 6(4); FEDS 2010-10 | Definition inkrementeller CVA, „appropriate for pricing new trades“, nicht additiv, reihenfolgeabhängig; **Abschn. 7: Normal-Approximation** für schnelle Beiträge | Volltext geprüft |
| Vrins & Gregory (2011), Risk | Fehler < 2 %; CCS inkrementell zu Netting-Set mit 17 Trades; negativer inkrementeller CVA durch Netting | Volltext geprüft |
| Kenyon & Green (2014), arXiv:1412.5332 / J. Risk 2015 | Trade-Level-Regression + Global Conditioning; inkrementell „exactly and simply“ | Abstract; **ins Exposé aufnehmen** |
| Crépey & Dixon (2019), arXiv:1901.11081 | GP-Metamodell für MtM-Cube, Portfolio-CVA (IRS), kein inkrementeller Fokus | Abstract; in 2.3 ergänzen |
| Glau, Pachon & Pötz (2019), arXiv:1912.01280 | Exposure-Speed-up per Funktionsapproximation (Equity, Bermudan Swaption); kein inkrementeller/Intraday-Fokus | Abstract |
| Lu & Juan (2010), arXiv:1010.1689 | Szenario-Wiederverwendung für inkrementellen CVA | Quelle existiert |
| Biagini, Gnoatto & Oliva (2021), arXiv:1905.11328 | Beispiel: Summe Standalone 457,49 / Portfolio 232,69 / inkrementell 84,52 EUR | Zahlen bestätigt |
| Albanese et al. (2020/21), arXiv:2009.00368 | Inkrementelles XVA-Profil für neuen Swap, Common Random Numbers | Abstract |

### 6.2 Praxis-/Vendor-Quellen (nur Relevanzbeleg, nicht zitierfähig als Forschung)
- Numerix Oneview, Murex MX.3, OSTTRA: Real-Time-Pre-Deal-XVA (standalone/inkrementell/marginal).
  → belegt Praxisrelevanz, **aber** zeigt auch, dass manche Systeme intraday aktualisieren →
  Prämisse „keine Intraday-Aktualisierung“ an Bankpraxis festmachen.
- Alef Analytics (CCR/CVA-Serie): Reihenfolgeabhängigkeit mit zwei gegenläufigen Trades.
- IHS Markit xVA-E-Book: ~10 Mio. Trade-Bewertungen für eine CVA-Replikation (100.000 Trades,
  200 Zeitpunkte).

### 6.3 Weitere Exposé-Quellen (noch nicht einzeln geprüft)
Gregory (2020), Green (2015), Ruiz (2015), Cesari et al. (2009), Longstaff & Schwartz (2001),
Zeron & Ruiz (2018, 2020), Gaß et al. (2018), Deelstra et al. (2024), Capriotti et al. (2011),
Zeitsch (2017), Ranaldo (2022), Gnoatto & Seiffert (2020), Brigo & Pallavicini (2013),
Hull & White (1990).

### 6.4 Literaturstand zur neuen Lücke (vorläufig, Websuche 24.10.2026)
- **Kein** akademischer Beitrag gefunden, der den Fehler durch veraltete Portfolio-/Marktdaten
  beim Pre-Trade-CVA quantifiziert.
- Reihenfolgeabhängigkeit bekannt (Pykhtin & Rosen), aber nicht empirisch an Intraday-Sequenzen
  gemessen.
- **Noch nicht sicher:** Google Scholar, SSRN, arXiv systematisch prüfen; Risk.net (Paywall) –
  Justus sucht dort selbst: „pre-deal CVA“, „incremental XVA intraday“.

---

## 7. Nächste Schritte

1. [ ] Systematische Literaturprüfung (Scholar, SSRN, arXiv) → Lücke absichern
2. [ ] Mail an Dozentin mit neuer Richtung (vor dem Umbau) – Entwurf durch Claude
3. [ ] Datenanfrage an die Bank (Checkliste Abschnitt 5)
4. [ ] Exposé umbauen: Literatur vor Zielsetzung, Lücken je Literaturpunkt, Zeitraum + Data
       Sources, kurze Forschungsfrage, Hauptziel + Teilziele, Results inkl. Diskussion;
       Pykhtin & Rosen Abschn. 7, Kenyon & Green, Crépey & Dixon einarbeiten
5. [ ] Code erweitern: Nacht-Matrix je Tag, sequenzielles Bepreisen, Einlesen der Bankdaten,
       Fehlerzerlegung
6. [ ] Startpaket der Dozentin sichten; formale Vorgaben klären

---

## 8. Hinweise zur Arbeitsweise
- Claude in Chrome ist in der Cloud-Session nicht verfügbar; Recherche per Websuche (nur frei
  zugängliche Inhalte). Paywall-PDFs liefert Justus oder Desktop-App mit Chrome-Erweiterung.
- Jede Quelle wird vor Aufnahme ins Exposé geprüft.
