# Wissensstand Bachelorarbeit – Justus Röder

Stand: 24.10.2026. Sammeldokument für das Projekt in seiner **neuen Ausrichtung**: Feedback,
Entscheidungen, Daten, geprüfte Quellen und offene Punkte.
Rohtranskript der Sprachnachricht: `feedback_dozentin_transkript_roh.txt`

---

## 1. Thema

**Arbeitstitel (Entwurf):** Intraday-Verzerrung von CVA-Proxies beim Pre-Trade-Pricing von
EUR/USD-Forwards und -Swaps – eine empirische Analyse von Under- und Overcharging auf
anonymisierten Bankdaten

**Idee:** Banken berechnen die Szenario-Matrix ihrer Netting-Sets im Nacht-Batch. Untertags
werden neue Trades gegen diesen Nacht-Stand bepreist, ohne dass die Matrix aktualisiert wird.
Die Arbeit misst, wie stark dadurch über einen Beobachtungszeitraum unter- oder überberechnet
wird – auf echten, anonymisierten Trade-Strömen einer Bank, gerechnet mit einem eigenen Modell.

**Warum relevant:**
- Der Fehler hängt von echten Abläufen ab (Trades pro Batch, Richtung, Marktbewegung) und ist
  nicht vorab ausrechenbar → echte empirische Frage.
- Unter-/Überberechnung hat direkte wirtschaftliche Folgen: Undercharging = Kreditrisiko nicht
  vergütet; Overcharging = Quote nicht wettbewerbsfähig.
- Reihenfolgeabhängigkeit des inkrementellen CVA ist theoretisch bekannt, aber nicht an realen
  Intraday-Sequenzen gemessen (siehe Abschnitt 6).

---

## 2. Forschungsfrage und Ziele (Entwurf)

> *How large and in which direction are the CVA pricing errors that arise when incoming FX trades
> are charged against an overnight scenario matrix that is not updated intraday?*

**Hauptziel:** Den Intraday-Fehler auf echten Trade-Strömen beziffern und in seine Ursachen
zerlegen.

**Teilfragen:**
1. Wie groß ist der Fehler durch fehlende Trades seit dem Nacht-Lauf und wie groß der durch
   Marktbewegungen untertags?
2. Wird systematisch unter- oder überberechnet, abhängig von der Richtung der Trade-Batches?
3. Wie schneidet der veraltete inkrementelle Proxy gegen den Standalone-Proxy als Referenz ab?

**Hypothesen:**
- H1: Gleichgerichtete Trade-Batches (z. B. Kunde kauft wiederholt EUR) führen zu
  systematischem **Undercharging**, weil die Nacht-Matrix die gestiegene Exposure nicht kennt.
- H2: Gegenläufige Trades führen zu **Overcharging**, weil der Netting-Effekt der heutigen
  Trades fehlt.
- H3: Der Fehler wächst mit der Anzahl der Trades seit dem Nacht-Lauf und mit der Größe der
  Marktbewegung.

---

## 3. Methodik

**Ablauf je Evaluationstag:**
1. Nacht-Lauf: Szenario-Matrix aller offenen Trades je Netting-Set (Marktstand Tagesabschluss)
2. Untertags je eingehendem Trade:
   - **Standalone-Proxy:** Trade isoliert bepreist (kein Netting)
   - **Veralteter inkrementeller Proxy:** Trade gegen die Nacht-Matrix bepreist
   - **Wahrer inkrementeller CVA:** volle Simulation mit aktuellem Portfolio (inkl. der heute
     bereits gebuchten Trades) und aktuellem Marktstand zum Trade-Zeitpunkt
3. Fehler = Proxy − wahrer Wert (positiv = Overcharging, negativ = Undercharging)

**Fehlerzerlegung je Trade:**
1. Netting ignoriert (Standalone-Proxy, Referenz)
2. Veraltetes Portfolio (Trades seit dem Nacht-Lauf fehlen in der Matrix)
3. Veralteter Markt (Spot/Zinsen seit Tagesabschluss bewegt)

Hinweis: Der Standalone-Proxy hängt nicht vom Portfolio ab; bei ihm veraltet nur der Markt.

**Auswertung:**
- Summen über den Evaluationszeitraum (EUR unter-/überberechnet), je Counterparty und gesamt
- Verteilung der Fehler, Anteil der Ursachen
- Abhängigkeit von Anzahl Trades im Batch, Richtung, Größe der Marktbewegung, Produkt
  (Swap/Forward), Laufzeit, Rating-Klasse
- Diskussion der Ergebnisse direkt mit der Literatur verknüpft (Vorgabe der Dozentin)

**Modellbasis (im Repo vorhanden, wird weiterverwendet):**
- Drei-Faktor-Modell: Hull-White USD, Hull-White EUR (mit Quanto-Drift), EUR/USD-Spot
- Analytische Bewertung von FX-Forwards und -Swaps, kein genestetes Monte Carlo
- Netting-aware CVA, Standalone- und inkrementeller Proxy, Interpolation (linear/Chebyshev)
- Validierungssuite (Kurven, FX-Martingal, Quanto, Zero-Vol, Konvergenz)

**Code-Erweiterungen (zu bauen):**
- Einlesen der Bankdaten (Bestand, Trades, Marktdaten)
- Kalibrierung an echte Kurven statt Nelson-Siegel mit synthetischen Parametern
- Nacht-Matrix je Tag, sequenzielles Bepreisen untertags
- Wahrer CVA zum Trade-Zeitpunkt, Fehlerzerlegung, Auswertung über den Zeitraum

---

## 4. Daten und Annahmen

**Aus der Bank (anonymisiert; keine Bankalgorithmen):**
- ca. 100 Counterparties
- je Trade: Kunden-ID, EUR/USD-Produkt (Forward/Swap), Laufzeit(en), Notional, Kurs, Zeitstempel
- Spot und Zins zum Trade-Zeitpunkt (live)
- Nicht jeder Kunde handelt täglich, oft batchweise (z. B. 1×/Woche 10 Trades) – für den Effekt
  reicht: mehrere Trades zwischen zwei Nacht-Läufen

**Zeitraum:** 6 Monate Historie, **Monat 7 als Evaluationsmonat**.
- **Wichtig:** Zusätzlich **Snapshot aller offenen Trades** zum letzten Tag von Monat 6.
  Historie allein reicht nicht – ältere, noch laufende Swaps/Forwards (bis 5 J.) würden fehlen,
  Netting wäre falsch.
- Historie nutzen, um das Handelsverhalten zu beschreiben (Frequenz, Richtung, Batches).
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
- [ ] (allgemeine) Bestätigung, dass Pre-Trade-CVA untertags gegen den Nacht-Stand gerechnet wird

---

## 5. Feedback der Dozentin (Sprachnachricht 11.08.2026) – Umsetzung im neuen Exposé

**Gesamturteil:** Thema „richtig gut“, Gliederung gut, Motivation in Ordnung, Framework gut
zitiert. Nach Überarbeitung → **Registrierung**. Diesen Monat keine Termine → schriftlich.
Achtung: Die neue Richtung ist eine Themenänderung → Dozentin vorab informieren.

| # | Vorgabe | Umsetzung im neuen Projekt |
|---|---|---|
| 1 | Analysezeitraum nennen | Monat 7 (Evaluation), Monate 1–6 als Historie |
| 2 | Data Sources als eigener Punkt | Bankdaten (anonymisiert) + öffentliche Marktdaten |
| 3 | Forschungsfrage kürzen | Entwurf in Abschnitt 2 |
| 4 | Hauptziel + Teilziele | Abschnitt 2 |
| 5 | Literatur vor die Zielsetzung | Gliederung: Motivation → Literatur → Ziele/Fragen |
| 6 | Research Gaps je Literaturpunkt | Lücke direkt bei jedem Strang formulieren |
| 7 | Results inkl. Diskussion | Ergebnisse mit Literatur verknüpfen, Hypothesen beantworten |
| 8 | Startpaket durchsehen | offen |
| 9 | Einreichen → registrieren | nach Umbau |

---

## 6. Literatur

### 6.1 Geprüfte Quellen und Rolle im neuen Projekt

| Quelle | Befund | Rolle |
|---|---|---|
| Pykhtin & Rosen (2010), J. Credit Risk 6(4); FEDS 2010-10 | Definition inkrementeller CVA, „appropriate for pricing new trades“, nicht additiv, **reihenfolgeabhängig**; Abschn. 7: Normal-Approximation für schnelle Beiträge | Kernquelle: Reihenfolgeabhängigkeit = theoretische Basis der Intraday-Verzerrung |
| Vrins & Gregory (2011), Risk | negativer inkrementeller CVA durch Netting; Proxy-Fehler < 2 % (CCS inkrementell, 17 Trades) | Netting-Effekte, Vorzeichen |
| Lu & Juan (2010), arXiv:1010.1689 | Szenario-Wiederverwendung: neuer Trade in gespeicherten Szenarien bepreist | Mechanismus der Nacht-Matrix |
| Kenyon & Green (2014), arXiv:1412.5332 / J. Risk 2015 | Trade-Level-Regression + Global Conditioning; inkrementelle Änderungen „exactly and simply“, Allokation inkrementell im Tagesgeschäft | nahe Quelle zu inkrementellem Pricing im Tagesablauf (Volltext prüfen) |
| Biagini, Gnoatto & Oliva (2021), arXiv:1905.11328 | Beispiel: Summe Standalone 457,49 / Portfolio 232,69 / inkrementell 84,52 EUR | Größenordnung Netting-Effekt |
| Albanese et al. (2020/21), arXiv:2009.00368 | inkrementelles XVA-Profil für neuen Trade, Common Random Numbers | inkrementelles Pricing, Methodik |
| Crépey & Dixon (2019), arXiv:1901.11081 | GP-Metamodell für MtM-Cube, Portfolio-CVA | Proxy-Literatur (Randstrang) |
| Glau, Pachon & Pötz (2019), arXiv:1912.01280 | Exposure-Speed-up per Funktionsapproximation | Proxy-Literatur (Randstrang) |

### 6.2 Praxis-/Vendor-Quellen (Relevanzbeleg, keine Forschung)
- Numerix Oneview, Murex MX.3, OSTTRA: Real-Time-Pre-Deal-XVA (standalone/inkrementell/marginal)
  → belegt Praxisrelevanz, **aber** manche Systeme aktualisieren intraday → Prämisse „keine
  Intraday-Aktualisierung“ an der Praxis der Bank festmachen.
- Alef Analytics (CCR/CVA-Serie): Reihenfolgeabhängigkeit an zwei gegenläufigen Trades.
- IHS Markit xVA-E-Book: ~10 Mio. Trade-Bewertungen für eine CVA-Replikation → Begründung,
  warum Banken nicht bei jedem Trade neu simulieren.

### 6.3 Grundlagenquellen (aus altem Exposé, weiter nutzbar; noch nicht einzeln geprüft)
Gregory (2020), Green (2015), Ruiz (2015), Cesari et al. (2009), Zeitsch (2017), Hull & White
(1990), Gnoatto & Seiffert (2020), Ranaldo (2022), Brigo & Pallavicini (2013).
Proxy-/Interpolationsstrang (an Bedeutung verloren, ggf. kürzen): Longstaff & Schwartz (2001),
Zeron & Ruiz (2018, 2020), Gaß et al. (2018), Deelstra et al. (2024), Capriotti et al. (2011).

### 6.4 Stand zur Forschungslücke (vorläufig, Websuche 24.10.2026)
- **Kein** akademischer Beitrag gefunden, der den Fehler durch veraltete Portfolio-/Marktdaten
  beim Pre-Trade-CVA quantifiziert.
- Reihenfolgeabhängigkeit bekannt (Pykhtin & Rosen), aber nicht empirisch an
  Intraday-Sequenzen gemessen.
- **Noch nicht sicher:** Google Scholar, SSRN, arXiv systematisch prüfen; Risk.net (Paywall) –
  Justus sucht selbst: „pre-deal CVA“, „incremental XVA intraday“.

---

## 7. Nächste Schritte

1. [ ] Systematische Literaturprüfung (Scholar, SSRN, arXiv) → Lücke absichern
2. [ ] Mail an Dozentin mit neuer Richtung (vor dem Umbau) – Entwurf durch Claude
3. [ ] Datenanfrage an die Bank (Checkliste Abschnitt 4)
4. [ ] Neues Exposé schreiben nach Abschnitt 5
5. [ ] Code erweitern (Abschnitt 3)
6. [ ] Startpaket der Dozentin sichten; formale Vorgaben klären

---

## 8. Hinweise zur Arbeitsweise
- Claude in Chrome ist in der Cloud-Session nicht verfügbar; Recherche per Websuche (nur frei
  zugängliche Inhalte). Paywall-PDFs liefert Justus oder Desktop-App mit Chrome-Erweiterung.
- Jede Quelle wird vor Aufnahme ins Exposé geprüft.
