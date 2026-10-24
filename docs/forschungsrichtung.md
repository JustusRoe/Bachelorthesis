# Neue Forschungsrichtung (Stand 24.10.2026)

## Entscheidung
Weg vom reinen Vergleich Standalone vs. inkrementeller Proxy (Ergebnis theoretisch
vorbestimmt bzw. Interpolationsfehler ~0 %) hin zur **intraday-Verzerrung**:
Proxies werden nachts berechnet (Szenario-Matrix aus Nacht-Batch), untertags nicht
aktualisiert. Gemessen wird Under-/Overcharging über einen Beobachtungszeitraum.

## Daten (Bank, anonymisiert)
- ca. 100 Counterparties, 1 Monat
- je Trade: anonyme Kunden-ID, EUR/USD-Produkt (Forward/Swap), Laufzeit(en), Notional, Kurs, Zeitstempel
- Spot + Zinsen zum Trade-Zeitpunkt (live)
- Bankalgorithmen werden NICHT genutzt; Proxies + "wahrer" CVA aus eigenem Modell
- Zeitraum: 6 Monate Historie + 7. Monat als Evaluationsmonat
- Anfangsbestand: Snapshot ALLER offenen Trades zum Ende Monat 6 anfragen
  (Historie allein reicht nicht: ältere, noch offene Swaps/Forwards fehlen sonst)
- CSA: Annahme unbesichert; bevorzugt CSA-Flag von Bank → nur unbesicherte CPs auswerten
- Kreditqualität: bevorzugt Rating-Klasse je CP → Hazard aus öffentlichen Spreads je Rating;
  Fallback zufällig → dann primär relative Fehler berichten
- Marktdaten: Live-Spot/Zins von Bank zum Trade; Kurven tagesweise öffentlich (EZB/FRED) oder Bank;
  FX-Vola historisch geschätzt oder Annahme

## Fehlerzerlegung je Trade
1. Netting ignoriert (Standalone-Proxy, Referenz)
2. Veraltetes Portfolio (heutige Trades fehlen in Matrix)
3. Veralteter Markt (Spot/Zinsen seit Tagesabschluss bewegt)

Hypothese: gleichgerichtete Trade-Batches → systematisches Undercharging,
gegenläufige → Overcharging.

## Literaturstand (vorläufig, Websuche 24.10.2026)
- Kein akademischer Beitrag gefunden, der den Fehler durch veraltete Portfolio-/Marktdaten
  beim Pre-Trade-CVA quantifiziert.
- Order-Dependence des inkrementellen CVA bekannt (Pykhtin & Rosen 2010), aber nicht empirisch
  für Intraday-Sequenzen gemessen.
- Vendoren (Numerix, Murex, OSTTRA) bieten Real-Time-Pre-Deal-XVA → Praxisrelevanz belegt,
  aber auch: Annahme "keine Intraday-Aktualisierung" muss mit Bankpraxis begründet werden.
- Zu ergänzen: Kenyon & Green (2014), Pykhtin & Rosen Abschn. 7 (Normal-Approximation),
  Crépey & Dixon (2019).
- Noch offen: SSRN/Google Scholar/Risk.net (Paywall) systematisch prüfen.
