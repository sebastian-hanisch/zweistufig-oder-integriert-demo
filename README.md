# Zweistufig oder integriert: auf die Modellklasse kommt es an – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-zweistufig-oder-integriert-demo.streamlit.app/)** -->

Demo der Reihe **Entscheidungsorientiertes Lernen** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie baut auf `newsvendor-direkt-demo` (direkt auf den Kosten lernen) und `spo-kuerzester-weg-demo` (SPO+ im Gitter) auf und stellt die Frage, die beide offen lassen: **Wann gewinnt welcher Weg?** Vehikel ist dasselbe abstrakte **5 × 5 Gitter** wie dort (Wege von links oben nach rechts unten, 40 Bögen, 70 Wege) und, als zweite, kleinere Welt, der **Newsvendor mit Merkmalen**. Verglichen werden drei Wege, aus Merkmalen eine Entscheidung zu machen: **zweistufig** (erst die Kosten schätzen, dann den Löser fragen; im Text ETO nach „estimate, then optimize“), **auf der Entscheidung trainiert** (SPO+, der Löser läuft im Training) und **gewichtete Stichprobe** (ohne Modell).

## Warum dieses Problem

Zwei Arbeiten, die ich im Volltext gelesen habe (Hu/Kallus/Mao, Elmachtoub/Lam/Zhang/Zhao), ordnen die drei Wege asymptotisch ein, und beide hängen an einer Bedingung, die in echten Daten selten prüfbar ist: ob die Modellklasse die Wahrheit enthält. Passt sie, ist die einfache zweistufige Regel in ihren Sätzen die bessere (bei Hu/Kallus/Mao als Ratenvergleich unter einer Rauschbedingung); passt sie nicht, kehrt sich die Ordnung um (Satz bei Elmachtoub/Lam/Zhang/Zhao, bei Hu/Kallus/Mao nur im Gitterversuch beobachtet; genaue Aussagen unter „Modell“). Die Demo misst, was in endlichen Stichproben einer Familie von Welten passiert, in denen man die Klasse einstellen kann. Die Kernfrage: **Welche Regel gewinnt, wenn die Klasse passt, welche, wenn sie nicht passt, und was ändert die Datenmenge?**

- **Passt die Klasse, gewinnt in dieser Welt keine Alternative etwas gegenüber KQ.** Bei Grad 1 (die lineare Klasse enthält die Wahrheit; Grad-1-Gitter, multiplikatives Rauschen) ohne Rauschen haben KQ, SPO+ und Kern-ETO im Median 0,00 % Regret; mit Rauschen 0,5 liegt KQ in n = 100, 300 und 1000 nicht über den beiden anderen, SPO+ darüber (der Abstand zu Kern-ETO ist klein, praktisch gleich).
- **Passt sie nicht, hilft im Gitter ohne Rauschen und bei genug Daten eine andere Klasse mehr als ein anderer Verlust.** Bei Grad 6 ohne Rauschen und n = 1000 liegt der Median-Regret von Kern-ETO bei 0,05 %, von SPO+ bei 0,51 %, von KQ bei 1,96 %; Kern-ETO ist in 20 von 20 Seeds besser als SPO+, obwohl der Löser in seinem Training nicht vorkommt. Im Newsvendor mit quadratischer Wahrheit liegen die Mehrkosten bei den Gewichten (28,7 % bei Kerngewichten) weit unter denen der linearen Regeln (140,7 % bei KQ + empirisch, 141,1 % bei direktem Lernen in derselben Klasse).
- **Wenig Daten und Rauschen: kein klarer Gewinner.** Bei Grad 6, Rauschstufe 0,5 und n = 100 liegt SPO+ im Median über KQ, Kern-ETO knapp darunter (siehe „Befunde“).

## Modell

**Welt im Gitter** (aus `spo-kuerzester-weg-demo` kopiert, dort aus dem Gedächtnis nach dem Schema der SPO-Literatur nachgebaut, nicht gegen sie geprüft): Merkmale $x \sim \mathcal N(0, I_5)$; eine Gewichtsmatrix $B \in \{0,1\}^{40 \times 5}$ mit fairen Münzwürfen; mittlere Kosten $m_j(x) = ((Bx)_j/\sqrt{5} + 3)^{\deg} / 3{,}5^{\deg} + 1$; beobachtete Kosten $c_j = m_j(x)\,\varepsilon_j$ mit $\varepsilon_j \sim U[1 - e, 1 + e]$. Grad 1 ist linear in $x$ (die lineare Klasse enthält die Wahrheit), Grad 2, 4 und 6 weichen zunehmend davon ab. $e$ ist die Rauschstufe (0 oder 0,5). Das **Orakel** kennt $m(x)$; bei $e = 0$ hat es Regret 0.

**Löser und Regret.** Rückwärtsrechnung über die 25 Knoten (das Gitter ist azyklisch), bei Gleichstand rechts vor unten. Der Regret eines Falles ist $c^\top w(\hat c) - z^*(c)$ (wahre Kosten des gewählten Weges minus die besten wahren Kosten); der **normierte Regret** einer Testmenge ist die Summe der Regrets geteilt durch die Summe der besten Wegkosten, in Prozent. Mit Rauschen hat auch das Orakel Regret, weil der beste Weg auf den *beobachteten* Kosten bestimmt wird (der unvermeidbare Teil).

**Die drei Sorten im Gitter.**

- **KQ** (zweistufig, linear): lineares Modell mit Achsenabschnitt (6 × 40 = 240 Parameter), Kleinste Quadrate auf dem Fehler der Kosten, dann der Löser.
- **SPO+** (auf der Entscheidung, derselbe lineare Parameterraum): der konvexe Ersatz $\max_{w}(c - 2\hat c)^\top w + 2\hat c^\top w^*(c) - z^*(c)$ des Regrets, Adam, Lernrate 0,05, Vollbatch, 300 Schritte, Start bei KQ; der Löser läuft in jeder Runde (Mechanismus in `spo-kuerzester-weg-demo`).
- **Kern-ETO** (zweistufig, flexible Klasse): Gauß-Kern-Ridge, je Bogen $\hat c(x) = \mu + k(x, X)\,(K + aI)^{-1}(C - \mu)$ mit $K_{ij} = \exp(-\gamma\lVert x_i - x_j\rVert^2)$, gelöst mit `numpy.linalg.solve`; $\gamma \in \{0{,}02;\ 0{,}05;\ 0{,}1;\ 0{,}3\}$ und die Strafe $a \in \{0{,}001;\ 0{,}01;\ 0{,}1;\ 1\}$ werden per Validierung gewählt (fester Zufallsschnitt 80 % / 20 %, kleinster mittlerer quadratischer Fehler der Kosten). Der Löser kommt im Training nicht vor.

**Newsvendor mit Merkmalen** (`zoi_news.py`, aus `newsvendor-direkt-demo` und `gewichtete-stichprobe-demo` kopiert und gekürzt): Nachfrage $D = \max(0, Y)$, $Y \sim \mathcal N(\mu(x), 8^2)$, drei Merkmale, $\mu(x) = 60 + 10\,(\beta^\top x)$ in der **linearen** Welt und $\mu(x) = 60 + 10\,(\beta^\top x) + 8\,(x_1^2 - 1)$ in der **quadratischen** Welt ($\beta$ ein zufälliger Einheitsvektor). Bestellt wird das $\tau$-Quantil (kritisches Verhältnis $\tau = b/(b+h)$, $h = 1$). Regeln: **KQ + empirisch** (Kleinste Quadrate plus das empirische $\tau$-Quantil der Residuen), **direkt** (lineare Quantilregression, ein lineares Programm, `scipy.optimize.linprog` mit HiGHS; ihr Verlust ist bis auf den Faktor $b + h$ der Kostenverlust), **kNN-** und **Kerngewichte** (gewichtetes $\tau$-Quantil der Nachfragen, $k$ beziehungsweise Bandbreite per Leave-one-out) und **Wald-Gewichte** (nur als Vormessung). Die Kosten einer Regel werden exakt über die Nachfrage gerechnet; Mehrkosten = (Kosten der Regel − Kosten des Orakels) / Kosten des Orakels.

**Die Lücke Δ.** Mit den sortierten Wegwerten $v_1 \le v_2 \le \dots$ unter den mittleren Kosten $m(x)$ ist $\Delta(x) = v_2 - v_1$ die Lücke zwischen bestem und zweitbestem Weg; die Demo benutzt die relative Lücke $\Delta(x)/v_1$ (Anteil des besten Wertes). Gleichstand im besten Wert gibt $\Delta = 0$.

**Was die Theorie sagt** (Volltext gelesen; Nummern nach der genannten Fassung, Beweise und Anhänge nicht gelesen; beides **asymptotische** Aussagen, die hier eingeordnet, nicht gemessen werden):

- Elmachtoub, Lam, Zhang, Zhao (arXiv 2304.06833, Fassung v4 vom 22.05.2025): Das Modell ist eine parametrische Verteilungsfamilie; ETO schätzt sie per Maximum Likelihood und optimiert, IEO wählt den Parameter über den empirischen Entscheidungsverlust, SAA benutzt kein Modell. **Satz 2:** bei Wohlspezifikation und den Annahmen 1 bis 3 (Konsistenz, Regularität, Glattheit) konvergiert $n\cdot$Regret in Verteilung, mit stochastischer Dominanz erster Ordnung ETO $\le$ IEO $\le$ SAA. **Satz 3:** bei Fehlspezifikation kehrt sich die Ordnung um. IEO dort ist nicht das hier trainierte SPO+, und die Autoren nennen die Lücke zwischen Asymptotik und endlichen, leicht falsch spezifizierten Fällen selbst.
- Hu, Kallus, Mao (*Fast Rates for Contextual Linear Optimization*, Management Science 68(6), 2022, arXiv 2011.03030, Abschnitte nach arXiv v3; Satznummern hier weggelassen): lineares Ziel über einem Polytop, $f^*(x) = E[Y \mid X = x]$. Annahme: $f^*$ liegt in der Modellklasse. Ohne weitere Annahme erreichen die auf die Entscheidung trainierten Regeln $n^{-1/2}$ (Abschnitt 2 der Fassung v3). Rauschbedingung: $P(0 < \Delta(X) \le \delta) \le (\gamma\delta/B)^{\alpha}$ mit der Lücke Δ; mit ihr (und weiteren technischen Annahmen) erreicht die auf die Entscheidung trainierte Regel $n^{-(1+\alpha)/(2+\alpha)}$ (Abschnitt 3), die zweistufige Plug-in-Regel unter Annahmen an den Schätzer (unter anderem eine Recovery-Annahme und eine sternförmige Klasse) bis $n^{-(1+\alpha)/2}$ (Abschnitt 3), bei $\alpha = 1$ also $n^{-1}$ gegen $n^{-2/3}$. Ihr Gitterversuch (Abschnitt 5: 5 × 5, Polynom, ETO gegen SPO+ mit drei Klassen: korrekt linear, falsch linear, Gauß-Kern) zeigt qualitativ dieselbe Anordnung wie Experiment 1; die Theorie gilt laut den Autoren nicht direkt für SPO+. Dieselbe Quelle nennt SPO+ unter Regularitätsbedingungen Fisher-konsistent, wenn $f^*$ in der Klasse liegt.

## Methodik

- **Auswertung.** Jede Zelle (Grad, n, Rauschstufe) läuft mit den festen Zufalls-Seeds 0 bis 19 und 1000 Testfällen; angegeben werden **Median und Spanne (Minimum bis Maximum)** über die Seeds, nie ein einzelner Seed. Die Bänder stehen vorgerechnet in `zoi_bands.json` und sind mit `python tools/measure_bands.py` reproduzierbar (einige Minuten mit zwei Prozessen; Zufall nur über den ganzzahligen SplitMix64 aus `zoi_rng.py`, nie Pythons `hash()`).
- **Datenmengen.** n = 100, 300, 1000 mit allen drei Sorten; n = 5000 nur mit KQ und SPO+ (die Kernmatrix von 5000 × 5000 Zahlen ist für eine Demo zu groß, Kern-ETO fehlt dort und die App sagt es). Die Live-Rechnung der App geht bis n = 1000.
- **Zufallswald nicht live.** Zufallswald-Zahlen stammen aus Vormessungen mit scikit-learn (Gitter: 100 Bäume, mindestens 5 Fälle je Blatt, nicht abgestimmt, 20 Seeds; Newsvendor: 50 Bäume, 30 Seeds, 5000 Testfälle als Monte-Carlo-Stichprobe), eingefroren in `zoi_vormessung.json` (`tools/freeze_vormessung.py`); scikit-learn gehört nicht zu den Laufzeit-Paketen. In der App und hier heißt das Verfahren „Zufallswald (Vormessung, nicht live)“. Die Vormessungen haben andere Zufallsquellen (ihre SPO+-Spalte benutzt 600 statt 300 Adam-Schritte): ihre Zahlen sind vergleichbar, nicht gleich.
- **Vormessung V4 vor dem Bau von Experiment 3.** Vorab festgelegtes Kriterium (`tools/vormessung_v4.py`, Rohausgabe `tools/out_vormessung_v4.txt`): Experiment 3 trägt, wenn (1) der Anteil der Fälle mit relativer Lücke bis 5 % bei Grad 6 zwischen 5 % und 95 % liegt und mindestens 2 Prozentpunkte vom Anteil bei Grad 1 entfernt ist und (2) in jeder Zelle mit Grad 4 und 6 sowie Rauschstufe 0 und 0,5 und für jedes der drei Verfahren der Anteil gewählter Nicht-Bestwege im untersten Drittel der Lücke in mindestens 90 % der Seeds größer ist als im obersten. Beides erfüllt (Teil A und B unten); Experiment 3 ist deshalb gebaut.
- **Trainierte Zahlen sind plattformabhängig** (Adam, Gleitkomma, Löserentscheidungen bei knappen Unterschieden, Validierung). Die Tests prüfen sie deshalb nur als Bänder und Ungleichungen; exakt geprüft werden nur Rechnungen ohne Training.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen dieses Abschnitts stammen aus `zoi_bands.json` (eigene Messung, Zufalls-Seeds 0 bis 19, 1000 Testfälle, normierter Regret in Prozent als Median [Minimum; Maximum]), wenn nichts anderes dabeisteht. Die Tabellen werden von `tests/readme_tables.py` aus den JSON-Dateien erzeugt, und `tests/test_claims.py` prüft, dass jede Zeile wörtlich hier steht; die Aussagen darunter stehen als Bänder und Ungleichungen im selben Test. Die Spalten „besser als“ zählen die Seeds (von 20), in denen der Regret des ersten Verfahrens streng unter dem des zweiten liegt.

### Experiment 1 und 2: Gitter ohne Rauschen (Rauschstufe 0)

| Grad | n | Orakel | KQ | SPO+ | Kern-ETO | SPO+ besser als KQ | Kern-ETO besser als KQ | Kern-ETO besser als SPO+ |
|---|---|---|---|---|---|---|---|---|
| 1 | 100 | 0,00 | 0,00 [0,00; 0,00] | 0,00 [0,00; 0,12] | 0,00 [0,00; 0,00] | 0 von 20 | 0 von 20 | 1 von 20 |
| 1 | 300 | 0,00 | 0,00 [0,00; 0,00] | 0,00 [0,00; 0,02] | 0,00 [0,00; 0,00] | 0 von 20 | 0 von 20 | 3 von 20 |
| 1 | 1000 | 0,00 | 0,00 [0,00; 0,00] | 0,00 [0,00; 0,00] | 0,00 [0,00; 0,00] | 0 von 20 | 0 von 20 | 0 von 20 |
| 1 | 5000 | 0,00 | 0,00 [0,00; 0,00] | 0,00 [0,00; 0,00] | nicht gerechnet | 0 von 20 | – | – |
| 2 | 100 | 0,00 | 0,07 [0,04; 0,10] | 0,23 [0,16; 0,32] | 0,00 [0,00; 0,00] | 0 von 20 | 20 von 20 | 20 von 20 |
| 2 | 300 | 0,00 | 0,07 [0,04; 0,10] | 0,07 [0,04; 0,08] | 0,00 [0,00; 0,00] | 9 von 20 | 20 von 20 | 20 von 20 |
| 2 | 1000 | 0,00 | 0,06 [0,04; 0,08] | 0,03 [0,02; 0,05] | 0,00 [0,00; 0,00] | 20 von 20 | 20 von 20 | 20 von 20 |
| 2 | 5000 | 0,00 | 0,06 [0,04; 0,09] | 0,02 [0,01; 0,03] | nicht gerechnet | 20 von 20 | – | – |
| 4 | 100 | 0,00 | 0,76 [0,48; 1,5] | 0,57 [0,42; 0,75] | 0,09 [0,04; 0,18] | 18 von 20 | 20 von 20 | 20 von 20 |
| 4 | 300 | 0,00 | 0,66 [0,48; 1,1] | 0,31 [0,21; 0,38] | 0,01 [0,01; 0,02] | 20 von 20 | 20 von 20 | 20 von 20 |
| 4 | 1000 | 0,00 | 0,68 [0,41; 0,91] | 0,19 [0,13; 0,27] | 0,00 [0,00; 0,01] | 20 von 20 | 20 von 20 | 20 von 20 |
| 4 | 5000 | 0,00 | 0,69 [0,39; 0,89] | 0,17 [0,08; 0,24] | nicht gerechnet | 20 von 20 | – | – |
| 6 | 100 | 0,00 | 2,6 [1,5; 5,1] | 1,2 [0,84; 2,3] | 0,60 [0,24; 1,2] | 20 von 20 | 20 von 20 | 18 von 20 |
| 6 | 300 | 0,00 | 2,1 [1,4; 3,5] | 0,74 [0,46; 0,91] | 0,13 [0,06; 0,68] | 20 von 20 | 20 von 20 | 20 von 20 |
| 6 | 1000 | 0,00 | 2,0 [1,4; 3,0] | 0,51 [0,36; 0,83] | 0,05 [0,02; 0,19] | 20 von 20 | 20 von 20 | 20 von 20 |
| 6 | 5000 | 0,00 | 2,0 [1,3; 3,1] | 0,50 [0,25; 0,74] | nicht gerechnet | 20 von 20 | – | – |

### Experiment 1 und 2: Gitter mit Rauschen (Rauschstufe 0,5)

| Grad | n | Orakel | KQ | SPO+ | Kern-ETO | SPO+ besser als KQ | Kern-ETO besser als KQ | Kern-ETO besser als SPO+ |
|---|---|---|---|---|---|---|---|---|
| 1 | 100 | 17,8 | 19,6 [18,6; 20,9] | 20,5 [18,9; 21,3] | 19,6 [18,8; 21,0] | 1 von 20 | 4 von 20 | 19 von 20 |
| 1 | 300 | 17,8 | 18,3 [17,7; 19,4] | 18,9 [18,2; 20,4] | 18,5 [17,8; 19,5] | 0 von 20 | 4 von 20 | 19 von 20 |
| 1 | 1000 | 17,8 | 18,1 [17,3; 18,9] | 18,3 [17,5; 19,3] | 18,2 [17,4; 19,2] | 6 von 20 | 4 von 20 | 11 von 20 |
| 1 | 5000 | 17,8 | 17,8 [17,2; 18,8] | 17,8 [17,1; 18,9] | nicht gerechnet | 13 von 20 | – | – |
| 2 | 100 | 15,0 | 16,3 [14,9; 17,6] | 17,7 [15,9; 19,1] | 16,6 [15,0; 18,1] | 0 von 20 | 6 von 20 | 18 von 20 |
| 2 | 300 | 15,0 | 15,3 [14,5; 16,4] | 16,0 [14,6; 17,0] | 15,4 [14,6; 16,3] | 0 von 20 | 8 von 20 | 19 von 20 |
| 2 | 1000 | 15,0 | 15,4 [14,2; 16,2] | 15,4 [14,0; 16,4] | 15,3 [14,0; 16,1] | 6 von 20 | 10 von 20 | 16 von 20 |
| 2 | 5000 | 15,0 | 15,1 [14,0; 16,0] | 15,1 [14,2; 16,0] | nicht gerechnet | 9 von 20 | – | – |
| 4 | 100 | 11,9 | 13,8 [11,7; 15,2] | 14,6 [13,1; 16,4] | 13,8 [12,1; 16,5] | 0 von 20 | 9 von 20 | 17 von 20 |
| 4 | 300 | 11,9 | 13,0 [11,5; 14,5] | 13,2 [11,9; 14,4] | 12,7 [11,7; 14,1] | 3 von 20 | 15 von 20 | 17 von 20 |
| 4 | 1000 | 11,9 | 12,8 [11,6; 14,1] | 12,7 [11,2; 13,8] | 12,4 [11,1; 13,9] | 13 von 20 | 18 von 20 | 17 von 20 |
| 4 | 5000 | 11,9 | 12,8 [11,3; 14,0] | 12,5 [11,3; 13,7] | nicht gerechnet | 16 von 20 | – | – |
| 6 | 100 | 10,5 | 14,0 [11,5; 16,7] | 14,4 [12,5; 16,7] | 13,8 [11,3; 16,7] | 10 von 20 | 16 von 20 | 16 von 20 |
| 6 | 300 | 10,5 | 13,0 [11,6; 14,8] | 12,6 [11,3; 14,7] | 12,3 [10,6; 14,0] | 14 von 20 | 16 von 20 | 13 von 20 |
| 6 | 1000 | 10,5 | 13,1 [11,3; 14,5] | 12,3 [10,7; 13,4] | 11,8 [10,1; 13,6] | 20 von 20 | 20 von 20 | 15 von 20 |
| 6 | 5000 | 10,5 | 13,0 [10,9; 14,6] | 11,9 [10,5; 13,5] | nicht gerechnet | 20 von 20 | – | – |

### Was die Tabellen sagen (Zählungen über die Zellen)

- **Klasse passt (Grad 1):** Ohne Rauschen haben alle drei Sorten in allen drei Zellen mit Kern-ETO (n = 100 / 300 / 1000) und bei n = 5000 für KQ und SPO+ einen Median-Regret von 0,00 %, KQ ist in keinem der 20 Seeds von SPO+ oder Kern-ETO geschlagen. Mit Rauschen 0,5 ist der Median von KQ in n = 100, 300 und 1000 nicht größer als der von Kern-ETO und kleiner als der von SPO+ (auf eine Stelle gerundet: 19,6 gegen 19,6 und 20,5; 18,3 gegen 18,5 und 18,9; 18,1 gegen 18,2 und 18,3); bei n = 5000 sind KQ und SPO+ gleich (17,8 %).
- **Klasse passt nicht, ohne Rauschen (Grad 4 und 6):** In allen 6 Zellen (Grad 4 und 6, n = 100 / 300 / 1000) liegt der Median von Kern-ETO unter dem von SPO+ und der von SPO+ unter dem von KQ. Bei Grad 6 und n = 1000: Kern-ETO 0,05 %, SPO+ 0,51 %, KQ 2,0 % (Kern-ETO besser als SPO+ in 20 von 20 Seeds, besser als KQ in 20 von 20).
- **Klasse passt nicht, mit Rauschen (Grad 6):** n = 1000: Kern-ETO 11,8 %, SPO+ 12,3 %, KQ 13,1 %, Orakel 10,5 %; SPO+ schließt 31 % der Lücke zwischen KQ und Orakel, Kern-ETO 51 % (aus den ungerundeten Medianwerten berechnet). Der Rest ist unvermeidbares Rauschen oder bleibt im jeweiligen Verfahren.
- **Datenmenge (Grad 6):** SPO+ liegt in 7 von 8 Zellen (n = 100 / 300 / 1000 / 5000, Rauschstufe 0 und 0,5) im Median unter KQ; die Ausnahme ist n = 100 mit Rauschen (14,4 % gegen 14,0 %). Kern-ETO liegt in allen 6 Zellen mit Kern-ETO (n ≤ 1000) unter SPO+.

### Wenig Daten und Rauschen: Stabilitätsprobe (Grad 6, n = 100, Rauschstufe 0,5, 100 weitere Zufalls-Seeds 20 bis 119)

| Verfahren | Regret (Median über 100 Seeds) | besser als KQ | Median der Differenz zu KQ |
|---|---|---|---|
| Orakel | 10,4 % | – | – |
| KQ | 13,7 % | – | – |
| SPO+ | 14,2 % | 29 von 100 | 0,53 Punkte |
| Kern-ETO | 13,5 % | 64 von 100 | -0,45 Punkte |

In den 20 Seeds der Bänder ist SPO+ in 10 und Kern-ETO in 16 von 20 Seeds besser als KQ; mit 100 weiteren Seeds liegt SPO+ in 29 und Kern-ETO in 64 von 100 Seeds unter KQ, im Median der Paare 0,53 Punkte darüber beziehungsweise 0,45 Punkte darunter. Der Vorsprung von Kern-ETO ist klein (unter einem halben Punkt im Median der Paare, in 36 von 100 Seeds kein Vorsprung), der Rückstand von SPO+ ebenso; ein klarer Gewinner ist das nicht. Die „10 von 20“ der Bänder wäre als Zahl allein nicht tragfähig: erst die Wiederholung mit 100 anderen Seeds zeigt die Richtung (29 von 100 für SPO+).

### Experiment 3: die Lücke zwischen bestem und zweitbestem Weg

Vormessung V4 (`tools/vormessung_v4.py`, 20 Welten je Grad mit je 2000 Testpunkten; die Lücke hängt nur von den mittleren Kosten ab, nicht vom Rauschen). Relative Lücke Δ / z\* über die Testpunkte, Median der Anteile über die 20 Welten:

| Grad | Lücke ≤ 1 % | Lücke ≤ 2 % | Lücke ≤ 5 % | Lücke ≤ 10 % | Median der Lücke |
|---|---|---|---|---|---|
| 1 | 85 % | 98 % | 100 % | 100 % | 0,36 % |
| 2 | 67 % | 89 % | 100 % | 100 % | 0,62 % |
| 4 | 54 % | 76 % | 97 % | 100 % | 0,89 % |
| 6 | 51 % | 71 % | 92 % | 98 % | 0,98 % |

Die Lücke ist in dieser Welt in 51 % (Grad 6) bis 85 % (Grad 1) der Fälle kleiner als 1 % des besten Wertes (Median über die 20 Welten). Gleichstand im besten Wert (Lücke 0) kommt nur bei Grad 1 vor: in 5 von 20 Welten bei 0,45 % bis 40,3 % der Testpunkte, bei Grad 2, 4 und 6 in keiner Welt; die Theorie schließt Δ = 0 aus ihrer Bedingung aus. Das ist eine Messung in der relativen Skala dieser Demo; die Theorie rechnet mit der absoluten Lücke und einer Konstanten $B$, und $\alpha$ wird hier nicht geschätzt.

Je Drittel der Testfälle, geordnet nach relativer Lücke (n = 1000, Median über 20 Seeds): **Fehlgriff** = der gewählte Weg ist nicht der beste Weg unter den mittleren Kosten; **Regret je Fall** = mittlere Kosten des gewählten Weges minus bester Wert, in Prozent des besten Wertes; **Anteil** = Anteil des Drittels am gesamten Regret in Erwartung (die drei Mediane ergeben nicht genau 100 %).

| Grad | Rauschstufe | Verfahren | Fehlgriff (Lücke klein / mittel / groß) | Regret je Fall | Anteil am gesamten Regret |
|---|---|---|---|---|---|
| 4 | 0,0 | KQ | 58 % / 42 % / 20 % | 0,54 % / 0,68 % / 0,69 % | 29 % / 36 % / 35 % |
| 4 | 0,0 | SPO+ | 42 % / 17 % / 4 % | 0,20 % / 0,22 % / 0,13 % | 37 % / 38 % / 24 % |
| 4 | 0,0 | Kern-ETO | 9 % / 0 % / 0 % | 0,01 % / 0,00 % / 0,00 % | 86 % / 14 % / 0 % |
| 4 | 0,5 | KQ | 61 % / 44 % / 23 % | 0,66 % / 0,79 % / 0,83 % | 30 % / 36 % / 35 % |
| 4 | 0,5 | SPO+ | 60 % / 43 % / 21 % | 0,57 % / 0,68 % / 0,70 % | 27 % / 36 % / 36 % |
| 4 | 0,5 | Kern-ETO | 56 % / 34 % / 11 % | 0,49 % / 0,50 % / 0,34 % | 35 % / 40 % / 27 % |
| 6 | 0,0 | KQ | 66 % / 55 % / 36 % | 1,3 % / 1,7 % / 2,4 % | 24 % / 32 % / 44 % |
| 6 | 0,0 | SPO+ | 50 % / 28 % / 12 % | 0,39 % / 0,45 % / 0,50 % | 26 % / 35 % / 37 % |
| 6 | 0,0 | Kern-ETO | 32 % / 4 % / 0 % | 0,13 % / 0,04 % / 0,01 % | 73 % / 22 % / 4 % |
| 6 | 0,5 | KQ | 67 % / 56 % / 38 % | 1,4 % / 1,8 % / 2,6 % | 23 % / 32 % / 44 % |
| 6 | 0,5 | SPO+ | 64 % / 52 % / 28 % | 0,93 % / 1,3 % / 1,7 % | 23 % / 32 % / 42 % |
| 6 | 0,5 | Kern-ETO | 75 % / 55 % / 22 % | 1,4 % / 1,3 % / 1,0 % | 38 % / 36 % / 27 % |

In allen 12 Kombinationen aus Grad (4, 6), Rauschstufe (0, 0,5) und Verfahren (KQ, SPO+, Kern-ETO) ist der Anteil gewählter Nicht-Bestwege im untersten Drittel der Lücke in 20 von 20 Seeds größer als im obersten. Ohne Rauschen und bei Grad 6 fallen 73 % des Regrets von Kern-ETO ins unterste Drittel der Lücke, bei KQ 24 %; im obersten Drittel wählt KQ in 36 % der Fälle einen anderen als den besten Weg, Kern-ETO in 0 % (gerundet). Das ist eine Messung, keine Erklärung: warum KQ auch bei großer Lücke danebenliegt, ist nicht untersucht (naheliegend ist die Fehlspezifikation der linearen Klasse, aber nicht durch einen eigenen Versuch belegt).

### Experiment 4: Newsvendor in der falschen Klasse

Eigene Messung (20 Zufalls-Seeds 0 bis 19, drei Merkmale, 2000 feste Testmerkmale, Mehrkosten exakt gerechnet), Mehrkosten gegen das Orakel in Prozent, Median [Minimum; Maximum]; die letzte Spalte ist die Vormessung mit scikit-learn (30 Seeds, 5000 Testfälle als Stichprobe, nicht live).

**Quadratische Welt (die lineare Klasse ist falsch):**

| n | τ | KQ + empirisch | direkt (Quantilregression) | kNN-Gewichte | Kerngewichte | Wald-Gewichte (Vormessung) |
|---|---|---|---|---|---|---|
| 100 | 0,50 | 60,3 [52,1; 69,4] | 59,2 [50,3; 65,8] | 44,4 [33,5; 60,2] | 37,1 [29,3; 47,5] | 43,2 [31,1; 53,6] |
| 100 | 0,80 | 91,9 [80,7; 112,4] | 94,9 [82,2; 109,9] | 65,6 [41,2; 101,5] | 48,6 [35,8; 73,3] | 52,2 [41,0; 70,3] |
| 100 | 0,95 | 154,8 [128,4; 175,9] | 161,9 [127,4; 240,6] | 87,7 [48,4; 161,2] | 86,2 [52,5; 138,9] | 89,4 [69,8; 133,4] |
| 1000 | 0,50 | 56,5 [49,8; 62,1] | 56,5 [49,9; 61,9] | 22,7 [17,3; 26,0] | 14,8 [11,9; 17,1] | 17,4 [12,4; 23,7] |
| 1000 | 0,80 | 88,0 [78,0; 98,6] | 87,9 [78,5; 98,9] | 24,0 [18,0; 29,7] | 16,9 [12,1; 21,3] | 18,4 [14,3; 27,1] |
| 1000 | 0,95 | 140,7 [118,3; 163,4] | 141,1 [120,9; 172,9] | 30,1 [16,7; 40,6] | 28,7 [17,7; 38,1] | 29,7 [19,7; 44,9] |

**Lineare Welt (die lineare Klasse ist richtig):**

| n | τ | KQ + empirisch | direkt (Quantilregression) | kNN-Gewichte | Kerngewichte | Wald-Gewichte (Vormessung) |
|---|---|---|---|---|---|---|
| 100 | 0,50 | 1,4 [0,33; 5,2] | 2,1 [0,21; 8,9] | 20,0 [14,2; 27,8] | 17,0 [11,4; 28,4] | 20,2 [12,6; 28,2] |
| 100 | 0,80 | 2,0 [0,44; 4,2] | 3,1 [0,60; 13,8] | 20,4 [8,3; 33,8] | 17,6 [7,9; 29,9] | 22,4 [17,2; 27,5] |
| 100 | 0,95 | 3,0 [0,34; 8,7] | 7,0 [0,26; 42,3] | 26,5 [17,6; 61,0] | 25,8 [18,1; 49,4] | 27,8 [19,9; 40,2] |
| 1000 | 0,50 | 0,10 [0,04; 0,64] | 0,13 [0,03; 0,65] | 6,2 [4,5; 8,1] | 5,0 [3,6; 6,2] | 8,7 [6,9; 10,3] |
| 1000 | 0,80 | 0,10 [0,04; 0,50] | 0,26 [0,06; 0,56] | 7,1 [5,1; 8,3] | 5,9 [4,7; 7,7] | 9,5 [7,7; 11,7] |
| 1000 | 0,95 | 0,30 [0,07; 1,1] | 0,72 [0,14; 2,9] | 9,3 [6,5; 11,9] | 8,8 [6,8; 11,5] | 12,4 [9,2; 16,9] |

- **Falsche Klasse:** In der quadratischen Welt liegen kNN- und Kerngewichte in allen 6 Zellen (n = 100 / 1000, τ = 0,5 / 0,8 / 0,95) im Median unter beiden linearen Regeln; bei n = 1000 und τ = 0,95 sind es 30,1 % und 28,7 % gegen 140,7 % (KQ + empirisch) und 141,1 % (direkt). Direktes Lernen in der linearen Klasse ist dort in keiner Zelle um mehr als 7,1 Punkte vom Median von KQ + empirisch entfernt (größter Abstand n = 100, τ = 0,95 zu Lasten von direkt) und liegt in 3 von 6 Zellen im Median knapp darunter (um höchstens 1,1 Punkte): ein anderer Verlust in derselben Klasse rettet nichts.
- **Richtige Klasse:** In der linearen Welt liegt KQ + empirisch in allen 6 Zellen im Median unter kNN- und Kerngewichten (bei n = 1000 zwischen 0,10 und 0,30 % gegen 5,0 bis 9,3 %), und direktes Lernen liegt in allen 6 Zellen über KQ + empirisch (bei n = 1000: 0,13 / 0,26 / 0,72 % gegen 0,10 / 0,10 / 0,30 % bei τ = 0,5 / 0,8 / 0,95). Die Gewichte zahlen hier den Preis ohne Modellannahme.
- **Vormessung der Quelle des Plans** (V3, andere Seeds und Monte-Carlo-Testfälle), die Zellen aus dem Plan (p = 3, n = 1000, τ = 0,95):

| Welt | KQ + empirisch | direkt | kNN-Gewichte | Wald-Gewichte |
|---|---|---|---|---|
| linear | 0,3 % | 0,6 % | 9,3 % | 12,4 % |
| quadratisch | 134,9 % | 136,5 % | 30,4 % | 29,7 % |

### Vormessungen mit scikit-learn (nicht live): Zufallswald und Kern-Ridge im Gitter

Zufallswald (Grad 6; Quelle `out_eo_v1_sp.txt`; SPO+ dort mit 600 Schritten):

| n | Rauschstufe | Zufallswald | KQ | SPO+ (600 Schritte) |
|---|---|---|---|---|
| 100 | 0,0 | 5,4 % | 2,4 % | 1,3 % |
| 1000 | 0,0 | 1,7 % | 1,9 % | 0,6 % |
| 5000 | 0,0 | 0,8 % | 1,8 % | 0,5 % |
| 100 | 0,5 | 17,2 % | 13,4 % | 14,5 % |
| 1000 | 0,5 | 12,6 % | 12,5 % | 11,8 % |
| 5000 | 0,5 | 11,5 % | 12,2 % | 11,6 % |

Der Zufallswald ist unabgestimmt (100 Bäume, keine Wahl der Blattgröße); die Zahlen sind kein Urteil über Zufallswälder. Bei Grad 6, Rauschstufe 0,5 und n = 5000 sind Wald (11,5 %) und SPO+ der Vormessung (11,6 %) gleichauf, KQ liegt bei 12,2 %.

Kern-Ridge der Vormessung V1d (scikit-learn `KernelRidge`, `out_eo_v1d_kernel.txt`) neben dem Kern-ETO dieser Demo (numpy), Grad 6:

| Rauschstufe | n | Kern-ETO Vormessung (scikit-learn) | Kern-ETO dieser Demo (numpy) | SPO+ Vormessung (300 Schritte) | SPO+ dieser Demo |
|---|---|---|---|---|---|
| 0,0 | 100 | 0,49 % | 0,60 % | 1,3 % | 1,2 % |
| 0,0 | 300 | 0,12 % | 0,13 % | 0,72 % | 0,74 % |
| 0,0 | 1000 | 0,04 % | 0,05 % | 0,55 % | 0,51 % |
| 0,5 | 100 | 13,5 % | 13,8 % | 14,3 % | 14,4 % |
| 0,5 | 300 | 12,5 % | 12,3 % | 12,4 % | 12,6 % |
| 0,5 | 1000 | 11,3 % | 11,8 % | 11,8 % | 12,3 % |

Die Abweichung zwischen beiden Kern-ETO-Spalten beträgt bis zu 0,45 Punkte (n = 1000, Rauschstufe 0,5); die Vormessung hat eine andere Zufallsquelle und einen anderen Validierungsschnitt.

### Prüfungen

- **Scratch-Läufe dieses Repos** (nicht im Repo, Messung dieser README, nicht im Test): Löser gegen Aufzählung der 70 Wege mit 300 Instanzen stetiger Kosten (größte Abweichung im Wert 7,1·10⁻¹⁵, dieselben Wege) und 300 Instanzen ganzzahliger Kosten (gleiche Werte und Wege, auch bei Gleichstand); Lücke Δ gegen die Aufzählung mit 300 Instanzen (Abweichung 0); Regret in Erwartung gegen die Aufzählung mit 300 Instanzen (3,6·10⁻¹⁵); Kern-Ridge gegen die Eigenzerlegung mit 200 zufälligen Größen und Parametern des Rasters (6,4·10⁻¹²); erwartete Newsvendor-Kosten gegen numerische Integration mit 100 Zellen (relativ 4,7·10⁻¹⁴); Quantilregression gegen die Aufzählung der Ecken mit 100 Instanzen (Zielwert 1,2·10⁻¹⁴). Der SPO+-Teilgradient und SPO+ gegen das Brute-Force-Maximum sind in `spo-kuerzester-weg-demo` mit 2000 Instanzen gemessen (dort dokumentiert) und hier nur in der kleinen Fassung erneut getestet.
- **Im Repo** laufen die kleinen Fassungen derselben Vergleiche (`tests/test_oracle_zweistufig_oder_integriert.py`, `tests/test_oracle_kernel_luecke_news.py`) mit den Toleranzen 10⁻⁹ (Kern-Ridge gegen Eigenzerlegung), 10⁻⁸ (gegen scikit-learn), 10⁻⁷ relativ (Newsvendor-Kosten gegen Integration) und 10⁻⁷ (Quantilregression gegen die Ecken); die Parameterwahl des Kern-Ridge wird gegen die Aufzählung der 16 Paare geprüft.
- **Exakt lineare, rauschfreie Welt (Grad 1, Rauschstufe 0):** In 9 Läufen (n = 100 / 200 / 1000, Zufalls-Seeds 0 bis 2) liegt der Regret von KQ unter 10⁻¹², der von Kern-ETO unter 10⁻⁴ und der von SPO+ unter 10⁻³ (Bänder; SPO+ gibt den Parameter mit dem kleinsten Trainingsverlust zurück, und der muss nicht der von KQ sein).

## Befunde und Korrekturen gegenüber dem Plan

- **Messabweichung aus `spo-kuerzester-weg-demo`:** „Bei n = 100, Rauschen 0,5, Grad 6 verliert SPO+ klar (2 von 20 Seeds besser)“ trägt nicht. In den 20 Seeds dieser Demo (gleiche Welt und gleicher Code wie dort) ist SPO+ in 10 von 20 Seeds besser als KQ (Median 14,4 gegen 14,0 %), in 100 weiteren Seeds in 29 von 100 und im Median 0,53 Punkte zu Lasten von SPO+. Es gilt die neue Messung; „SPO+ verliert klar“ steht deshalb nirgends, nur „im Median etwas über KQ“.
- **„Bei n = 100 mit Rauschen ist keine der Sorten besser als KQ“ (Verdikt im Plan) trägt für Kern-ETO nicht ganz:** Kern-ETO liegt im Median knapp darunter (13,8 gegen 14,0 %; in 16 von 20 und 64 von 100 Seeds, im Median der Paare 0,45 Punkte). Die Vormessung V1d hatte 13,54 gegen 13,43 % (Kern-ETO nicht besser); die Rangfolge KQ–Kern-ETO bei kleinem n hängt an Seeds und Zufallsquelle. Das Verdikt der App meldet deshalb „kein klarer Gewinner“, wenn Kern-ETO in weniger als 90 % der Seeds unter KQ liegt (bei n = 100 mit Rauschen: 16 von 20).
- **„Kern-ETO liegt in 5 von 6 Zellen vor SPO+“ (Plan, aus V1d, Grad 6):** in der eigenen Messung mit numpy-Kern-Ridge liegt er in 6 von 6 Zellen (n = 100 / 300 / 1000, Rauschstufe 0 und 0,5) im Median unter SPO+. Die Zelle n = 300 mit Rauschen war in V1d gleichauf (12,50 gegen 12,44 %); hier 12,3 gegen 12,6 % bei 13 von 20 Seeds mit Kern-ETO besser. Es gilt die neue Messung, die Unterschiede sind klein (Zählung 13 von 20).
- **Exakt lineare, rauschfreie Welt:** Der Plan verlangt Regret 0 beziehungsweise unter 10⁻⁶ für KQ, SPO+ und Kern-ETO. Das gilt für KQ (unter 10⁻¹²). SPO+ und Kern-ETO liegen in einzelnen Läufen darüber (Vorversuch mit fünf Seeds und n = 100 / 200 / 1000, Scratch-Lauf, nicht im Test: Kern-ETO bis etwa 9·10⁻⁶, SPO+ bis etwa 2,4·10⁻⁴); im Test stehen deshalb Bänder (siehe „Prüfungen“).
- **V4 trägt (Z2):** Experiment 3 bleibt. Die Lücke ist in dieser Welt in 51 % (Grad 6) bis 85 % (Grad 1) der Fälle kleiner als 1 % des besten Wertes (Schwelle willkürlich gewählt; bei Grad 1 zählen Gleichstände mit), der Fehlgriff nimmt mit der Lücke ab; die schnellen Raten sind damit **nicht** bestätigt, nur die Verteilung der Lücke gemessen.
- **Kern-ETO bei n = 5000 nicht gerechnet** (Plan: Raster n = 100 / 300 / 1000 / 5000 für Experiment 2): bei n = 5000 gibt es nur KQ und SPO+ (und den Zufallswald der Vormessung).
- **Newsvendor-Zahlen weichen von der Vormessung ab** (Plan: direkt 136,5, KQ + empirisch 134,9, kNN 30,4, Wald 29,7 %): hier andere Seeds und exakte statt Stichproben-Kosten, gemessen 141,1 / 140,7 / 30,1 (Kerngewichte 28,7) in der quadratischen Welt; die Rangfolge ist dieselbe.

## Ehrliche Grenzen

- **Eine Familie von Welten:** nur 5 × 5, nur diese Kostenformel (Grad 1, 2, 4, 6), Rauschstufe 0 und 0,5, fünf Merkmale, ein lineares Modell für KQ und SPO+, SPO+ mit Adam, fester Lernrate und 300 Schritten; im Newsvendor zwei Welten, drei Merkmale, drei Werte von τ. Aussagen über andere Gitter, Wahrheiten, Trainer oder Lernraten sind nicht gemessen.
- **Das Gitter und die Kostenformel sind aus dem Gedächtnis nachgebaut**, nicht gegen den Primärtext geprüft. Elmachtoub und Grigas (Management Science 68(1), 2022) und Bertsimas und Kallus (Management Science 2020) habe ich nur auf der Ebene des Abstracts gelesen; Hu/Kallus/Mao und Elmachtoub/Lam/Zhang/Zhao im Volltext (Beweise und Anhänge nicht), Satznummern nach arXiv v3 beziehungsweise v4.
- **„Flexible Klasse“ heißt hier Gauß-Kern-Ridge** mit einem festen Raster; andere flexible Lerner (Wälder, Netze) kommen nur als unabgestimmter Zufallswald der Vormessung vor. Dass die flexible Klasse gewinnt, gilt für diesen Lerner in diesen Welten.
- **Wohlspezifikation im Sinne der Theorie ist nicht nachgebaut:** Grad 1 ist die lineare Wahrheit im Gitter, nicht die 31 Monome der Quelle; ETO ist hier ein Kostenmodell (Kleinste Quadrate), nicht die Maximum-Likelihood-Schätzung einer Verteilungsfamilie, SPO+ ist nicht IEO. Die Theorie wird eingeordnet, nicht geprüft; **die Raten werden nicht gemessen**, $\alpha$ nicht geschätzt.
- **Die Lücke Δ ist relativ gemessen** (Anteil des besten Wertes); die Rauschbedingung der Theorie verwendet die absolute Lücke und eine Konstante. Die Verteilung gilt für diese Welt, nicht für „realistische“ Probleme.
- **Zahlen sind trainiert, validiert und plattformabhängig.** Gemessen unter Windows; unter Linux (CI und Streamlit Cloud) können einzelne Werte um Zehntelpunkte abweichen. Die Tests prüfen Bänder und Ungleichungen.
- **Median über 20 Seeds, kein Signifikanztest.** Die Spanne (Minimum bis Maximum) ist keine Konfidenzaussage; Zellen mit „10 von 20“ sind nicht von gleichauf zu unterscheiden; bei kleinem n ist die Stabilitätsprobe mit 100 Seeds belastbarer.
- **Ursachen sind Deutungen:** dass SPO+ bei wenig Daten verliert, weil 240 Parameter auf 100 Fälle kommen, ist die naheliegende Lesart, aber nicht durch einen eigenen Versuch belegt; ebenso, warum die zweistufige Regel in der falschen linearen Klasse auch bei großer Lücke danebenliegt.
- **Keine Aussage über Rechenzeit.** Gemessen sind Regret und Mehrkosten, keine Sekunden; die App nennt deshalb keine feste Zeit.

## Weitergabe: Module

Der Kern liegt in Modulen ohne Streamlit-Abhängigkeit (numpy, für den Newsvendor scipy):

| Modul | Inhalt |
|---|---|
| `zoi_rng.py`, `zoi_grid.py`, `zoi_world.py`, `zoi_losses.py`, `zoi_training.py` | Gitter-Gerüst aus `spo-kuerzester-weg-demo` (`skw_*`), kopiert und im Präfix umbenannt, Herkunft im Modulkopf |
| `zoi_kernel.py` | Gauß-Kern-Ridge mit Parameterwahl per Validierung (Kern-ETO), neu geschrieben |
| `zoi_gap.py` | Lücke Δ zwischen bestem und zweitbestem Weg, Regret in Erwartung je Fall |
| `zoi_runs.py` | Lauf einer Welt (KQ, SPO+, Kern-ETO, Orakel), Läufe über Seeds, Zusammenfassung |
| `zoi_news.py` | Newsvendor mit Merkmalen: Welten, exakte Kosten, KQ + empirisch, Quantilregression, kNN- und Kerngewichte (aus `newsvendor-direkt-demo` und `gewichtete-stichprobe-demo` kopiert und gekürzt) |

## Verifikation

Die Suite hat **234 Tests**.

- **Kern gegen Handrechnung:** Löser auf 2 × 2, 3 × 3 und 5 × 5 mit von Hand gerechneten Instanzen, Handbeispiel (SPO 2, SPO+ 4, Teilgradient (2, −2)), Adam-Schritte von Hand (`tests/test_grid.py`, `tests/test_losses.py`, `tests/test_training.py`); Kern-Ridge mit zwei Fällen von Hand (Vorhersage 3 und 5), Grenzfall $\gamma \to 0$ gleich konstantes Mittel (`tests/test_kernel.py`); Lücke Δ mit zwei und drei Wegen (`tests/test_gap.py`); Newsvendor: Quantile, Gewichte, Pinball von Hand (`tests/test_news.py`).
- **Orakel** (`tests/test_oracle_zweistufig_oder_integriert.py`, `tests/test_oracle_kernel_luecke_news.py`, unabhängige Rechenwege): Löser, SPO+, SPO, Teilgradient und KQ gegen Aufzählung, Brute Force und Differenzenquotient; Kern-Ridge gegen Eigenzerlegung und scikit-learn; Parameterwahl gegen die Aufzählung; Lücke und Regret in Erwartung gegen die Aufzählung; Newsvendor-Kosten gegen numerische Integration, Quantilregression gegen die Ecken (p = 1) und scikit-learn; Leave-one-out gegen eine Schleife; SplitMix64 gegen eine reine Python-Ganzzahlrechnung (`tests/test_rng.py`).
- **Eigenschaften:** SPO+ ≥ SPO, Teilgradienten-Ungleichung, Orakel hat Regret 0 ohne Rauschen, mehr Rauschen erhöht den Rauschboden monoton, Kern-ETO mit unendlicher Bandbreite gleich konstantem Modell, relative Lücke skaleninvariant.
- **Randfälle:** n = 1 bis 7 (kleiner als die Parameterzahl), Gleichstände, Rauschstufe 0, kleinstes Gitter, Nullkosten, Regler an den Grenzen (`tests/test_edge_cases.py`).
- **Claims:** jede Zahl dieser README steht in `tests/test_claims.py` als wörtliche Tabellenzeile, Band oder Ungleichung; die Zahlen der Vormessungen werden nur als eingefroren geprüft, nicht nachgerechnet. Zusatzmessungen ohne Test: die Scratch-Läufe der Gitter-Prüfungen aus `spo-kuerzester-weg-demo` (siehe „Prüfungen“) und die Rohausgabe der Vormessung V4 (`tools/out_vormessung_v4.txt`, im Test nur als Struktur und Eckwerte geprüft).
- **Verdikte:** reine Funktionen, Schwellen und jede Zahl im Text (`tests/test_verdicts.py`); jedes Preset zeigt das erwartete Verdikt (`tests/test_app.py`).
- **Oberfläche:** AppTest (jeder Button, jeder Regler an den Grenzen, größte Live-Welt, Permalink mit Einrasten), Layout (Schnellstart und Kostenwelt in der Seitenleiste, Newsvendor-Regler im Hauptbereich, fester Kopf, aktives Preset hervorgehoben), Footer, Laufzeit-Importe, PDF.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Seitenleiste (Schnellstart, Kostenwelt), Ergebnis, Kernfrage, vier Experimente, Theorie, Grenzen, Formeln |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, scipy, pandas, plotly, fpdf2 |
| `requirements-dev.txt` | Laufzeit plus pytest und scikit-learn (nur für die Gegenproben der Orakel-Tests) |
| `zoi_rng.py`, `zoi_grid.py`, `zoi_world.py`, `zoi_losses.py`, `zoi_training.py`, `zoi_kernel.py`, `zoi_gap.py`, `zoi_runs.py`, `zoi_news.py` | Kern ohne Streamlit (siehe „Weitergabe“) |
| `zoi_bands.py` | lädt die vorgerechneten Bänder |
| `zoi_bands.json` | eigene Messung (`tools/measure_bands.py`) |
| `zoi_vormessung.json` | eingefrorene Zahlen der Vormessungen mit scikit-learn (`tools/freeze_vormessung.py`) |
| `zoi_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `zoi_evaluation.py` | Einstellungen, Zeilen und Zählungen der Experimente, Verdikte (reine Funktionen) |
| `zoi_layout.py` | Fester Kopf (Stylesheet) |
| `zoi_presets.py` | Permalink (Begrenzen und Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `zoi_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `zoi_pdf_export.py` | PDF-Export der aktuellen Welt (fpdf2) |
| `tools/measure_bands.py`, `tools/freeze_vormessung.py`, `tools/vormessung_v4.py` | erzeugen die Bänder, frieren die Vormessungen ein, messen die Lücke (V4); `tools/out_vormessung_v4.*` ist die Rohausgabe von V4 |
| `tests/` | Tests (siehe „Verifikation“); `tests/readme_tables.py` erzeugt die Tabellen dieser README |

## Bewusst nicht umgesetzt

- **Zufallswald live:** braucht scikit-learn, das nicht zu den Laufzeit-Paketen gehört; die Zahlen sind eingefroren. Ein Wald in numpy ist nicht Teil der Demo.
- **Kern-ETO bei n = 5000** und weitere flexible Lerner (abgestimmte Wälder, Netze): nicht gerechnet.
- **Messung der Raten** (Konvergenzgeschwindigkeit gegen n, Schätzung von α) und die Wohlspezifikation mit den 31 Monomen der Quelle: nicht nachgebaut.
- **Andere Gittergrößen, andere Wahrheiten, Minibatches, Lernratenwahl, Strafe für SPO+:** nicht gemessen (siehe `spo-kuerzester-weg-demo` für Strafe und früher Abbruch).
- **Gewichtete Stichprobe im Detail, direktes Lernen mit Strafe, Verfahrenswahl nach Diagnose:** gehören in andere Demos der Reihe (`gewichtete-stichprobe-demo`, `newsvendor-direkt-demo`).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
