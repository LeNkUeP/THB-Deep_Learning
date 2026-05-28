This repo consists of submission task for the university module "Deep Learning". Web applications were created in which ai models and classifications were studied. Main purpose of the module was to understand these models, how to train, test, evaulate and how to improve results.

## EA 1 - Bilderkennung mit ml5

Erweitern Sie das ml5  Image Classification Tutorial so, dass ein Nutzer beliebige Bilder klassifizieren kann. Sie verwenden dazu wie im Tutorial ein bestehendes, vor-trainiertes Model. Sie müssen also nichts selber trainieren/anlernen. 

A1) Zeigen Sie Beispiel-Bilder, jeweils drei für korrekte und falsche Klassifikation, zusammen mit den Ergebnissen der Klassifikation als Diagramm (siehe Resultate und Visualisierung). Machen Sie bei der Darstellung deutlich, welche Bilder korrekt oder falsch klassifiziert werden. Die Bilder können aus dem ImageNet Datensatz zum Image Classification Tutorial stammen (siehe Daten) oder von Ihnen kommen. Sie können Bilder auch künstlich modifizieren. 

A2) Der Nutzer soll ein eigenes Bild in die Anwendung laden und klassifizieren lassen können (siehe Interaktion).

Interaktion: Der Nutzer kann ein Bild zur Klassifikation hochladen oder per Drag-and-drop in die Anwendung ziehen. Dieses wird zunächst dargestellt und dann automatisch oder mittels eines Buttons "Classify" klassifiziert.

Resultate und Visualisierung:

Das Netzwerk gibt eine Wahrscheinlichkeitsverteilung aus. Diese Werte kann man als Confidence interpretieren. Diese Confidence sollen Sie als Klassifikationsergebnis ausgeben beziehungsweise darstellen in Form eines Diagramms (Balken-, Pie-, etc.).  In den Diagrammen sollen auch die Zahlenwerte der Confidence in Prozent für die dargestellten Klassen stehen. Zur Visualisieren Ihrer Ergebnisse können Sie Bibliotheken wie z. B. Plotly oder Chart.js nutzen (siehe Libraries).

Layout: Stellen Sie Ihre Lösung in zwei Spalten dar, links das Bild und rechts das Diagramm mit der Klassifikation:

R1) Zunächst die drei richtig klassifizierte Bilder übereinander, dann die drei falsch Klassifizierungen.

R2)  Darunter das vom Nutzer geladene Bild mit den zugehörigen Interaktionselementen.

Diskussion: Diskutieren Sie Ihre Ergebnisse (unter den Resultaten auf der gleichen HTML-Seite, max. 10 Sätze). Was haben Sie beobachtet/gelernt?  Bei welchen Bildern hat die Klassifikation funktioniert und bei welchen nicht und warum ist das wohl so?

Dokumentation: Nutzen Sie die gleiche HTML-Seite (unter der Diskussion) wie zur Abgabe Ihrer Lösung zur Dokumentation der folgenden Aspekte:

1) Technisch: Listen Sie alle verwendeten Frameworks auf und erklären Sie kurz (1–3 Sätze) wozu Sie diese verwenden. Dokumentieren Sie technische Besonderheiten Ihrer Lösung.

2) Fachlich: Erläutern Sie Ihre Implementierung der Logik und alles, was für ihre Lösung wichtig ist (Ansatz, Resultate, Quellen, etc.)

[Lösung](https://lenkuep.github.io/THB-Deep_Learning/hello-ml5/)

## EA 2 - Regression mit FFNN

Nutzen Sie ein Feed-Forward Neural Network (FFNN) als Modell zur Regression der reellwertigen Funktion:  y(x) = 0.5*(x+0.8)*(x+1.8)*(x-0.2)*(x-0.3)*(x-1.9)+1 auf dem Interval [-2,+2] (Definitionsmenge). Wir wollen simulieren, wie es im richtigen Leben ist, das bedeutet, Sie kennen die Funktion y(x) nicht (das ist die unbekannte Ground-Truth). Stattdessen generieren Sie Daten von dieser Funktion und verrauschen diese.

A1) Zum Erzeugen der Daten generieren Sie N=100 zufällige, gleich-verteilte x Werte aus dem Intervall [-2,+2] (die Verteilung ist dabei flach, es gibt keine Anhäufung von x-Werten z. B. um null). Zu den x-Werten berechnen Sie y(x). Sie haben dann einen unverrauschten Datensatz mit 100 (x, y) = (Input, Output-Lable) Paaren. Diesen Datensatz teilen Sie auf in N/2=50 Trainingsdaten-Paare und N/2=50 Testdaten-Paare (die Aufteilung ist zufällig). Dann verrauschen Sie die Daten künstlich (Trainingsdaten und Testdaten). Dazu addieren Sie zu y (also zum Output/Lable) normal-verteiles Rauschen (Gaussian Noise) mit einer Varianz von V=0.05 (die x-Werte/Input bleiben dabei unverrauscht, wir simulieren nur sogenanntes Lable-Rauschen). Sie haben nun zwei Datensätze, einen ohne und einen mit Rauschen, jeweils aufgeteilt in Trainingsdaten und Testdaten.

A2) Den unverrauschten Datensatz benutzen Sie, um ein erstes Model zu trainieren. Da es kein Rauschen gibt (alle Daten liegen schön auf der Ground-Truth Funktion), sollte Ihr trainiertes Modell auf den Testdaten etwa genauso gut abschneiden wie auf den Trainingsdaten (Loss_train = Loss_test). Denn ohne Rauschen kann kein Overfitting auftreten (überlegen Sie, warum nicht).

Trainieren Sie zwei Modelle auf dem verrauschten Datensatz mit unterschiedlicher Anzahl an Trainings-Epochen (Epochs). 

A3) Das erste Modell (Best-Fit) soll möglichst gut generalisieren, also im Test einen möglichst kleinen Loss (MSE) aufweisen. 

A4) Versuchen Sie das zweite Model (Over-Fit) so lange zu trainieren, bis es overfittet, bis also der Trainings-Loss (MSE) deutlich besser (geringer) ist als der auf den Testdaten. 

Modell und Optimierung:  Nutzen Sie für Ihr Modell die folgende Netzwerkarchitektur und Parametern für die Neuronen und den Lernalgorithmus:

Anzahl der hidden Layer und Neuron pro Layer: 2 hidden Layer mit je 100 Neuronen (Sie können auch mit anderen/größeren Architekturen experimentieren)
Aktivierungsfunktionen in den hidden Layer: ReLU (Sie können auch mit anderen experimentieren)
Aktivierungsfunktionen im Output Layer (1 Neuron): linear (y=x) "none"
Als Loss nutzen Sie Mean-Squared-Error (MSE)
Lernrate und Optimizer: Adam mit Lernrate(Learning Rate)=0.01 und Batch-Size=32 (Sie können auch mit anderen experimentieren)
Anzahl der Trainings-Epochen (Epochs): Ausprobieren, dazu den Loss beobachten (Sie können dazu den Tensorflow (TF) Visor nutzen).
Datenpunkte: N=100 (50 Train + 50 Test) (Sie können auch hier experimentieren, und schauen was passiert, wenn Sie nur sehr wenig Daten haben)
Rauschen: Gaussian Noise mit einer Varianz V=0.05 
Experimente und Fragestellungen:  Was ist das beste Ergebnis (Loss/MSE), das Sie auf den beiden Datensetzen (mit und ohne Rauschen) erzielen können?  Unterscheiden Sie dabei sorgfältig zwischen Trainings- und Test-Loss. Stellen Sie die Anzahl der Trainings Epochs so ein, dass Sie das Phänomen Overfitting beobachten oder auch vermeiden können. 

Seien Sie sich bewusst darüber, dass man die Testdaten in keiner Weise zum Training oder zur Optimierung nutzen darf, sonst sind es keine unabhängigen Testdaten mehr. Man darf sie also auch nicht zum early Stopping, d. h. zur optimalen Einstellung der Trainings-Epochs benutzen. Dazu verwendet man normalerweise einen dritten, so genannten Validierungs-Datensatz (Validationset). Wir verzichten bei dieser Aufgabe zu Gunsten der Übersichtlichkeit auf einen solchen Validationset.

Resultate, Visualisierung und Layout: 

Visualisieren Sie die Daten, den Loss und die Vorhersagen innerhalb von Diagrammen/Plots (siehe Libraries). 

Stellen Sie auf einer HTML-Seite alle Resultate in zwei Spalten übereinander in der folgenden Reihenfolge dar:

R1) Die Datensätze: links ohne Rauschen, rechts mit Rauschen, Trainingsdaten und Testdaten farblich unterschieden jeweils zusammen in einem Diagramm.

R2) Die Vorhersage des Modells, das ohne Rauschen trainiert wurde y_unverrauscht(x), links auf den Trainingsdaten, rechts auf den Testdaten (beide ohne Rauschen).


R3) Die Vorhersage Ihres besten Modells y_best(x) trainiert auf den verrauschten Daten, links auf den Trainingsdaten, rechts auf den Testdaten (alles mit Rauschen).

R4) Die Vorhersage Ihres Overfit-Modells y_overfit(x) trainiert auf den verrauschten Daten, links auf den Trainingsdaten, rechts auf den Testdaten (alles mit Rauschen).

Schreiben Sie unter die Diagramme mit den Vorhersagen, den Loss=MeanSquaredError(MSE) jeweils für die Trainings- und Testdaten. Achten Sie darauf, dass die Werte für den Loss zu den dargestellten Daten passen.

Diskussion: Dokumentieren und begründen Sie Ihre Parameter und Einstellungen. Diskutieren Sie Ihre Ergebnisse (unter den Resultaten auf der gleichen HTML-Seite, max. 10 Sätze). Was haben Sie beobachtet/gelernt? 

Dokumentation: Nutzen Sie die gleiche HTML-Seite (unter der Diskussion) wie zur Abgabe Ihrer Lösung zur Dokumentation der folgenden Aspekte:

1) Technisch: Listen Sie alle verwendeten Frameworks auf und erklären Sie kurz (1–3 Sätze) wozu Sie diese verwenden. Dokumentieren Sie technische Besonderheiten Ihrer Lösung.

2) Fachlich: Erläutern Sie Ihre Implementierung der Logik und alles, was für ihre Lösung wichtig ist (Ansatz, Resultate, Quellen, etc.)

[Lösung](https://lenkuep.github.io/THB-Deep_Learning/reegression_ffnn/)

## EA 3 - Language Model mit LSTM

Erstellen Sie ein Language Models (LM) zur Wortvorhersage. Trainieren Sie dazu als Modell ein Long Short-Term Memory (LSTM) Netzwerk auf der Basis der Daten (siehe den Punkt „Daten“ unten) zur Wortvorhersage (Next Word Prediction). Mittels des trainierten LSTM Language-Models kann autoregressiv ein Text generiert werden, in dem das jeweils vorhergesagte Wort an den vorhandenen Text angehängt wird.

Modell und Optimierung:  Nutzen Sie für Ihr Modell die folgende Netzwerkarchitektur und Parametern für den Lernalgorithmus:

Stacked LSTM: 2 hidden Layer (in sich rekursiv) mit je 100 LSTM Units (Sie können auch mit anderen/größeren Architekturen experimentieren).
Softmax Output mit der Dimension des Dictionaries.
Als Loss nutzen Sie Cross-Entropy.
Lernrate und Optimizer: Adam mit Lernrate(Learning Rate)=0.01 und Batch-Size=32 (Sie können auch mit anderen experimentieren).
Anzahl der Trainings-Epochen (Epochs): Ausprobieren, dazu den Loss beobachten (Sie können dazu den Tensorflow (TF) Visor nutzen).
Interaktion: 

I1) Der Nutzer kann einen Text (Prompt) eingeben. Dieser sollte nur aus vollständigen, durch Leerzeichen getrennten Wörtern (Tokens) bestehen. Er kann dann jederzeit (am Ende eines vollständig eingegebenen Wortes) den Button „Vorhersage“ betätigen und erhält eine Darstellung der wahrscheinlichsten nun folgenden Wörter mit deren Wahrscheinlichkeiten. Er kann eines dieser Wörter auswählen, sodass es an den Text angehängt wird. Daraufhin beginnt automatisch eine neue Wortvorhersage.

I2) Der Nutzer kann mittels des „Weiter“ Buttons das wahrscheinlichste vorhergesagte Wort annehmen. Diese wird an den bisher eingegebenen Text angehängt, darauf beginnt automatisch eine neue Wortvorhersage. Der Nutzer kann also über wiederholtes Betätigen des „Weiter“ Buttons einen Text generieren lassen.

I3) Der Nutzer kann über einen „Auto“ Button automatisch bis zu 10 Wörter vorhersagen lassen. Diese automatische Vorhersage kann mittels eines „Stopp“ Buttons unterbrochen werden.

I4) Über ein „Reset“ Button werden der eingegebene Text und das Netzwerk zurückgesetzt.

Buttons: I1) Vorhersage, I2) Weiter, I3) Auto, Stopp und I1) die Auswahl eines der nächsten Wörter.

Experimente und Fragestellungen: 

1) Experimentieren Sie mit der Netzwerkarchitektur. Dokumentieren und begründen Sie Ihre finale Architektur.

2) Notieren Sie als Resultat, wie oft die Vorhersage genau richtig ist (k=1), und wie oft das korrekte nächste Wort unter den ersten k Worten, die Sie vorhersagen liegt, mit k gleich 5, 10, 20 und 100. Sie können auch die Perplexity (siehe Hintergrund) als Maß Ihrer Resultate nutzen.

3) Können Sie Ihre ursprünglichen Trainingsdaten mittels des trainierten Models rekonstruieren?  (überlegen Sie, ob sich daraus ein Datenschutzproblem ergibt).

Visualisierung: Sie können, dazu außer der API von TF, z. B. folgende Bibliotheken zur Visualisierung der Ergebnisse als Diagramm nutzen: Plotly, D3.

Diskussion: Diskutieren Sie Ihre Ergebnisse (unter den Resultaten auf der gleichen HTML-Seite, max. 10 Sätze). Was haben Sie beobachtet/gelernt? 

Dokumentation: Nutzen Sie die gleiche HTML-Seite (unter der Diskussion) wie zur Abgabe Ihrer Lösung zur Dokumentation der folgenden Aspekte:

1) Technisch: Listen Sie alle verwendeten Frameworks auf und erklären Sie kurz (1–3 Sätze) wozu Sie diese verwenden. Dokumentieren Sie technische Besonderheiten Ihrer Lösung.

2) Fachlich: Erläutern Sie Ihre Implementierung der Logik und alles, was für ihre Lösung wichtig ist (Ansatz, Resultate, Quellen, etc.)

[Lösung](https://lenkuep.github.io/THB-Deep_Learning/language_model_lstm/)
