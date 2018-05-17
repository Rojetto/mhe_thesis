Hinweise zur Ausführung unter Windows
=====================================

* Abhängigkeiten aus requirements.txt installieren (Achtung: pymoskito Fork der mehr Beobachterausgänge erlaubt)
* numpy mit pip deinstallieren
* numpy+mkl von https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy installieren

Das ist notwendig, weil cvxopt unter Windows eine mit MKL gelinkte Numpy-Version benötigt.