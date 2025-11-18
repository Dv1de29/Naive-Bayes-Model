# 📚 Clasificator de Titluri Sportive - Naive Bayes Multinomial

Acest proiect implementează \\textbf{de la zero} un model de clasificare a textelor folosind algoritmul **Naive Bayes Multinomial** în Python, fără a folosi biblioteci de machine learning pre-existente (precum `scikit-learn`) pentru logica de clasificare.

## 📁 Structura Proiectului

În directorul proiectului, găsești următoarele fișiere cheie:

```
project/
│
├── MB.py             # Implementarea clasei Naive Bayes Multinomial (core logic)
├── getter.py         # Funcții pentru preprocesarea textului și încărcarea datelor
├── main.py           # Script principal pentru training, testare și evaluare
└── news_dataset.csv  # Dataset-ul de știri sportive
```

-----

## 🛠️ Pregătirea Proiectului

Asigură-te că ai **Python 3.x** instalat.

### Dependențe

Acest proiect necesită biblioteca `pandas` (pentru citirea fișierului CSV) și, opțional, `numpy`.

În directorul proiectului, poți rula:

```bash
pip install pandas numpy
```

-----

## 🏃 Scripturi Disponibile

În directorul proiectului, poți rula:

### `python main.py`

Rulează întregul flux de lucru al clasificatorului:

1.  **Încarcă** și **preprocesează** datele din `news_dataset.csv`.
2.  **Antrenează** modelul **Naive Bayes Multinomial** implementat manual.
3.  **Evaluează** performanța pe setul de testare.
4.  **Afișează** acuratețea, matricea de confuzie și raportul de clasificare detaliat.
5.  **Testează** modelul pe exemple predefinite și afișează predicțiile.

-----

## 📖 Învățare Suplimentară

Poți aprofunda conceptele folosite în acest proiect consultând următoarele resurse:

### Naive Bayes & NLP

  * **Multinomial Naive Bayes:** Află despre fundamentul acestui clasificator, adesea folosit pentru clasificarea documentelor.
  * **Laplace Smoothing (Additive Smoothing):** Înțelege de ce este necesar să adaugi $\alpha=1$ pentru a gestiona cuvintele care nu apar în setul de antrenare.
  * **Text Preprocessing:** Studiază etapele de tokenizare, eliminare a *stopwords*-urilor și impactul lor asupra performanței.

-----

## 🎯 Evaluarea Performanței

Când rulezi `main.py`, rezultatul va include:

### Acuartețea (Accuracy)

Acuratețea generală a clasificatorului pe setul de test.

### Matricea de Confuzie (Confusion Matrix)

O vizualizare tabelară a predicțiilor corecte și incorecte (True Positives, False Positives, etc.) pentru fiecare clasă.

### Raportul de Clasificare (Classification Report)

Acesta oferă metrici esențiale per clasă:

  * **Precision (Precizie):** Din toate instanțele clasificate ca fiind o anumită clasă, cât de multe au fost corecte.
  * **Recall (Rechemare):** Din toate instanțele care *ar fi trebuit* să fie clasificate ca o anumită clasă, câte au fost clasificate corect.
  * **F1-Score:** Media armonică a Preciziei și Recall-ului, utilă mai ales în cazul dataset-urilor dezechilibrate.
