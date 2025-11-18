\documentclass[12pt]{article}
\usepackage[romanian]{babel}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{geometry}
\usepackage{titlesec}
\usepackage{enumitem}
\geometry{a4paper, margin=1in}

\title{Clasificator de Titluri Sportive folosind Naive Bayes Multinomial}
\author{}
\date{}

\begin{document}

\maketitle

\section*{Descriere Generală}

Acest proiect implementează \textbf{de la zero} un model de clasificare a textelor folosind \textbf{Naive Bayes Multinomial}, având ca scop identificarea tipului de sport din titlurile știrilor. Programul încarcă date dintr-un fișier CSV, le procesează, le împarte în seturi de antrenare şi testare, antrenează modelul şi evaluează performanţa folosind acurateţe, matrice de confuzie şi clasificare detaliată.

\section*{Funcționalități principale}

\begin{itemize}[itemsep=2pt]
    \item Încărcarea datasetului din fișier CSV.
    \item Tokenizare completă a textului (regex, lowercase, eliminare stopword-uri).
    \item Implementare manuală a modelului Multinomial Naive Bayes.
    \item Aplicarea smoothing-ului Laplace ($\alpha = 1.0$).
    \item Filtrarea datelor pentru sporturile principale.
    \item Împărțire 80/20 între seturile de antrenare și test.
    \item Evaluare prin acuratețe, matrice de confuzie și classification report.
    \item Testare pe exemple reale și custom.
\end{itemize}

\section*{Structura Proiectului}

\begin{verbatim}
project/
│
├── MB.py             # Implementarea modelului Naive Bayes
├── getter.py         # Funcții de preprocesare
├── main.py           # Cod principal (training, testare)
├── news_dataset.csv  # Datasetul utilizat
└── README.tex        # Documentația proiectului
\end{verbatim}

\section*{Preprocesarea Textului}

Textul este supus următoarelor transformări:

\begin{enumerate}[itemsep=4pt]
    \item Transformare la lowercase.
    \item Extracție de tokeni cu expresie regulată: \\
    \verb![a-zA-Z]+(?:-[a-zA-Z]+)*!
    \item Eliminarea stopword-urilor precum: \emph{a, the, of, for, in, on, and}.
    \item Eliminarea cuvintelor cu lungime mai mică de 3 caractere.
    \item Returnarea listei de tokeni.
\end{enumerate}

\section*{Modelul Naive Bayes Multinomial}

Modelul este implementat fără biblioteci externe de machine learning. Pentru fiecare clasă se calculează:

\begin{itemize}
    \item numărul aparițiilor fiecărui cuvânt,
    \item numărul total de documente din clasă,
    \item priorul $P(c)$,
    \item probabilitatea fiecărui cuvânt cu Laplace smoothing.
\end{itemize}

Formula utilizată este:

\[
P(\text{class} \mid \text{words}) \propto
P(\text{class}) \prod_{\text{word}}
\frac{count(\text{word}, \text{class}) + \alpha}
{total\_words(\text{class}) + \alpha \cdot |V|}
\]

Predicția se obţine selectând clasa cu probabilitatea logaritmică maximă.

\section*{Rulare}

Asigură-te că ai Python instalat, apoi rulează:

\begin{verbatim}
python main.py
\end{verbatim}

\section*{Output generat}

La rulare, programul afișează:

\begin{itemize}
    \item numărul de mostre menținute după filtrare,
    \item distribuția datelor în train/test,
    \item acuratețea clasificatorului,
    \item predicții pentru exemple suplimentare,
    \item matricea de confuzie,
    \item classification report (precision, recall, F1-score).
\end{itemize}

\section*{Exemple de predicții}

Programul clasifica exemple precum:

\begin{itemize}
    \item \texttt{Messi scores a hat-trick for PSG} $\rightarrow$ Football
    \item \texttt{Virat Kohli scores century against Australia} $\rightarrow$ Cricket
    \item \texttt{Hamilton wins Italian Grand Prix} $\rightarrow$ Formula1
    \item \texttt{Curry drops 45 points in Warriors win} $\rightarrow$ Basketball
\end{itemize}

\section*{Posibile Extensii}

\begin{itemize}[itemsep=3pt]
    \item Introducerea vectorizării TF-IDF.
    \item Compararea cu implementarea scikit-learn.
    \item Curățarea și augmentarea datasetului.
    \item Implementarea Logistic Regression sau SVM.
    \item Crearea unei interfețe web pentru clasificare în timp real.
\end{itemize}

\section*{Licență}

Proiect open-source — codul poate fi utilizat liber pentru studiu și extindere.

\end{document}
