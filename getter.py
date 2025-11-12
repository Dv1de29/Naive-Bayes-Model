
import csv
import re

stopwords = {"a","an","and","or","the","of","in","on","to","for","by","at","is","are","be","was","were"}
def cleanWords(words):
    return [word for word in words if word not in stopwords and len(word) > 2]


def tokenize(text):
    return cleanWords(re.findall(r"[a-zA-Z]+(?:-[a-zA-Z]+)*", text.lower()))

def load_csv_label_text(path, label_col=0, text_col=1, delimiter=',', header=False):
    labels = []
    docs = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        if header:
            next(reader)  # skip the header row
        for row in reader:
            label = row[label_col].strip()
            text = row[text_col].strip()
            tokens = tokenize(text)
            labels.append(label)
            docs.append(tokens)
    return docs, labels

if __name__ == "__main__":
    data_path = "./news_dataset.csv" 
    
    # Încarcă titlurile și etichetele
    docs, labels = load_csv_label_text(
        path=data_path,
        label_col=1,  # Sport
        text_col=0,   # Headline
        delimiter=',',
        header=True
    )

    print("Number of records:", len(docs))
    print("First headline tokens:", docs[0])
    print("First category:", labels[0])