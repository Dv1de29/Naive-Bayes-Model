import random

from getter import *
from MB import *


if __name__ == "__main__":
    data_path = "./news_dataset.csv" 

    docs, labels = load_csv_label_text(
        path=data_path,
        label_col=1,  # Sport
        text_col=0,   # Headline
        delimiter=',',
        header=True
    )

    top_sports = {'Football', 'Cricket', 'Basketball', 'Tennis', 'Baseball', 'Formula1'}
    # top_sports = {'Wrestling', 'Gymnastics', 'Tae Kwon Do', 'Snowboarding', 'Archery', 'Skiing', 'Table Tennis', 'Basketball', 'Cricket', 'Rugby', 'Volleyball', 'Fencing', 'Surfing', 'Tennis', 'Football', 'Formula1', 'Esports', 'Boxing', 'Ice Hockey', 'Golf', 'Badminton', 'Horse Racing', 'American Football', 'Athletics', 'MMA', 'Swimming', 'Baseball', 'Cycling'}


    filtered = [(d, l) for d, l in zip(docs, labels) if l in top_sports]
    docs, labels = zip(*filtered)

    print(f"Filtered samples: {len(docs)}")
    print(f"Remaining categories: {set(labels)}")

    #Shuffle of the data
    combined = list(zip(docs, labels))
    random.shuffle(combined)
    docs, labels = zip(*combined)



    split_ratio = 0.8
    split_index = int(len(docs) * split_ratio)

    train_docs = docs[:split_index]
    train_labels = labels[:split_index]

    test_docs = docs[split_index:]
    test_labels = labels[split_index:]

    print(Counter(train_labels))
    print(Counter(test_labels))

    print(f"Total samples: {len(docs)}")
    print(f"Training: {len(train_docs)}")
    print(f"Testing: {len(test_docs)}")
    print(set(labels))


    # clf = MultinomialNB(alpha=1.0)
    # clf.fit(train_docs, train_labels)

    augmented_docs = list(train_docs)
    augmented_labels = list(train_labels)

    # sport_keywords = {
    #     'Football': ['goal','penalty','hat', 'trick','offside','corner'],
    #     'Basketball': ['free_throw','three_pointer','dunk','alley_oop','shot_clock'],
    #     'Formula1': ['grand_prix','pole_position','lap_time','pit_stop','qualifying'],
    #     'Tennis': ['break_point','ace','deuce','backhand','grand_slam'],
    #     'Baseball': ['home_run','innings','pitcher','bullpen','strikeout'],
    #     'Cricket': ['wicket','overs','innings','bowled','test_match']
    # }

    # print(set(labels))

    # for sport, keywords in sport_keywords.items():
    #     for _ in range(100): 
    #         augmented_docs.append(keywords) 
    #         augmented_labels.append(sport)   

    

    # Train once with the augmented data
    clf = MultinomialNB(alpha=1.0)
    clf.fit(augmented_docs, augmented_labels)

    correct = 0
    for doc, label in zip(test_docs, test_labels):
        predicted = clf.predict(doc)
        if predicted == label:
            correct += 1

    accuracy = correct / len(test_labels)
    print(f"Accuracy: {accuracy:.4f}")

    examples = [
        "Messi scores a hat-trick for PSG",  # Football
        "LeBron James leads Lakers to victory",  # Basketball
        "Hamilton wins Italian Grand Prix",  # Formula1
        "UFC fighter wins via knockout",  # Combat Sports
        "Novak Djokovic wins Australian Open"  # Tennis
    ]

    for headline in examples:
        print(headline, "->", clf.predict(tokenize(headline)))

    confusion = defaultdict(lambda: defaultdict(int))

    for doc, true_label in zip(test_docs, test_labels):
        predicted = clf.predict(doc)
        confusion[true_label][predicted] += 1

    # Print confusion matrix header
    sports = sorted(set(labels))
    print("\nConfusion Matrix:\n")
    print(" " * 15 + " ".join(f"{s[:10]:>10}" for s in sports))  # header row

    # Print each row of the matrix
    for true_label in sports:
        row = f"{true_label[:13]:>13} | " + " ".join(f"{confusion[true_label][pred]:>10}" for pred in sports)
        print(row)


