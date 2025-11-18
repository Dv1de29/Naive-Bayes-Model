import random
from sklearn.metrics import classification_report

from getter import *
from MB import *


if __name__ == "__main__":
    data_path = "./news_dataset.csv" 

    docs, labels = load_csv_label_text(
        path=data_path,
        label_col=1,
        text_col=0,  
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


    augmented_docs = list(train_docs)
    augmented_labels = list(train_labels) 

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
    # Football
    "Messi scores a hat-trick for PSG",  # Football
    "Ronaldo nets late winner for Al Nassr",  # Football
    "Liverpool dominate Manchester United in derby clash",  # Football
    "Real Madrid lift Champions League trophy",  # Football
    "Mbappé signs new contract with PSG",  # Football
    "Chelsea sack head coach after poor results",  # Football
    "Arsenal top Premier League after big win",  # Football
    "Barcelona claim victory in El Clasico",  # Football
    "Haaland scores five goals in Champions League",  # Football
    "Tottenham seal comeback win in London derby",  # Football
    "England secure Euro qualification spot",  # Football
    "Neymar returns from injury in Copa America",  # Football
    "Juventus beat Inter Milan in Serie A thriller",  # Football
    "Bayern Munich clinch Bundesliga title again",  # Football
    "AC Milan announce new midfield signing",  # Football
    "Argentina defeat Brazil in World Cup qualifier",  # Football
    "Manchester City crowned Premier League champions",  # Football,

    # Cricket
    "Virat Kohli scores century against Australia",  # Cricket
    "England win Ashes series at Lord’s",  # Cricket
    "India beat Pakistan in T20 World Cup",  # Cricket
    "Australia claim ODI victory over New Zealand",  # Cricket
    "Ben Stokes leads England to dramatic Test win",  # Cricket
    "Pakistan chase down record total in Asia Cup",  # Cricket
    "South Africa collapse in final innings",  # Cricket
    "Bumrah takes five wickets in second Test",  # Cricket
    "Rohit Sharma named new Indian captain",  # Cricket
    "New Zealand bowlers dominate in day-night match",  # Cricket
    "Bangladesh stun West Indies in major upset",  # Cricket
    "Steve Smith returns to form with unbeaten 120",  # Cricket
    "India clinch series after tight final match",  # Cricket
    "Dhoni announces retirement from international cricket",  # Cricket
    "Afghanistan secure historic Test win",  # Cricket
    "Sri Lanka edge past Zimbabwe in thriller",  # Cricket
    "West Indies post massive total in first innings",  # Cricket,

    # Basketball
    "LeBron James leads Lakers to victory",  # Basketball
    "Curry drops 45 points in Warriors win",  # Basketball
    "Giannis dominates as Bucks crush Celtics",  # Basketball
    "Doncic hits buzzer-beater to stun Clippers",  # Basketball
    "Durant returns from injury for Brooklyn Nets",  # Basketball
    "Embiid scores career-high in Sixers triumph",  # Basketball
    "Tatum leads Boston to Eastern Conference title",  # Basketball
    "Heat advance to NBA Finals after tough series",  # Basketball
    "Wembanyama shines in rookie debut for Spurs",  # Basketball
    "Jokic records triple-double in Nuggets win",  # Basketball
    "Mavericks beat Suns in overtime thriller",  # Basketball
    "Lillard hits game-winning three-pointer",  # Basketball
    "Bulls upset Raptors on the road",  # Basketball
    "Knicks sign major free agent ahead of season",  # Basketball
    "Lakers clinch playoff spot with big win",  # Basketball
    "Warriors celebrate back-to-back championships",  # Basketball
    "USA wins gold in Olympic basketball final",  # Basketball,

    # Tennis
    "Novak Djokovic wins Australian Open",  # Tennis
    "Rafael Nadal claims French Open title",  # Tennis
    "Carlos Alcaraz defeats Medvedev in US Open final",  # Tennis
    "Iga Swiatek dominates in WTA Finals",  # Tennis
    "Serena Williams announces retirement",  # Tennis
    "Roger Federer says goodbye after Laver Cup",  # Tennis
    "Coco Gauff wins first Grand Slam trophy",  # Tennis
    "Murray battles to five-set victory in Wimbledon",  # Tennis
    "Zverev advances to semifinal in Paris",  # Tennis
    "Sabalenka lifts Madrid Open trophy",  # Tennis
    "Djokovic breaks record for most Grand Slam wins",  # Tennis
    "Kyrgios fined after on-court outburst",  # Tennis
    "Nadal withdraws from tournament due to injury",  # Tennis
    "Tsitsipas shocks top seed in quarterfinal",  # Tennis
    "Osaka returns with win after long break",  # Tennis
    "Thiem wins comeback match after injury",  # Tennis
    "Sinner secures victory in Monte Carlo Masters",  # Tennis,

    # Baseball
    "Yankees beat Red Sox in extra innings",  # Baseball
    "Shohei Ohtani hits two home runs in win",  # Baseball
    "Dodgers clinch National League title",  # Baseball
    "Mets pitcher throws complete-game shutout",  # Baseball
    "Astros defeat Braves in World Series opener",  # Baseball
    "Aaron Judge breaks home run record",  # Baseball
    "Cubs rally late to beat Cardinals",  # Baseball
    "Padres sign star shortstop to record deal",  # Baseball
    "Phillies secure playoff spot with key victory",  # Baseball
    "Rangers win dramatic walk-off game",  # Baseball
    "Blue Jays pitcher strikes out twelve batters",  # Baseball
    "Giants claim series win over Rockies",  # Baseball
    "Yankees bullpen shuts down Orioles offense",  # Baseball
    "Nationals trade veteran catcher to Brewers",  # Baseball
    "Marlins edge Diamondbacks in ninth inning",  # Baseball
    "Tigers announce new manager for 2025 season",  # Baseball
    "Twins slugger hits grand slam in first inning",  # Baseball,

    # Formula1
    "Hamilton wins Italian Grand Prix",  # Formula1
    "Verstappen dominates Monaco Grand Prix",  # Formula1
    "Leclerc claims pole position in Bahrain",  # Formula1
    "Norris finishes second in thrilling race",  # Formula1
    "Ferrari unveil new car for upcoming season",  # Formula1
    "Perez secures victory at Singapore GP",  # Formula1
    "Mercedes team celebrates double podium finish",  # Formula1
    "Alonso returns to podium after strong drive",  # Formula1
    "Russell outpaces teammate in qualifying",  # Formula1
    "Red Bull confirm championship title win",  # Formula1
    "Sainz retires early due to engine failure",  # Formula1
    "Ricciardo makes comeback with AlphaTauri",  # Formula1
    "Piastri scores first points in Formula 1",  # Formula1
    "Hamilton criticizes tire strategy after loss",  # Formula1
    "McLaren introduce major aerodynamic upgrade",  # Formula1
    "Bottas sets fastest lap in practice session",  # Formula1
    "Gasly penalized for track limits violation",  # Formula1
]


    for headline in examples:
        print(headline, "->", clf.predict(tokenize(headline)))

    confusion = defaultdict(lambda: defaultdict(int))

    for doc, true_label in zip(test_docs, test_labels):
        predicted = clf.predict(doc)
        confusion[true_label][predicted] += 1

    sports = sorted(set(labels))
    print("\nConfusion Matrix:\n")
    print(" " * 15 + " ".join(f"{s[:10]:>10}" for s in sports))

    for true_label in sports:
        row = f"{true_label[:13]:>13} | " + " ".join(f"{confusion[true_label][pred]:>10}" for pred in sports)
        print(row)

    print(classification_report(test_labels, [clf.predict(d) for d in test_docs]))

