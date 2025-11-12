import math
from collections import defaultdict, Counter

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.class_word_counts = defaultdict(Counter)
        self.class_counts = Counter()
        self.vocabulary = set()
        self.class_prob = {}

    def fit(self, X, y):
        # X = list of token lists, y = list of labels
        for doc, label in zip(X, y):
            self.class_counts[label] += 1
            self.class_word_counts[label].update(doc)
            self.vocabulary.update(doc)
        
        total_docs = len(X)
        # prior probabilities P(c)
        self.class_prob = {c: count / total_docs for c, count in self.class_counts.items()}

    def predict(self, doc):
        # calculate log-probabilities for numerical stability
        log_probs = {}
        for c in self.class_counts:
            log_prob = math.log(self.class_prob[c])
            total_words_in_class = sum(self.class_word_counts[c].values())
            for word in doc:
                word_count = self.class_word_counts[c][word]
                # P(word|class) with Laplace smoothing
                prob = (word_count + self.alpha) / (total_words_in_class + self.alpha * len(self.vocabulary))
                log_prob += math.log(prob)
            log_probs[c] = log_prob
        # choose class with highest probability
        return max(log_probs, key=log_probs.get)
