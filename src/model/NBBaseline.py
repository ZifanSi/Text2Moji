import numpy as np
from src.vectorizer import tfidf_vectorizer, count_vectorizer
from sklearn.naive_bayes import MultinomialNB

class NBBaseline:
    def __init__(self, X_train, y_train):
        self.class_priors = {} # P(c)
        self.word_likelihoods = {} # P(w|c)
        self.vocab_size = 0
        self.vectorizer = count_vectorizer
        self.X_train = self.vectorizer.fit_transform(X_train)
        self.y_train = y_train
        self.model = MultinomialNB()

    def train(self):
        model = MultinomialNB()
        model.fit(self.X_train, self.y_train)
        self.model = model

    def predict(self, text):
        x = self.vectorizer.transform([text])
        predicted_label = self.model.predict(x)[0]
        log_probs = self.model.predict_log_proba(x)[0]
        scores = {
            label: log_prob
            for label, log_prob in zip(self.model.classes_, log_probs)
        }
        return predicted_label, scores