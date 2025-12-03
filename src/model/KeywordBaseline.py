import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.vectorizer import tfidf_vectorizer


class KeywordBaseline:
    def __init__(self, X_train, y_train):
        self.keyword_mapping = {}
        self.label_word_weights = {}
        self.default_label = 0
        self.vectorizer = tfidf_vectorizer
        self.X_train = self.vectorizer.fit_transform(X_train)
        self.y_train = y_train
    
    def extract_keywords(self, top_n=10):
        feature_names = self.vectorizer.get_feature_names_out()
        unique_labels = np.unique(self.y_train)

        for label in unique_labels:
            label_indices = np.where(self.y_train == label)[0]
            X_label = self.X_train[label_indices]
            word_weights = np.asarray(X_label.mean(axis=0)).flatten() # Average TF-IDF scores
            word_weights = word_weights / np.linalg.norm(word_weights) # Normalize
            
            self.label_word_weights[label] = word_weights # Store for prediction

            top_indices = word_weights.argsort()[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices if word_weights[i] > 0]
            self.keyword_mapping[label] = keywords

    def predict_weights(self, text):
        feature_names = self.vectorizer.get_feature_names_out()
        word_to_index = {w: i for i, w in enumerate(feature_names)}
        tokens = text.lower().split()

        scores = {}
        for label, weights in self.label_word_weights.items():
            score = 0.
            for w in tokens:
                if w in word_to_index:
                    idx = word_to_index[w]
                    score += weights[idx]
            scores[label] = score

        best_label = max(scores, key=scores.get)
        return best_label, scores
    
    def predict_cosine(self, text):
        x = self.vectorizer.transform([text])
        scores = {
            label: cosine_similarity(x, self.label_word_weights[label].reshape(1, -1))[0, 0]
            for label in self.label_word_weights
        }
        best_label = max(scores, key=scores.get)
        return best_label, scores
