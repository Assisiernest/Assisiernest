import nltk
from nltk.corpus import movie_reviews
import random

# Download required resources
nltk.download('movie_reviews')
nltk.download('punkt')

# Prepare dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# Feature extractor
def document_features(words):
    return {word: True for word in words}

# Train classifier (simple example)
train_set = [(document_features(d), c) for (d, c) in documents[:1500]]
test_set = [(document_features(d), c) for (d, c) in documents[1500:]]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Classify a given sentence
sentence = "This movie was amazing and full of suspense"
words = nltk.word_tokenize(sentence)
features = document_features(words)

print("Sentence:", sentence)
print("Classification:", classifier.classify(features))
