import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')

sentence="The cats are running and wolves are eating happily"

ps=PorterStemmer()

words=word_tokenize(sentence)

stems=[ps.stem(word) for word in words]

print("Original Sentence:",sentence)
print("After stemming:"," ".join(stems))