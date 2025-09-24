import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')

# Read passage from file
# Example: assign passage directly or read from a file
passage = "This is an example passage to demonstrate stopword removal."

# Tokenize
words = nltk.word_tokenize(passage)

# Remove stopwords
filtered = [word for word in words if word.lower() not in stopwords.words("english")]

print("Original Passage:\n", passage)
print("\nAfter Removing Stop Words:\n", " ".join(filtered))
