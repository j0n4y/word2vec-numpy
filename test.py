import nltk
nltk.download('brown')
nltk.download('punkt_tab')

from nltk.corpus import brown
import string

def preprocess(sentences):
    cleaned = []
    for sent in sentences:
        words = [w.lower() for w in sent if w.isalpha()]
        if len(words) > 1:
            cleaned.append(" ".join(words))
    return cleaned

sentences = preprocess(brown.sents(categories='news'))

print(len(sentences))