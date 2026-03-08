import numpy as np

# TODO: Replace with a data loader for larger datasets
sentences = [
    "i like pizza with lot of cheese",
    "pizza has a lot of tomato",
    "cheese and tomato for the pizza",
    "we eat pizza everyday",
    "i love tomato"
]

class Vocabulary:
    """Handles vocabulary mapping and corpus indexing"""
    def __init__(self, sentences):
        self.word2idx = {}
        self.idx2word = {}
        self.corpus = []
        self.vocab_size = 0
        self.corpus_size = 0
        self._build_vocab(sentences)

    def _build_vocab(self, sentences):
        count = 0
        for row in sentences:
            for word in row.split():
                word = word.lower()
                self.corpus.append(word)

                if self.word2idx.get(word) is None:
                    self.word2idx[word] = count
                    self.idx2word[count] = word
                    count += 1
        self.vocab_size = len(self.word2idx)
        self.corpus_size = len(self.corpus)

vocab = Vocabulary(sentences)
