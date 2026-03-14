import numpy as np

import nltk
from nltk.corpus import brown

nltk.download('brown') # Execute this line only if brown corpus is not already downloaded
sentences = brown.sents(categories='news')

def preprocess(sentences):
    processed = []
    for sentence in sentences:
        words = []
        for word in sentence:
            if word.isalpha():
                words.append(word.lower())
        processed.append(words)
    return processed

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
        for sentence in sentences:
            for word in sentence:
                self.corpus.append(word)
                if self.word2idx.get(word) is None:
                    self.word2idx[word] = count
                    self.idx2word[count] = word
                    count += 1
        self.vocab_size = len(self.word2idx)
        self.corpus_size = len(self.corpus)

class Word2vec:
    def __init__(self, vocab, window_size, embedding_dimension, init_alpha):
        self.vocab = vocab
        self.pairs = self._generate_training_pairs(window_size)
        self.W1 = np.random.randn(vocab.vocab_size, embedding_dimension) * 0.01
        self.W2 = np.random.randn(embedding_dimension, vocab.vocab_size) * 0.01
        self.alpha = init_alpha

    def _generate_training_pairs(self, window_size):
        """Generates (target, context) pairs from corpus with the window-size specified (stores indices instead of creating one-hot vectors to optimize speed)"""
        pairs = []
        for i in range(len(self.vocab.corpus)):
            word = self.vocab.corpus[i]
            target_idx = self.vocab.word2idx[word]
            start = max(0, i - window_size)
            end = min(len(self.vocab.corpus), i + window_size + 1)
            for j in range(start, end):
                if i != j: 
                    context_word = self.vocab.corpus[j]
                    context_idx = self.vocab.word2idx[context_word]
                    pairs.append((target_idx, context_idx))       
        return np.array(pairs)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x)) # substracting the max to prevent overflow
        return e_x / e_x.sum()

    def _forward_prop(self, target_idx):
        """Computers the forward propagation to predict context word probabilities given the index of a target word."""
        self.h = self.W1[target_idx].reshape(-1, 1)
        self.u = np.dot(self.W2.T, self.h)
        self.y_pred = self._softmax(self.u)

    def _back_prop(self, target_idx, context_idx):
        self.y_pred[context_idx] -= 1
        dLdW2 = np.dot(self.h, self.y_pred.T)
        dLdW1 = np.dot(self.W2, self.y_pred).T
        self.W2 = self.W2 - self.alpha * dLdW2
        self.W1[target_idx] = self.W1[target_idx] - self.alpha * dLdW1

    def train(self, epochs):
        for x in range(0, epochs):
            loss = 0
            for target_idx, context_idx in self.pairs:
                self._forward_prop(target_idx)
                self._back_prop(target_idx, context_idx)
                loss += -self.u[context_idx][0] + np.log(np.sum(np.exp(self.u)))
            print(f"epoch {x+1}: loss = {loss}")
            self.alpha *= 0.9 

    def most_similar(self, word, nPredictions):
        if word in self.vocab.word2idx:
            idx = self.vocab.word2idx[word]
            vec = self.W1[idx]
            top_words = []
            prev_max = 2
            for x in range(nPredictions):
                max_sim = -2
                max_idx = 0
                for i in range(self.vocab.vocab_size):
                    if i != idx:
                        cos_sim = np.dot(vec, self.W1[i]) / (np.linalg.norm(vec) * np.linalg.norm(self.W1[i]))
                        if cos_sim > max_sim and cos_sim < prev_max:
                            max_sim = cos_sim
                            max_idx = i
                top_words.append(f"{self.vocab.idx2word[max_idx]}: {max_sim}")
                prev_max = max_sim
            print(f"predictions for {word}: {top_words}")
        else:
            print(f'word "{word}" is not in dictionary')

vocab = Vocabulary(preprocess(sentences))
word2vec = Word2vec(vocab, 1, 1, 0.5)
word2vec.train(1)