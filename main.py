import numpy as np

import nltk
from nltk.corpus import brown

# nltk.download('brown') # Execute this line only if brown corpus is not already downloaded
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
        freq = {} # count frequency of each word for computing noise distribution and subsampling of frequent words
        for sentence in sentences:
            for word in sentence:
                self.corpus.append(word)
                if self.word2idx.get(word) is None:
                    self.word2idx[word] = count
                    self.idx2word[count] = word
                    count += 1
                if freq.get(word) is None:
                    freq[word] = 0
                freq[word] += 1

        self.vocab_size = len(self.word2idx)
        self.corpus_size = len(self.corpus)

        # subsampling of frequent words
        t = 1e-5
        filtered_corpus = []
        for word in self.corpus:
            f = freq[word] / self.corpus_size 
            prob_discard = 1 - (t / f) ** 0.5
            if prob_discard < 0 or np.random.random() > prob_discard:
                filtered_corpus.append(word)
        self.corpus = filtered_corpus
        self.corpus_size = len(self.corpus)

        # compute noise distribution
        counts = [] 
        for i in range(self.vocab_size):
            counts.append(freq[self.idx2word[i]] ** 0.75)
        total = sum(counts)
        self.noise_distribution = []
        for i in range(len(counts)):
            self.noise_distribution.append(counts[i] / total)


class Word2vec:
    def __init__(self, vocab, window_size, embedding_dimension, num_negatives, init_alpha):
        self.vocab = vocab
        self.num_negatives = num_negatives
        self.pairs = self._generate_training_pairs(window_size)
        self.W1 = np.random.randn(vocab.vocab_size, embedding_dimension) * 0.01
        self.W2 = np.random.randn(embedding_dimension, vocab.vocab_size) * 0.01
        self.alpha = init_alpha

    def _generate_training_pairs(self, window_size):
        pairs = []
        for i in range(len(self.vocab.corpus)):
            target_idx = self.vocab.word2idx[self.vocab.corpus[i]]
            start = max(0, i - window_size)
            end = min(len(self.vocab.corpus), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    context_idx = self.vocab.word2idx[self.vocab.corpus[j]]
                    pairs.append((target_idx, context_idx))
        return pairs

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _forward_prop(self, target_idx, context_idx, negatives):
        self.h = self.W1[target_idx].flatten()
        self.score_pos = self._sigmoid(np.dot(self.W2.T[context_idx], self.h))
        self.score_negs = []
        for neg in negatives:
            self.score_negs.append(self._sigmoid(np.dot(self.W2.T[neg], self.h)))

    def _back_prop(self, target_idx, context_idx, negatives):
        dLdh = (self.score_pos - 1) * self.W2.T[context_idx]
        for k in range(len(negatives)):
            dLdh += self.score_negs[k] * self.W2.T[negatives[k]]

        self.W2.T[context_idx] -= self.alpha * (self.score_pos - 1) * self.h
        for k in range(len(negatives)):
            self.W2.T[negatives[k]] -= self.alpha * self.score_negs[k] * self.h
        
        self.W1[target_idx] -= self.alpha * dLdh

    def train(self, epochs):
        init_alpha = self.alpha
        total_steps = epochs * len(self.pairs)
        step = 0
        for x in range(epochs):
            loss = 0
            # pre-sample all negatives of this epoch at once
            neg_samples = np.random.choice(
                self.vocab.vocab_size,
                size=len(self.pairs) * self.num_negatives * 2,  # double because we have 2 invalid words per pair (target_idx and context_idx), enough in most cases
                p=self.vocab.noise_distribution
            )
            neg_list_idx = 0
            for i in range(len(self.pairs)):
                target_idx, context_idx = self.pairs[i]
                negatives = []
                while len(negatives) < self.num_negatives:
                    neg_idx = neg_samples[neg_list_idx % len(neg_samples)]
                    neg_list_idx += 1
                    if neg_idx != target_idx and neg_idx != context_idx:
                        negatives.append(neg_idx)
                self._forward_prop(target_idx, context_idx, negatives)
                self._back_prop(target_idx, context_idx, negatives)
                loss += -np.log(self.score_pos + 1e-9)
                for k in range(len(negatives)):
                    loss += -np.log(1 - self.score_negs[k] + 1e-9)

                step += 1
                self.alpha = init_alpha * (1 - step / total_steps)

                print(f"epoch {x+1}: {100 * i // len(self.pairs)}%", end="\r")
            print(f"epoch {x+1}: loss = {loss}")

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
word2vec = Word2vec(vocab, window_size=10, embedding_dimension=100, num_negatives=5, init_alpha=0.025)
word2vec.train(10)

while True:
    word = input("Enter a word: ")
    word2vec.most_similar(word, 5)