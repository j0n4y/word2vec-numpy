import numpy as np

class Vocabulary:
    """Handles vocabulary mapping and corpus indexing"""
    def __init__(self, sentences, min_count=5):
        self.word2idx = {}
        self.idx2word = {}
        self.corpus = []
        self.vocab_size = 0
        self.corpus_size = 0
        self._build_vocab(sentences, min_count)

    def _build_vocab(self, sentences, min_count):
        count = 0
        freq = {} # count frequency of each word for removing rare words and computing subsampling of frequent words and noise distribution  
        full_corpus = []
        for sentence in sentences:
            for word in sentence:
                full_corpus.append(word)
                if self.word2idx.get(word) is None:
                    self.word2idx[word] = count
                    self.idx2word[count] = word
                    count += 1
                if freq.get(word) is None:
                    freq[word] = 0
                freq[word] += 1

        # remove rare words
        for word in full_corpus:
            if freq[word] < min_count:
                continue
            self.corpus.append(word)
            if self.word2idx.get(word) is None:
                self.word2idx[word] = count
                self.idx2word[count] = word
                count += 1

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