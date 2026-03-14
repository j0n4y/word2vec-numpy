import numpy as np

class Word2vec:
    def __init__(self, vocab, window_size, embedding_dimension, num_negatives, init_alpha):
        self.vocab = vocab
        self.num_negatives = num_negatives
        self.pairs = self._generate_training_pairs(window_size)
        self.W1 = np.random.randn(vocab.vocab_size, embedding_dimension).astype(np.float32) * 0.01
        self.W2 = np.random.randn(vocab.vocab_size, embedding_dimension).astype(np.float32) * 0.01
        self.alpha = init_alpha

    def _generate_training_pairs(self, window_size):
        '''Compute the pairs of each word and its context words with a dynamic window size'''
        pairs = []
        for i in range(len(self.vocab.corpus)):
            target_idx = self.vocab.word2idx[self.vocab.corpus[i]]
            random_window_size = np.random.randint(1, window_size + 1) # dynamic window size
            start = max(0, i - random_window_size)
            end = min(len(self.vocab.corpus), i + random_window_size + 1)
            for j in range(start, end):
                if i != j:
                    context_idx = self.vocab.word2idx[self.vocab.corpus[j]]
                    pairs.append((target_idx, context_idx))
        return np.array(pairs, dtype=np.int32)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _forward_prop(self, target_idx, context_idx, negatives):
        '''Computes forward propagation for a pair and its negatives'''
        self.h = self.W1[target_idx] # uses index instead of dot product for performance
        self.score_pos = self._sigmoid(np.dot(self.W2[context_idx], self.h))
        self.score_negs = []
        for neg in negatives:
            self.score_negs.append(self._sigmoid(np.dot(self.W2[neg], self.h)))

    def _back_prop(self, target_idx, context_idx, negatives):
        '''Computes back propagation'''

        # Calculate error for context word
        dLdh = (self.score_pos - 1) * self.W2[context_idx]

        # Acumulate error from negatives
        for k in range(len(negatives)):
            dLdh += self.score_negs[k] * self.W2[negatives[k]]

        # Update W2 weights
        self.W2[context_idx] -= self.alpha * (self.score_pos - 1) * self.h
        for k in range(len(negatives)):
            self.W2[negatives[k]] -= self.alpha * self.score_negs[k] * self.h
        
        # Update W1 weights
        self.W1[target_idx] -= self.alpha * dLdh

    def train(self, epochs):
        init_alpha = self.alpha
        total_steps = epochs * len(self.pairs)
        step = 0
        for x in range(epochs):
            np.random.shuffle(self.pairs)

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

                # linear alpha decay
                step += 1
                self.alpha = init_alpha * (1 - step / total_steps)

                print(f"epoch {x+1}: {100 * i // len(self.pairs)}%", end="\r")
            print(f"epoch {x+1}: loss = {loss}")

    def most_similar(self, word, nPredictions):
        '''Finds words with the highest cosine similarity'''
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

    def save(self, path):
        np.savez(path, W1=self.W1, W2=self.W2)
        print(f"Weights saved to {path}.npz")

    def load(self, path):
        data = np.load(path)
        self.W1 = data["W1"]
        self.W2 = data["W2"]
        print(f"Weights loaded from {path}")