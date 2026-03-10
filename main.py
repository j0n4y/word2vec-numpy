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

def generate_training_pairs(vocab, window_size):
    """Generates (target, context) pairs from corpus with the window-size specified (stores indices instead of creating one-hot vectors to optimize speed)"""
    pairs = []
    for i in range(len(vocab.corpus)):
        word = vocab.corpus[i]
        target_idx = vocab.word2idx[word]
        start = max(0, i - window_size)
        end = min(len(vocab.corpus), i + window_size + 1)
        for j in range(start, end):
            if i != j: 
                context_word = vocab.corpus[j]
                context_idx = vocab.word2idx[context_word]
                pairs.append((target_idx, context_idx))       
    return np.array(pairs)

def softmax(x):
    e_x = np.exp(x - np.max(x)) # substracting the max to prevent overflow
    return e_x / e_x.sum()

def forward_prop(W1, W2, target_idx):
    """Computers the forward propagation to predict context word probabilities given the index of a target word."""
    h = W1[target_idx].reshape(-1, 1)
    u = np.dot(W2.T, h)
    y_pred = softmax(u)
    return y_pred, h, u

def back_prop(W1, W2, target_idx, context_idx, y_pred, h, u, alpha):
    y_pred[context_idx] -= 1
    dLdW2 = np.dot(h, y_pred.T)
    dLdW1 = np.dot(W2, y_pred).T
    W2 = W2 - alpha * dLdW2
    W1[target_idx] = W1[target_idx] - alpha * dLdW1
    return W1, W2

def train(epochs, W1, W2, pairs):
    for x in range(0, epochs):
        loss = 0
        for target_idx, context_idx in pairs:
            y_pred, h, u = forward_prop(W1, W2, target_idx)
            W1, W2 = back_prop(W1, W2, target_idx, context_idx, y_pred, h, u, 0.1)
            loss += -u[context_idx][0] + np.log(np.sum(np.exp(u)))
        print(f"epoch {x+1}: loss = {loss}")

vocab = Vocabulary(sentences)
pairs = generate_training_pairs(vocab, 1)

# change 2 for the embedding dimension
W1 = np.random.randn(vocab.vocab_size, 2) * 0.01
W2 = np.random.randn(2, vocab.vocab_size) * 0.01

train(100000, W1, W2, pairs)


