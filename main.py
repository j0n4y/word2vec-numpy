import numpy as np

# TODO: Replace with a data loader for larger datasets
sentences = [
    # PIZZA cluster — strong pizza+cheese+tomato+italian repetition
    "pizza is topped with melted cheese",
    "pizza has rich tomato sauce",
    "pizza is a classic italian dish",
    "i love pizza with extra cheese",
    "pizza always has tomato and cheese",
    "italian pizza has fresh tomato",
    "cheese makes pizza delicious",
    "tomato sauce is essential for pizza",
    "italians invented pizza",
    "pizza is baked in a hot oven",
    "the best pizza has mozzarella cheese",
    "pizza with tomato and mozzarella is classic",
    "i eat pizza at italian restaurants",
    "fresh tomato makes great pizza sauce",
    "pizza is the most popular italian food",

    # PASTA cluster — strong pasta+sauce+boil+italian repetition
    "pasta is boiled in salted water",
    "pasta is served with tomato sauce",
    "pasta is a staple italian dish",
    "i love pasta with cheese on top",
    "pasta and pizza are italian classics",
    "italian pasta is cooked al dente",
    "pasta sauce is made from tomato",
    "i boil pasta every week",
    "pasta with fresh ingredients is delicious",
    "the best pasta has homemade sauce",
    "pasta is my favorite italian meal",
    "i cook pasta at home often",
    "pasta goes well with cheese",
    "boiled pasta with tomato sauce is simple",
    "restaurants serve pasta with rich sauce",

    # CHEESE cluster — strong cheese+milk+flavor repetition
    "cheese is made from milk",
    "cheese has a rich salty flavor",
    "mozzarella is a soft italian cheese",
    "cheese melts beautifully on pizza",
    "i love strong flavored cheese",
    "aged cheese has intense flavor",
    "cheese and tomato complement each other",
    "italian cheese is world famous",
    "fresh cheese is made daily",
    "cheese is an essential cooking ingredient",
    "milk is used to produce cheese",
    "soft cheese melts easily when heated",
    "cheese adds flavor to every dish",
    "i buy fresh cheese at the market",
    "cheese pairs perfectly with pasta",

    # TOMATO cluster — strong tomato+sauce+fresh+red repetition
    "tomato is a red juicy fruit",
    "tomato sauce is cooked slowly",
    "fresh tomato has bright flavor",
    "tomato is the base of italian cooking",
    "i grow fresh tomato in my garden",
    "red tomato makes the best sauce",
    "tomato and cheese are a perfect pair",
    "cooking tomato slowly makes rich sauce",
    "fresh tomato is better than canned",
    "tomato is used in almost every italian dish",
    "i buy red tomato at the market",
    "tomato sauce simmers for hours",
    "fresh tomato goes well with cheese",
    "tomato gives pasta sauce its color",
    "italian cooking relies on fresh tomato",

    # COOKING cluster — strong cooking+kitchen+heat+recipe repetition
    "cooking requires fresh ingredients",
    "i love cooking italian recipes at home",
    "cooking pasta is easy and quick",
    "cooking pizza at home is fun",
    "good cooking needs patience and practice",
    "i learned cooking from my grandmother",
    "cooking with fresh tomato is best",
    "italian cooking is about simple ingredients",
    "cooking cheese melts it perfectly",
    "home cooking is better than restaurants",
    "i enjoy cooking every evening",
    "cooking requires a hot oven for pizza",
    "cooking pasta needs boiling water",
    "i practice cooking new recipes weekly",
    "cooking italian food brings me joy",

    # ITALIAN cluster — strong italian+food+culture+restaurant repetition
    "italian food is loved worldwide",
    "italy is famous for pizza and pasta",
    "italian restaurants are popular everywhere",
    "italian cooking uses simple fresh ingredients",
    "italian cuisine is rich in flavor",
    "i love eating at italian restaurants",
    "italian food relies on tomato and cheese",
    "italy has a great food culture",
    "italian dishes are easy to cook at home",
    "the italian kitchen is warm and inviting",
    "italian food traditions are centuries old",
    "i learned italian recipes from books",
    "italian cuisine inspired chefs worldwide",
    "eating italian food makes me happy",
    "italian cooking is an art form",
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
    
    def predict(self, word, nPredictions):
        if word in self.vocab.corpus:
            idx = self.vocab.word2idx[word]
            self._forward_prop(idx)
            top_words = []
            prev_max = 2
            for x in range(nPredictions):
                max = -1
                max_idx = 0
                for i in range(self.vocab.vocab_size):
                    if self.y_pred[i][0] > max and self.y_pred[i][0] < prev_max:
                        max = self.y_pred[i][0]
                        max_idx = i
                top_words.append(f"{self.vocab.idx2word[max_idx]}: {self.y_pred[max_idx][0]}")
                prev_max = max
            print(f"predictions for {word}: {top_words}")   
        else:
            print(f'word "{word}" is not in dictionary')

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

vocab = Vocabulary(sentences)
word2vec = Word2vec(vocab, 5, 10, 0.5)
word2vec.train(500)
word2vec.most_similar("pizza", 5)
word2vec.most_similar("pasta", 5)
word2vec.most_similar("cheese", 5)
word2vec.most_similar("tomato", 5)
word2vec.most_similar("cooking", 5)
word2vec.most_similar("italian", 5)