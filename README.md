# word2vec-numpy

From-scratch implementation of Word2Vec (Skip-gram with Negative Sampling) using only numpy.

## What is it

Word2Vec is a model that learns word embeddings: vector representation of words where similar words (in meaning or in syntax) are close in the vector space.

## How to use it

**Install requirements**
```
pip install -r requirements.txt
```

**Run**
```
python main.py
```

On the first run it will train and save weights to `weights.npz`. Afterwards, you can either train again from scratch or load those weights and use the model directly.

The parameters can be changed editing `main.py`:
```
model = Word2vec(
  vocab=vocab, 
  window_size=5, # number of words at each side that should be taken as context
  embedding_dimension=80, # dimension of the vector space
  num_negatives=10, # number of words not from context whose weight should be updated in each iteration (negative sampling)
  init_alpha=0.25 # initial learning rate
)
```

After training, you can make queries to see the 5 most similar words to the word you enter.

## Pre-trained Weights
If you want to skip the training process, you can download the pre-trained weights (`weights.npz`) from the [Releases](https://github.com/j0n4y/word2vec-numpy/releases/tag/v1.0.0) section of this repository. 
Place the file in the root directory and the model will detect it automatically.

## How does it work

The model trains on the Brown corpus. It makes pairs of every word in the corpus and its context words (after removing the least frequent words and subsampling) and computes the forward propagation using one hidden layer and the sigmoid activation function.
Then, it computes the back-propagation for the context words and for some non-context words (negative sampling), selected by an unigram distribution raised to the 3/4rd power.

Most of these choices, including the choice of Skip-gram over CBOW, come from the results shown in the original Word2Vec papers: [first one](https://arxiv.org/pdf/1301.3781) and [second one](https://arxiv.org/pdf/1310.4546). 

## Things to improve

If I had more time, I would implement many things to have a better performance and results:
- Currently, the training uses a python for loop for doing each forward and back propagation. This is very slow, as in every iteration it is aplying an overhead. The optimal implementation should use batches to take advantage of numpy.
- The model generates all the possible pairs at the start of the program. This affects scalability because it uses a lot of memory. The pairs should be computed in training time.
- The corpus used is small, and because the model is slow I did not have time to experiment with training, so the result is not as good as it could be.
- This model doesn't implement phrases, which is shown in the second paper of Word2Vec.


