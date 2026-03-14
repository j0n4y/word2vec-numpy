import os
import nltk
from nltk.corpus import brown
from vocab import Vocabulary
from word2vec import Word2vec

def preprocess(sentences):
    processed = []
    for sentence in sentences:
        words = []
        for word in sentence:
            if word.isalpha():
                words.append(word.lower())
        processed.append(words)
    return processed

def main():
    print("Loading and preprocessing Brown Corpus")
    nltk.download('brown') # Not necessary after first time. Just in case
    sentences = brown.sents()
    processed_sentences = preprocess(sentences)

    vocab = Vocabulary(processed_sentences, min_count=5)
    print(f"Vocab size: {vocab.vocab_size}")
    print(f"Corpus size: {vocab.corpus_size}")

    model = Word2vec(
        vocab=vocab, 
        window_size=5, 
        embedding_dimension=50, 
        num_negatives=15, 
        init_alpha=0.025 
    )

    if os.path.exists("weights.npz"):
        choice = input(f"Found existing weights. Load them? (y/n): ").lower()
        if choice == 'y':
            model.load("weights.npz")
        else:
            print("Starting training from scratch...")
            model.train(epochs=5)
            model.save("weights.npz")
    else:
        print("No weights.npz file found. Starting training from scratch...")
        model.train(epochs=5)
        model.save("weights.npz")

    print("\nTraining complete!")

    while True:
        user_word = input("\nEnter a word to find similar ones (or 'q' to quit): ").lower()
        if user_word == 'q':
            break
        model.most_similar(user_word, 5)

if __name__ == "__main__":
    main()