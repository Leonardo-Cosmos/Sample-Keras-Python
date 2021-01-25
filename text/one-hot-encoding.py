from keras.preprocessing.text import Tokenizer
import numpy as np
import string


def encode_words(samples):
    token_index = {}
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    max_length = 10

    results = np.zeros(shape=(len(samples),
                              max_length,
                              max(token_index.values()) + 1))

    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1

    results


def encode_characters(samples):
    characters = string.printable
    token_index = dict(zip(characters, range(1, len(characters) + 1)))

    max_length = 50

    results = np.zeros((len(samples),
                        max_length,
                        max(token_index.values()) + 1))

    for i, sample in enumerate(samples):
        for j, character in enumerate(sample):
            index = token_index.get(character)
            results[i, j, index] = 1

    results


def encode_words_keras(samples):
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(samples)

    sequences = tokenizer.texts_to_sequences(samples)

    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))


def encode_words_hash(samples):
    dimensionality = 1000
    max_length = 10
    results = np.zeros((len(samples), max_length, dimensionality))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = abs(hash(word)) % dimensionality
            results[i, j, index] = 1

    results


def main():
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']
    # encode_words(samples)
    # encode_characters(samples)
    # encode_words_keras(samples)
    encode_words_hash(samples)


if __name__ == '__main__':
    main()
