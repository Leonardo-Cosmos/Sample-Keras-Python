import os
import random

import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop

nietzsche_dir = 'nietzsche'

# Length of extracted character sequences
max_length = 60

# Sample a new sequence every 'step' characters
step = 3


def load_data():
    file_path = os.path.join(nietzsche_dir, 'nietzsche.txt')

    with open(file_path) as file:
        text = file.read().lower()

    print('Corpus length:', len(text))
    return text


def vectorize(text):
    sentences = []
    next_chars = []

    for i in range(0, len(text) - max_length, step):
        sentences.append(text[i:i + max_length])
        next_chars.append(text[i + max_length])

    print('Number of sequences:', len(sentences))

    chars = sorted(list(set(text)))
    print('Unique characters:', len(chars))
    char_indices = dict((char, chars.index(char)) for char in chars)

    print('Vectorization...')
    data = np.zeros((len(sentences), max_length, len(chars)), dtype=np.bool)
    results = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            data[i, t, char_indices[char]] = 1
        results[i, char_indices[next_chars[i]]] = 1

    return chars, char_indices, data, results


def build_model(chars):
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_length, len(chars))))
    model.add(Dense(len(chars), activation='softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model


def sample(char_predicts, temperature=1.0):
    char_predicts = np.asarray(char_predicts).astype('float64')
    char_predicts = np.log(char_predicts) / temperature
    exp_predicts = np.exp(char_predicts)
    char_predicts = exp_predicts / np.sum(exp_predicts)
    probabilities = np.random.multinomial(1, char_predicts, 1)
    return np.argmax(probabilities)


def generate(text, chars, char_indices, data, results, model: Sequential):
    for epoch in range(1, 61):
        print('epoch', epoch)
        model.fit(data, results, batch_size=128, epochs=1)
        start_index = random.randint(0, len(text) - max_length - 1)
        seed_text = text[start_index: start_index + max_length]
        print('Generating with seed:', seed_text)

        for temperature in [0.2, 0.5, 1.0, 1.2]:
            print('Temperature:', temperature)
            print(seed_text, end='')

            generated_seed_text = seed_text
            generated_chars = []
            for i in range(400):
                sampled = np.zeros((1, max_length, len(chars)))
                for t, char in enumerate(generated_seed_text):
                    sampled[0, t, char_indices[char]] = 1.

                preds = model.predict(sampled, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_char = chars[next_index]

                generated_seed_text += next_char
                generated_seed_text = generated_seed_text[1:]

                generated_chars.append(next_char)
            print(''.join(generated_chars))


def main():
    text = load_data()
    chars, char_indices, data, results = vectorize(text)
    model = build_model(chars)
    generate(text, chars, char_indices, data, results, model)


if __name__ == '__main__':
    main()
