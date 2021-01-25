import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Embedding, Flatten, Dense
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

imdb_dir = 'aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
test_dir = os.path.join(imdb_dir, 'test')
glove_dir = 'glove.6B'
weights_file_name = 'pre_trained_glove_model.h5'

# Cut reviews after this number of words
max_length = 100
# Only consider at most this number of words in the dataset
num_words = 10000
# Dimension number of word vectors.
embedding_dim = 100


def load_imdb(data_dir: str, list_file_func):
    labels = []
    texts = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(data_dir, label_type)
        files = list_file_func(dir_name)
        for file_name in files:
            if file_name[-4:] == '.txt':
                file = open(os.path.join(dir_name, file_name), encoding='utf-8')
                texts.append(file.read())
                file.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)

    return texts, labels


def tokenize_data(texts, labels):
    num_training = 200
    num_validation = 10000

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=max_length)

    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    train_data = data[:num_training]
    train_labels = labels[:num_training]
    val_data = data[num_training: num_training + num_validation]
    val_labels = labels[num_training: num_training + num_validation]

    return tokenizer, train_data, train_labels, val_data, val_labels


def load_glove():
    embedding_dict = {}
    file = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf-8')
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_dict[word] = coefs
    file.close()

    print('Found %s word vectors.' % len(embedding_dict))
    return embedding_dict


def convert_embedding_array(tokenizer_word_dict, embedding_dict):
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in tokenizer_word_dict.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


def create_model() -> Model:
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def fit_pretrained_model(embedding_matrix, train_data, train_labels, val_data, val_labels):
    model = create_model()
    print(model.summary())

    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(train_data, train_labels,
                        epochs=10,
                        batch_size=32,
                        validation_data=(val_data, val_labels))
    model.save_weights(weights_file_name)
    return history


def fit_created_model(embedding_matrix, train_data, train_labels, val_data, val_labels):
    model = create_model()
    print(model.summary())

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(train_data, train_labels,
                        epochs=10,
                        batch_size=32,
                        validation_data=(val_data, val_labels))
    return history


def draw_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def evaluate_model(tokenizer: Tokenizer):
    texts, labels = load_imdb(test_dir, lambda dir_name: sorted(os.listdir(dir_name)))
    sequences = tokenizer.texts_to_sequences(texts)
    test_data = pad_sequences(sequences, maxlen=max_length)
    test_labels = np.asarray(labels)

    model = create_model()
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    model.load_weights(weights_file_name)
    model.evaluate(test_data, test_labels)


def main():
    texts, labels = load_imdb(train_dir, lambda dir_name: os.listdir(dir_name))
    tokenizer, train_data, train_labels, val_data, val_labels = tokenize_data(texts, labels)
    embedding_dict = load_glove()
    embedding_matrix = convert_embedding_array(tokenizer.word_index, embedding_dict)
    # history = fit_pretrained_model(embedding_matrix, train_data, train_labels, val_data, val_labels)
    history = fit_created_model(embedding_matrix, train_data, train_labels, val_data, val_labels)
    draw_history(history)
    evaluate_model(tokenizer)


if __name__ == '__main__':
    main()
