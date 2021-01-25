import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.layers import Embedding, SimpleRNN, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing import sequence

# Number of words to consider as features
max_features = 10000
# Cut texts  after this number of words
max_length = 500

batch_size = 32


def load_data():
    print('Loading data...')
    (train_input, train_labels), (test_input, test_labels) = imdb.load_data(num_words=max_features)
    print(len(train_input), 'train sequences')
    print(len(test_input), 'test sequences')

    print('Pad sequences (samples * time)')
    train_input = sequence.pad_sequences(train_input, maxlen=max_length)
    test_input = sequence.pad_sequences(test_input, maxlen=max_length)
    print('train input shape:', train_input.shape)
    print('test input shape:', test_input.shape)

    return train_input, train_labels, test_input, test_labels


def fit_simple_rnn(train_input, train_labels):
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(train_input, train_labels,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)
    return history


def fit_lstm(train_input, train_labels):
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(LSTM(32))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(train_input, train_labels,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)
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


def main():
    train_input, train_labels, test_input, test_labels = load_data()
    # history = fit_simple_rnn(train_input, train_labels)
    history = fit_lstm(train_input, train_labels)
    draw_history(history)


if __name__ == '__main__':
    main()
