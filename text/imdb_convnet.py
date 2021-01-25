from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# Number of words to consider as features
max_features = 10000
# Cut texts  after this number of words
max_length = 500

# Number of words to consider as features
max_features = 10000
# Cut texts  after this number of words
max_length = 500


def load_data():
    print('Loading data...')
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)
    print(len(train_data), 'train sequences')
    print(len(test_data), 'test sequences')

    print('Pad sequences (samples * time)')
    train_data = sequence.pad_sequences(train_data, maxlen=max_length)
    test_data = sequence.pad_sequences(test_data, maxlen=max_length)
    print('train input shape:', train_data.shape)
    print('test input shape:', test_data.shape)

    return train_data, train_labels, test_data, test_labels


def fit_conv(train_data, train_labels):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=max_length))
    model.add(Conv1D(32, 7, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(32, 7, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1))

    print(model.summary())

    model.compile(optimizer=RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(train_data, train_labels,
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
    train_data, train_labels, test_data, test_labels = load_data()
    history = fit_conv(train_data, train_labels)
    draw_history(history)


if __name__ == '__main__':
    main()
