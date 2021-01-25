from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print('train_data shape:', train_data.shape)
print('train_labels length:', len(train_labels))
print('test_data shape:', test_data.shape)
print('test_labels length:', len(test_labels))


# print('train_data max value:', max([max(sequence) for sequence in train_data]))


def decode_review(review):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, word) for (word, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(value - 3, '?') for value in review])
    print(decoded_review)


# decode_review(train_data[0])


def vectorize_sequences(sequences):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


DATA_VECTOR_DIMENSION = 10000
x_train = vectorize_sequences(train_data, DATA_VECTOR_DIMENSION)
x_test = vectorize_sequences(test_data, DATA_VECTOR_DIMENSION)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


def train_validate():
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(DATA_VECTOR_DIMENSION,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    history_dict = history.history
    loss_list = history_dict['loss']
    val_loss_list = history_dict['val_loss']
    accuracy_list = history_dict['accuracy']
    val_accuracy_list = history_dict['val_accuracy']

    epochs = range(1, len(loss_list) + 1)

    plt.plot(epochs, loss_list, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_list, 'b', label='Validation loss')
    plt.plot(epochs, accuracy_list, 'ro', label="Training accuracy")
    plt.plot(epochs, val_accuracy_list, 'r', label="Validation accuracy")
    plt.title('Training and validation loss with acc')
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.legend()
    plt.show()


# train_validate()


def train_evaluate():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(DATA_VECTOR_DIMENSION,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train,
              y_train,
              epochs=4,
              batch_size=512)
    loss, accuracy = model.evaluate(x_test, y_test)
    print('evaluate loss:', loss)
    print('evaluate accuracy:', accuracy)

    def predict():
        predictions = model.predict(x_test)
        comparisons = [(prediction[0], y_test[i]) for i, prediction in enumerate(predictions)]
        for comparison in comparisons:
            print(comparison)

    predict()


train_evaluate()
