from keras.datasets import reuters
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import copy as cp

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print('train_data shape:', train_data.shape)
print('train_labels length:', len(train_labels))
print('test_data shape:', test_data.shape)
print('test_labels length:', len(test_labels))


def decode_newswire(newswire):
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, word) for (word, value) in word_index.items()])
    decoded_newswire = ' '.join([reverse_word_index.get(value - 3, '?') for value in newswire])
    print(decoded_newswire)


# decode_newswire(train_data[0])


def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.
    return results


DATA_VECTOR_DIMENSION = 10000
x_train = vectorize_sequences(train_data, DATA_VECTOR_DIMENSION)
x_test = vectorize_sequences(test_data, DATA_VECTOR_DIMENSION)


def to_one_hot(labels, dimension):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


LABEL_DIMENSION = 46
one_hot_train_labels = to_one_hot(train_labels, LABEL_DIMENSION)
one_hot_test_labels = to_one_hot(test_labels, LABEL_DIMENSION)


def train_validate():
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]

    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(DATA_VECTOR_DIMENSION,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
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
    model.add(layers.Dense(64, activation='relu', input_shape=(DATA_VECTOR_DIMENSION,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train,
              one_hot_train_labels,
              epochs=9,
              batch_size=512)
    loss, accuracy = model.evaluate(x_test, one_hot_test_labels)
    print('evaluate loss:', loss)
    print('evaluate accuracy:', accuracy)

    def predict():
        one_hot_predictions = model.predict(x_test)
        predictions = [np.argmax(one_hot_prediction) for one_hot_prediction in one_hot_predictions]
        comparisons = [(prediction, test_labels[i]) for i, prediction in enumerate(predictions)]
        for (prediction, test_label) in comparisons:
            print((prediction, test_label), ' ', prediction == test_label)

    # predict()


train_evaluate()


def random_evaluate():
    test_labels_copy = cp.copy(test_labels)
    np.random.shuffle(test_labels_copy)
    hits_array = np.array(test_labels) == np.array(test_labels_copy)
    result = float(np.sum(hits_array)) / len(test_labels)
    print(result)


random_evaluate()
