from keras import preprocessing
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

embedding_layer = Embedding(1000, 64)


def main():
    # Number of words to consider as features
    max_features = 10000
    # Cut texts  after this number of words
    max_length = 20

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)

    train_data = preprocessing.sequence.pad_sequences(train_data, maxlen=max_length)
    test_data = preprocessing.sequence.pad_sequences(test_data, maxlen=max_length)

    model = Sequential()
    model.add(Embedding(10000, 8, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    print(model.summary())

    history = model.fit(train_data, train_labels,
                        epochs=10,
                        batch_size=32,
                        validation_split=0.2)


if __name__ == '__main__':
    main()
