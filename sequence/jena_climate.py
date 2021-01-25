import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Flatten, Dense, GRU, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras.optimizers import RMSprop

jena_dir = 'jena'

# look_back = 1440 # For RNN
# step = 6 # For RNN
look_back = 720  # CNN + RNN
step = 3  # For CNN + RNN
delay = 144
batch_size = 128

train_sample_num = 200000
validate_sample_num = 100000


def load_data():
    file_name = os.path.join(jena_dir, 'jena_climate_2009_2016.csv')

    file = open(file_name)
    data = file.read()
    file.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    print('Data table header:', header)
    print('Data row count:', len(lines))

    return header, lines


def parse_data(header, lines):
    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values
    return float_data


def draw_time_sequence(float_data):
    temp = float_data[:, 1]
    plt.plot(range(len(temp)), temp)

    plt.show()


def normalize_data(float_data):
    mean = float_data[:train_sample_num].mean(axis=0)
    float_data -= mean
    std = float_data[:train_sample_num].std(axis=0)
    float_data /= std


def generator(data, look_back, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + look_back
    while True:
        if shuffle:
            rows = np.random.randint(min_index + look_back, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + look_back
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), look_back // step, data.shape[-1]))
        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):
            indices = range(rows[j] - look_back, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets


def prepare_generator(float_data):
    train_min_index = 0
    train_max_index = train_sample_num
    train_gen = generator(float_data,
                          look_back=look_back,
                          delay=delay,
                          min_index=train_min_index,
                          max_index=train_max_index,
                          shuffle=True,
                          step=step,
                          batch_size=batch_size)

    validate_min_index = train_sample_num + 1
    validate_max_index = train_sample_num + validate_sample_num
    validate_gen = generator(float_data,
                             look_back=look_back,
                             delay=delay,
                             min_index=validate_min_index,
                             max_index=validate_max_index,
                             shuffle=False,
                             step=step,
                             batch_size=batch_size)

    test_min_index = train_sample_num + validate_sample_num + 1
    test_gen = generator(float_data,
                         look_back=look_back,
                         delay=delay,
                         min_index=test_min_index,
                         max_index=None,
                         shuffle=False,
                         step=step,
                         batch_size=batch_size)

    validate_steps = (validate_max_index - validate_min_index - look_back) // batch_size
    test_steps = (len(float_data) - test_min_index - look_back) // batch_size

    return train_gen, validate_gen, test_gen, validate_steps, test_steps


def evaluate_naive_method(validate_gen, validate_step_num):
    batch_maes = []
    for step in range(validate_step_num):
        samples, targets = next(validate_gen)
        predictions = samples[:, -1, 1]
        mae = np.mean(np.abs(predictions - targets))
        batch_maes.append(mae)

    print('MAE of naive method:', np.mean(batch_maes))


def fit_simple_model(float_data, train_gen, validate_gen, validate_steps):
    model = Sequential()
    model.add(Flatten(input_shape=(look_back // step, float_data.shape[-1])))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(),
                  loss='mae')
    history = model.fit(train_gen,
                        steps_per_epoch=500,
                        epochs=20,
                        validation_data=validate_gen,
                        validation_steps=validate_steps)
    draw_history_loss(history)


def draw_history_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def fit_gru_model(float_data, train_gen, validate_gen, validate_steps):
    model = Sequential()
    model.add(GRU(32, input_shape=(None, float_data.shape[-1])))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(),
                  loss='mae')
    history = model.fit(train_gen,
                        steps_per_epoch=500,
                        epochs=20,
                        validation_data=validate_gen,
                        validation_steps=validate_steps)
    draw_history_loss(history)


def fit_gru_dropout_model(float_data, train_gen, validate_gen, validate_steps):
    model = Sequential()
    model.add(GRU(32,
                  dropout=0.2,
                  recurrent_dropout=0.2,
                  input_shape=(None, float_data.shape[-1])))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(),
                  loss='mae')
    history = model.fit(train_gen,
                        steps_per_epoch=500,
                        epochs=20,
                        validation_data=validate_gen,
                        validation_steps=validate_steps)
    draw_history_loss(history)


def fit_bidirectional_gru(float_data, train_gen, validate_gen, validate_steps):
    model = Sequential()
    model.add(Bidirectional(GRU(32), input_shape=(None, float_data.shape[-1])))
    model.add(Dense(1))

    model.compile(optimizer=RMSprop(),
                  loss='mae')
    history = model.fit(train_gen,
                        steps_per_epoch=500,
                        epochs=40,
                        validation_data=validate_gen,
                        validation_steps=validate_steps)
    draw_history_loss(history)


def fit_conv_model(float_data, train_gen, validate_gen, validate_steps):
    model = Sequential()
    model.add(Conv1D(32, 5, activation='relu',
                     input_shape=(None, float_data.shape[-1])))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(32, 5, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(32, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1))

    model.compile(optimizer=RMSprop(),
                  loss='mae')

    history = model.fit(train_gen,
                        steps_per_epoch=500,
                        epochs=20,
                        validation_data=validate_gen,
                        validation_steps=validate_steps)
    draw_history_loss(history)


def fit_conv_gru_model(float_data, train_gen, validate_gen, validate_steps):
    model = Sequential()
    model.add(Conv1D(32, 5, activation='relu',
                     input_shape=(None, float_data.shape[-1])))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(32, 5, activation='relu'))
    model.add(GRU(32, dropout=0.1, recurrent_dropout=0.5))
    model.add(Dense(1))

    print(model.summary())

    model.compile(optimizer=RMSprop(),
                  loss='mae')
    history = model.fit(train_gen,
                        steps_per_epoch=500,
                        epochs=20,
                        validation_data=validate_gen,
                        validation_steps=validate_steps)
    draw_history_loss(history)


def main():
    header, lines = load_data()
    float_data = parse_data(header, lines)
    # draw_time_sequence(float_data)
    normalize_data(float_data)
    train_gen, validate_gen, test_gen, validate_steps, test_steps = prepare_generator(float_data)
    # evaluate_naive_method(validate_gen, validate_step_num)
    # evaluate_simple_model(float_data, train_gen, validate_gen, validate_steps)
    # fit_gru_model(float_data, train_gen, validate_gen, validate_steps)
    # fit_gru_dropout_model(float_data, train_gen, validate_gen, validate_steps)
    # fit_bidirectional_gru(float_data, train_gen, validate_gen, validate_steps)
    # fit_conv_model(float_data, train_gen, validate_gen, validate_steps)
    fit_conv_gru_model(float_data, train_gen, validate_gen, validate_steps)


if __name__ == '__main__':
    main()
