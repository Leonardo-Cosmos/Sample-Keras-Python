from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print('train_data shape: ', train_data.shape)
print('train_targets length: ', len(train_targets))
print('test_data shape: ', test_data.shape)
print('test_targets length: ', len(test_targets))

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])
    return model


def train_validate():
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 150
    all_mae_histories = []

    for i in range(k):
        print('processing fold #', i)
        start_index = i * num_val_samples
        end_index = start_index + num_val_samples
        val_data = train_data[start_index:end_index]
        val_targets = train_targets[start_index:end_index]

        partial_train_data = np.concatenate(
            [train_data[:start_index], train_data[end_index:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:start_index], train_targets[end_index:]],
            axis=0)

        model = build_model()
        history = model.fit(partial_train_data,
                            partial_train_targets,
                            epochs=num_epochs,
                            batch_size=1,
                            validation_data=(val_data, val_targets))
        mae_history = history.history['val_mae']
        all_mae_histories.append(mae_history)

    average_mae_history = [np.mean([mae_history[i] for mae_history in all_mae_histories]) for i in range(num_epochs)]

    def draw_plot(points):
        plt.plot(range(1, len(points) + 1), points)
        plt.xlabel('Epochs')
        plt.ylabel('Validation MAE')
        plt.show()

    # draw_plot(average_mae_history)

    def smooth_curve(points, factor=0.9):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    smoothed_mae_history = smooth_curve(average_mae_history[10:])
    draw_plot(smoothed_mae_history)


# train_validate()


def train_evaluate():
    model = build_model()
    model.fit(train_data,
              train_targets,
              epochs=80,
              batch_size=16)
    loss_mse, mae = model.evaluate(test_data, test_targets)
    print('evaluate mse:', loss_mse)
    print('evaluate mae:', mae)


train_evaluate()
