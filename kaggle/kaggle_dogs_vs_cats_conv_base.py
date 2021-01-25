from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from kaggle.kaggle_dogs_vs_cats_copy_data import train_dir
from kaggle.kaggle_dogs_vs_cats_copy_data import validation_dir
from kaggle.kaggle_dogs_vs_cats_copy_data import test_dir
import numpy as np
import matplotlib.pyplot as plt


def build_conv_base():
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))

    print(conv_base.summary())
    return conv_base


def extract_features(directory, sample_count, data_gen, batch_size, conv_base):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=sample_count)
    data_iterator = data_gen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )
    i = 0
    for inputs_batch, labels_batch in data_iterator:
        features_batch = conv_base.normalize_data(inputs_batch)
        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


def extract_data_features(conv_base):
    data_gen = ImageDataGenerator(rescale=1. / 255)
    batch_size = 20

    train_features, train_labels = extract_features(train_dir, 2000,
                                                    data_gen, batch_size, conv_base)
    validation_features, validation_labels = extract_features(validation_dir, 1000,
                                                              data_gen, batch_size, conv_base)
    test_features, test_labels = extract_features(test_dir, 1000,
                                                  data_gen, batch_size, conv_base)

    train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
    test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

    return ((train_features, train_labels),
            (validation_features, validation_labels),
            (test_features, test_labels))


def train_validate_with_extracted_features(train_data, validation_data):
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    (train_features, train_labels) = train_data
    (validation_features, validation_labels) = validation_data
    history = model.fit(train_features, train_labels,
                        epochs=30,
                        batch_size=20,
                        validation_data=(validation_features, validation_labels))

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


def train_validate_with_model_on_conv_base(conv_base, fine_tune=False):
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    print(model.summary())

    if not fine_tune:
        print('trainable weights before freezing the conv base: ', len(model.trainable_weights))
        conv_base.trainable = False
        print('trainable weights after freezing the conv base: ', len(model.trainable_weights))
    else:
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    train_data_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    validation_data_gen = ImageDataGenerator(rescale=1. / 255)

    train_data_iterator = train_data_gen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    validation_data_iterator = validation_data_gen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    history = model.fit(
        train_data_iterator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_data_iterator,
        validation_steps=50)

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
    conv_base = build_conv_base()
    # train_data, validation_data, test_data = extract_data_features(conv_base)
    # train_validate_with_extracted_features(train_data, validation_data)
    train_validate_with_model_on_conv_base(conv_base)


if __name__ == '__main__':
    main()
