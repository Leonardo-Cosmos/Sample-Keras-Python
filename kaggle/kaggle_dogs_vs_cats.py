from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from kaggle.kaggle_dogs_vs_cats_copy_data import train_dir
from kaggle.kaggle_dogs_vs_cats_copy_data import validation_dir
import matplotlib.pyplot as plt


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    return model


def train_validate(model):
    train_data_gen = image.ImageDataGenerator(rescale=1. / 255)
    validation_data_gen = image.ImageDataGenerator(rescale=1. / 255)

    train_data_iterator = train_data_gen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,          # train_files / steps_per_epoch
        class_mode='binary')

    validation_data_iterator = validation_data_gen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,          # validation_files / validation_steps
        class_mode='binary')

    history = model.fit(
        train_data_iterator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_data_iterator,
        validation_steps=50)

    model.save('dogs_vs_cats_small_1.h5')

    return history


def build_model_with_dropout():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    return model


def train_with_augment_validate(model):
    train_data_gen = image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    validation_data_gen = image.ImageDataGenerator(rescale=1. / 255)

    train_data_iterator = train_data_gen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    validation_data_iterator = validation_data_gen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    history = model.fit(
        train_data_iterator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_data_iterator,
        validation_steps=50)

    model.save('dogs_vs_cats_small_2.h5')

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
    # model = build_model()
    model = build_model_with_dropout()
    # history = train_validate(model)
    history = train_with_augment_validate(model)
    draw_history(history)


if __name__ == '__main__':
    main()
