from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images_shape = train_images.shape
print('train_images shape:', train_images_shape)
print('train_labels length:', len(train_labels))

test_images_shape = test_images.shape
print('test_images shape:', test_images_shape)
print('test_labels length:', len(test_labels))

assert len(train_images_shape) == len(test_images_shape)
assert train_images_shape[1] == test_images_shape[1]
assert train_images_shape[2] == test_images_shape[2]

train_images = train_images.reshape((train_images_shape[0], train_images_shape[1] * train_images_shape[2]))
train_images = train_images.astype('float32') / 255
print('train_images shape:', train_images.shape)

test_images = test_images.reshape((test_images_shape[0], test_images_shape[1] * test_images_shape[2]))
test_images = test_images.astype('float32') / 255
print('test_images shape:', test_images.shape)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(train_images_shape[1] * train_images_shape[2],)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
print('test_loss:', test_loss)
