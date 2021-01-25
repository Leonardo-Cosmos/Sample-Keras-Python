import keras
from keras.layers import Dense, LeakyReLU, Reshape, Conv2D, Conv2DTranspose, Input, Flatten, Dropout
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing import image
import numpy as np
import os

latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = keras.Input(shape=(latent_dim,))

data = Dense(128 * 16 * 16)(generator_input)
data = LeakyReLU()(data)
data = Reshape((16, 16, 128))(data)

data = Conv2D(256, 5, padding='same')(data)
data = LeakyReLU()(data)

data = Conv2DTranspose(256, 4, strides=2, padding='same')(data)
data = LeakyReLU()(data)

data = Conv2D(256, 5, padding='same')(data)
data = LeakyReLU()(data)
data = Conv2D(256, 5, padding='same')(data)
data = LeakyReLU()(data)

data = Conv2D(channels, 7, activation='tanh', padding='same')(data)
# data = Conv2D(channels, 7, activation='sigmoid', padding='same')(data)
generator = Model(generator_input, data)
generator.summary()

discriminator_input = Input(shape=(height, width, channels))
data = Conv2D(128, 3)(discriminator_input)
data = LeakyReLU()(data)
data = Conv2D(128, 4, strides=2)(data)
data = LeakyReLU()(data)
data = Conv2D(128, 4, strides=2)(data)
data = LeakyReLU()(data)
data = Conv2D(128, 4, strides=2)(data)
data = LeakyReLU()(data)
data = Flatten()(data)
data = Dropout(0.4)(data)

data = Dense(1, activation='sigmoid')(data)

discriminator = Model(discriminator_input, data)
discriminator.summary()

discriminator_optimizer = RMSprop(
    lr=0.0008,
    clipvalue=1.0,
    decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer,
                      loss='binary_crossentropy')

discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan_optimizer = RMSprop(lr=0.0004,
                        clipvalue=1.0,
                        decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

(data_train, labels_train), (_, _) = keras.datasets.cifar10.load_data()

data_train = data_train[labels_train.flatten() == 6]

data_train = data_train.reshape(
    (data_train.shape[0],) + (height, width, channels)
).astype('float32') / 255.

iterations = 10000
batch_size = 20
save_dir = 'generative_adversarial_network'

start = 0
for step in range(1, iterations + 1):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    generated_images = generator.predict(random_latent_vectors)

    stop = start + batch_size
    real_images = data_train[start:stop]
    combined_images = np.concatenate([generated_images, real_images])

    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape)

    discriminator_loss = discriminator.train_on_batch(combined_images, labels)

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    misleading_targets = np.zeros((batch_size, 1))

    adversarial_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size
    if start > len(data_train) - batch_size:
        start = 0

    if step % 100 == 0:
        gan.save_weights('gan.h5')

        print('step:', step)
        print('discriminator loss:', discriminator_loss)
        print('adversarial loss:', adversarial_loss)

        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, f'real_frog{str(step)}.png'))
