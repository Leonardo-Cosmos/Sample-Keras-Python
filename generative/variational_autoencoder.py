import keras
from keras.layers import Layer, Conv2D, Flatten, Dense, Lambda, Input, Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 2

input_img = keras.Input(shape=img_shape)

data = Conv2D(32, 3, padding='same', activation='relu')(input_img)
data = Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(data)
data = Conv2D(64, 3, padding='same', activation='relu')(data)
data = Conv2D(64, 3, padding='same', activation='relu')(data)
shape_before_flattening = K.int_shape(data)

data = Flatten()(data)
data = Dense(32, activation='relu')(data)

latent_mean = Dense(latent_dim)(data)
latent_log_var = Dense(latent_dim)(data)


def sampling(args):
    latent_mean, latent_log_var = args
    epsilon = K.random_normal(shape=(K.shape(latent_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return latent_mean + K.exp(latent_log_var) * epsilon


latent = Lambda(sampling)([latent_mean, latent_log_var])

decoder_input = Input(K.int_shape(latent)[1:])

data = Dense(np.prod(shape_before_flattening[1:]),
             activation='relu')(decoder_input)
data = Reshape(shape_before_flattening[1:])(data)

data = Conv2DTranspose(32, 3,
                       padding='same',
                       activation='relu',
                       strides=(2, 2))(data)

data = Conv2D(1, 3,
              padding='same',
              activation='sigmoid')(data)

decoder = Model(decoder_input, data)

latent_decoded = decoder(latent)


class CustomVariationLayer(Layer):

    def vae_loss(self, data, latent_decoded):
        data = K.flatten(data)
        latent_decoded = K.flatten(latent_decoded)
        xent_loss = keras.metrics.binary_crossentropy(data, latent_decoded)
        kl_loss = -5e-4 * K.mean(1 + latent_log_var - K.square(latent_mean) - K.exp(latent_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        data = inputs[0]
        latent_decoded = inputs[1]
        loss = self.vae_loss(data, latent_decoded)
        self.add_loss(loss, inputs=inputs)
        return data


results = CustomVariationLayer()([input_img, latent_decoded])

vae = Model(input_img, results)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

(data_train, _), (data_test, results_test) = mnist.load_data()

data_train = data_train.astype('float32') / 255.
data_train = data_train.reshape(data_train.shape + (1,))
data_test = data_test.astype('float32') / 255.
data_test = data_test.reshape(data_test.shape + (1,))

vae.fit(data_train, None,
        shuffle=True,
        epochs=10,
        batch_size=batch_size,
        validation_data=(data_test, None))

n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_data = norm.ppf(np.linspace(0.05, 0.95, n))
grid_results = norm.ppf(np.linspace(0.05, 0.95, n))

for i, result in enumerate(grid_data):
    for j, datum in enumerate(grid_results):
        latent_sample = np.array([[datum, result]])
        latent_sample = np.tile(latent_sample, batch_size).reshape(batch_size, 2)
        data_decoded = decoder.predict(latent_sample, batch_size=batch_size)
        digit = data_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
