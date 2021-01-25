from keras.applications import VGG16
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def deprocess_image(img_tensor):
    img_tensor -= img_tensor.mean()
    img_tensor /= (img_tensor.std() + 1e-5)
    img_tensor *= 0.1

    img_tensor += 0.5
    img_tensor = np.clip(img_tensor, 0, 1)

    img_tensor *= 255
    img_tensor = np.clip(img_tensor, 0, 255).astype('uint8')
    return img_tensor


def generate_pattern(model, layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    # loss = tf.reduce_mean(layer_output[:, :, :, filter_index])

    tf.compat.v1.disable_eager_execution()
    grads = K.gradients(loss, model.input)[0]
    # grads = tf.gradients(loss, model.input)[0]

    # grads_square = K.square(grads)
    # grads_mean = K.mean(grads_square)
    # grads_sqrt = K.sqrt(grads_mean) + 1e-5
    # grads /= grads_sqrt

    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # grads /= (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)

    iterate = K.function([model.input], [loss, grads])
    # iterate = tf.function([model.input], [loss, grads])

    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


def main():
    model = VGG16(weights='imagenet',
                  include_top=False)

    plt.imshow(generate_pattern(model, 'block3_conv1', 0))


if __name__ == '__main__':
    main()
