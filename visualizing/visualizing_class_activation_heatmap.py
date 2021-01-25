import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
from keras.preprocessing import image



def get_image():
    img_path = 'creative_commons_elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = preprocess_input(img_tensor)
    return img_tensor


def build_heat_map():
    # k.clear_session()
    model = VGG16(weights='imagenet')

    img_tensor = get_image()
    preds = model.predict(img_tensor)
    print('Predicted: ', decode_predictions(preds, top=3)[0])
    print(np.argmax(preds[0]))

    african_elephant_output = model.output[:, 386]
    last_conv_layer = model.get_layer('block5_conv3')

    # tf.compat.v1.disable_eager_execution()
    grads = k.gradients(african_elephant_output, last_conv_layer.output)[0]
    pooled_grads = k.mean(grads, axis=(0, 1, 2))

    iterate = k.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate(img_tensor)

    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)


def build_heat_map_gradient_tape():
    model = VGG16(weights='imagenet')
    img_tensor = get_image()

    last_conv_layer = model.get_layer('block5_conv3')
    heatmap_model = Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = heatmap_model(img_tensor)
        loss = predictions[:, np.argmax(predictions[0])]
        grads = tape.gradient(loss, conv_output)
        pooled_grads = k.mean(grads, axis=(0,1,2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    heatmap = heatmap[0]
    plt.matshow(heatmap)
    plt.show()


def main():
    # build_heat_map()
    build_heat_map_gradient_tape()


if __name__ == '__main__':
    main()
