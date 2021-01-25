from keras.applications import inception_v3
from keras import backend as K
import numpy as np
import scipy.ndimage as ndimage
import imageio

from keras.preprocessing import image

base_image_path = 'mountain.png'


def build_model():
    K.set_learning_phase(0)
    model = inception_v3.InceptionV3(weights='imagenet',
                                     include_top=False)
    layer_contributions = {
        'mixed2': 0.2,
        'mixed3': 3.,
        'mixed4': 2.,
        'mixed5': 1.5,
    }

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    loss = K.variable(0.)
    for layer_name in layer_contributions:
        coeff = layer_contributions[layer_name]
        activation = layer_dict[layer_name].output

        scaling = K.prod(K.cast(K.shape(activation), 'float32'))
        loss = loss + (coeff * K.sum(K.square(activation[:, 2:-2, 2:-2, :])) / scaling)

    return model, loss


def setup_gradient_ascent(model: inception_v3.InceptionV3, loss):
    dream = model.input

    grads = K.gradients(loss, dream)[0]

    grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

    outputs = [loss, grads]
    fetch_loss_and_grads = K.function([dream], outputs)

    def eval_loss_and_grads(data):
        outs = fetch_loss_and_grads([data])
        loss_value = outs[0]
        grad_values = outs[1]
        return loss_value, grad_values

    def gradient_ascent(data, iterations, step, max_loss=None):
        for i in range(iterations):
            loss_value, grad_values = eval_loss_and_grads()
            if max_loss is not None and loss_value > max_loss:
                break;
            print(f'Loss value at {i}: {loss_value}')
            data += step * grad_values
        return data

    return gradient_ascent


def deep_dream(gradient_ascent):
    step = 0.01
    num_octave = 3
    octave_scale = 1.4
    iterations = 20

    max_loss = 10.

    img = preprocess_image(base_image_path)

    original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)

    successive_shapes = successive_shapes[::-1]

    original_img = np.copy(img)
    shrunk_original_img = resize_image(img, successive_shapes[0])

    for shape in successive_shapes:
        print('Processing image shape', shape)
        img = resize_image(img, shape)
        img = gradient_ascent(img,
                              iterations=iterations,
                              step=step,
                              max_loss=max_loss)
        upscaled_shrunk_original_img = resize_image(shrunk_original_img, shape)
        same_size_original = resize_image(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img
        img += lost_detail
        shrunk_original_img = resize_image(original_img, shape)
        save_image(img, file_name=f'dream_at_scale_{str(shape)}.png')

    save_image(img, file_name='final_dream.png')


def resize_image(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return ndimage.zoom(img, factors, order=1)


def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(data):
    if K.image_data_format() == 'channels_first':
        data = data.reshape((3, data.shape[2], data.shape[3]))
        data = data.transpose((1, 2, 0))
    else:
        data = data.reshape((data.shape[1], data.shape[2], 3))

    data /= 2.
    data += 0.5
    data *= 255.
    data = np.clip(data, 0, 255).astype('uint8')
    return data


def save_image(img, file_name):
    pil_img = deprocess_image(np.copy(img))
    imageio.imwrite(file_name, pil_img)


def main():
    model, loss = build_model()
    gradient_ascent = setup_gradient_ascent(model, loss)
    deep_dream(gradient_ascent)


if __name__ == '__main__':
    main()
