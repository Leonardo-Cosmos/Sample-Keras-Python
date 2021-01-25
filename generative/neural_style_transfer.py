from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications import vgg19
from keras import backend as K

target_image_path = ''
style_reference_image_path = ''


def preprocess_image(image_path, img_height, img_width):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(data):
    data[:, :, 0] += 103.939
    data[:, :, 1] += 116.779
    data[:, :, 2] += 123.68
    data = data[:, :, ::-1]
    data = np.clip(data, 0, 255).astype('unit8')
    return data


def build_model(img_height, img_width):
    target_image = K.constant(preprocess_image(target_image_path, img_height, img_width))
    style_reference_image = K.constant(preprocess_image(style_reference_image_path, img_height, img_width))
    combination_image = K.placeholder((1, img_height, img_width, 3))
    input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)

    model = vgg19.VGG19(input_tensor=input_tensor,
                        weights='imagenet',
                        include_top=False)
    print('Model loaded.')


def content_loss(base, combination):
    return K.sum(K.square(combination - base))


def gram_matrix(data):
    features = K.batch_flatten(K.permute_dimensions(data, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination, img_height, img_width):
    style_matrix = gram_matrix(style)
    combination_matrix = gram_matrix(combination)
    channels = 3
    size = img_height, img_width
    return K.sum(K.square(style_matrix - combination_matrix)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(data, img_height, img_width):
    a = K.square(
        data[:, :img_height - 1, : img_width - 1, :] -
        data[:, 1:, :img_width - 1, :]
    )

    b = K.square(
        data[:, :img_height - 1, :img_width - 1, :] -
        data[:, :img_height - 1, 1:, :]
    )
    return K.sum(K.pow(a + b, 1.25))


def main():
    width, height = load_img(target_image_path).size
    img_height = 400
    img_width = int(width * img_height / height)
    build_model(img_height, img_width)


if __name__ == '__main__':
    main()
