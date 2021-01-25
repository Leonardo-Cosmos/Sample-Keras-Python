from keras import models
from keras.preprocessing import image
from kaggle.kaggle_dogs_vs_cats_copy_data import test_cats_dir
import matplotlib.pyplot as plt
import numpy as np
import os


def img_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    print('img_tensor shape: ', img_tensor.shape)

    return img_tensor


def show_channel(activations):
    first_layer_activation = activations[0]
    print('activation shape: ', first_layer_activation.shape)

    plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
    plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
    plt.show()


def show_all_channels(layer_names, activations):
    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_rows = n_features // images_per_row
        display_grid = np.zeros((size * n_rows, images_per_row * size))

        for row in range(n_rows):
            for col in range(images_per_row):
                channel_image = layer_activation[0, :, :, row * images_per_row + col]
                channel_image -= channel_image.mean()
                if not channel_image.std() == 0:
                    channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[row * size:(row + 1) * size, col * size:(col + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()


def main():
    model = models.load_model('dogs_vs_cats_small_2.h5')
    print(model.summary())

    img_path = os.path.join(test_cats_dir, 'cat.1700.jpg')
    img_tensor = img_to_tensor(img_path)

    plt.imshow(img_tensor[0])
    plt.show()

    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    # show_channel(activations)

    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)
    show_all_channels(layer_names, activations)


if __name__ == '__main__':
    main()
