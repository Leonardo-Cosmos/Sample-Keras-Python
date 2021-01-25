from keras.preprocessing import image
import os, shutil
import matplotlib.pyplot as plt

original_dataset_dir = 'dogs-vs-cats_download/train'

base_dir = 'dogs-vs-cats_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    else:
        print('directory exists already: ', dir_path)


def make_data_dir():
    make_dir(base_dir)
    make_dir(train_dir)
    make_dir(validation_dir)
    make_dir(test_dir)
    make_dir(train_cats_dir)
    make_dir(train_dogs_dir)
    make_dir(validation_cats_dir)
    make_dir(validation_dogs_dir)
    make_dir(test_cats_dir)
    make_dir(test_dogs_dir)


def copy_files(file_names: list, src_dir, dst_dir):
    for file_name in file_names:
        src = os.path.join(src_dir, file_name)
        dst = os.path.join(dst_dir, file_name)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        else:
            print('file exists already: ', dst)


def copy_data_files():
    cat_file_name_format = 'cat.{}.jpg'
    copy_files([cat_file_name_format.format(i) for i in range(1000)], original_dataset_dir, train_cats_dir)
    copy_files([cat_file_name_format.format(i) for i in range(1000, 1500)], original_dataset_dir, validation_cats_dir)
    copy_files([cat_file_name_format.format(i) for i in range(1500, 2000)], original_dataset_dir, test_cats_dir)

    dog_file_name_format = 'dog.{}.jpg'
    copy_files([dog_file_name_format.format(i) for i in range(1000)], original_dataset_dir, train_dogs_dir)
    copy_files([dog_file_name_format.format(i) for i in range(1000, 1500)], original_dataset_dir, validation_dogs_dir)
    copy_files([dog_file_name_format.format(i) for i in range(1500, 2000)], original_dataset_dir, test_dogs_dir)

    print('total training cat images: ', len(os.listdir(train_cats_dir)))
    print('total training dog images: ', len(os.listdir(train_dogs_dir)))
    print('total validation cat images: ', len(os.listdir(validation_cats_dir)))
    print('total validation dog images: ', len(os.listdir(validation_dogs_dir)))
    print('total training cat images: ', len(os.listdir(test_cats_dir)))
    print('total training dog images: ', len(os.listdir(test_dogs_dir)))


def augment_image():
    data_gen = image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    file_paths = [os.path.join(train_cats_dir, file_name) for file_name in os.listdir(train_cats_dir)]

    img_path = file_paths[3]

    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in data_gen.flow(x, batch_size=1):
        plt.figure(i)
        img_plot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break;
    plt.show()


def main():
    make_data_dir()
    copy_data_files()
    # augment_image()


if __name__ == '__main__':
    main()
