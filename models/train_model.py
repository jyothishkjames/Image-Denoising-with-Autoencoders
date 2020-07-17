import os
import numpy as np
import cv2

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Activation, Conv2D, BatchNormalization, Subtract
from tensorflow.python.keras.initializers import normal
from matplotlib import pyplot as plt


def transform_data(image_dir):
    """Gets numpy data from images that are in the folders."""

    image_files = [image_dir + '{0}'.format(f)
                   for f in os.listdir(image_dir) if not f.startswith('.')]

    num_images = len(image_files)
    images_np_arr = np.empty([num_images, 256, 256, 1], dtype=np.float32)

    for image in range(num_images):
        img = cv2.imread(image_files[image])
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(256, 256, 1)
        img = img.astype(np.float32)
        images_np_arr[image] = img

    data = images_np_arr

    return data


def denoise_model(image):
    """Model that denoises the noisy image."""

    initializer = normal(mean=0, stddev=0.01, seed=13)

    x = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializer)(image)

    bn1 = BatchNormalization()(x)

    act1 = Activation(activation='selu')(bn1)

    x = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializer)(act1)

    bn1 = BatchNormalization()(x)

    act1 = Activation(activation='selu')(bn1)

    encoded = Conv2D(32, (3, 3), padding='same', kernel_initializer=initializer)(act1)

    bn1 = BatchNormalization()(encoded)

    act1 = Activation(activation='selu')(bn1)

    x = Conv2D(32, (3, 3), padding='same', kernel_initializer=initializer)(act1)

    bn1 = BatchNormalization()(x)

    act1 = Activation(activation='selu')(bn1)

    x = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializer)(act1)

    bn1 = BatchNormalization()(x)

    act1 = Activation(activation='selu')(bn1)

    x = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializer)(act1)

    bn1 = BatchNormalization()(x)

    act1 = Activation(activation='selu')(bn1)

    decoded = Conv2D(1, (3, 3), padding='same', kernel_initializer=initializer)(act1)

    decoded = Subtract()([image, decoded])

    return decoded


def main():

    print("Transforming Data")

    train_data = transform_data("../data/dataset/image_patch/Train/")
    test_data = transform_data("../data/dataset/image_patch/Test/")

    train_data_noise = transform_data("../data/dataset/image_patch_noise/Train/")
    test_data_noise = transform_data("../data/dataset/image_patch_noise/Test/")

    print("Training/Running model")

    input_img = Input(shape=(256, 256, 1))

    output = denoise_model(input_img)

    auto_encoder = Model(input_img, output)

    auto_encoder.load_weights("../models/model.h5")

    auto_encoder.compile(optimizer='Adam', loss='mse')

    auto_encoder.fit(train_data_noise, train_data,
                     epochs=0,
                     batch_size=8,
                     shuffle=True,
                     validation_data=(test_data_noise, test_data))

    # autoencoder.save(cwd+"/model_new.h5")

    test_data_denoised = auto_encoder.predict(test_data_noise)

    # Noisy Image
    plt.imshow(test_data_noise[0].reshape(256, 256), cmap='gray')

    # Denoised Image
    plt.imshow(test_data_denoised[0].reshape(256, 256), cmap='gray')


if __name__ == '__main__':
    main()
