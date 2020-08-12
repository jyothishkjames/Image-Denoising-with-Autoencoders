from train_model import *


def main():

    print("Loading data to test...")

    test_data = transform_data("../data/dataset/image_patch/Test/")
    test_data_noise = transform_data("../data/dataset/image_patch_noise/Test/")

    print("Loading trained model weights...")

    input_img = Input(shape=(256, 256, 1))

    output = denoise_model(input_img)

    auto_encoder = Model(input_img, output)

    auto_encoder.load_weights("../models/model.h5")

    auto_encoder.compile(optimizer='Adam', loss='mse')

    test_data_denoised = auto_encoder.predict(test_data_noise)

    # Noisy Image
    plt.imshow(test_data_noise[0].reshape(256, 256), cmap='gray')

    # Denoised Image
    plt.imshow(test_data_denoised[0].reshape(256, 256), cmap='gray')

    # Orginal Image
    plt.imshow(test_data[0].reshape(256, 256), cmap='gray')


if __name__ == '__main__':
    main()
