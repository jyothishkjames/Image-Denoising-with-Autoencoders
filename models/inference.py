import argparse
import scipy.misc

from train_model import *


def main():
    # Read the command line arguments and store them
    parser = argparse.ArgumentParser()

    parser.add_argument('--file-path-noisy-image', action='store', dest='file_path_noisy_image', help='filepath of the '
                                                                                                      'noisy image '
                                                                                                      'folder',
                        default=False, required=True)

    parser.add_argument('--file-path-save-image', action='store', dest='file_path_save_image', help='filepath to save '
                                                                                                    'denoise image',
                        default=False, required=True)

    results = parser.parse_args()

    print("Loading data to test...")

    test_data_noise = transform_data(results.file_path_noisy_image)

    print("Loading trained model weights...")

    input_img = Input(shape=(256, 256, 1))

    output = denoise_model(input_img)

    auto_encoder = Model(input_img, output)

    auto_encoder.load_weights("../models/model.h5")

    auto_encoder.compile(optimizer='Adam', loss='mse')

    print("Denoising the noisy image...")

    test_data_denoised = auto_encoder.predict(test_data_noise)

    test_data_denoised = test_data_denoised.reshape(256, 256)

    print("Saving denoised image to path " + results.file_path_save_image)

    scipy.misc.imsave(results.file_path_save_image + 'outfile.jpg', test_data_denoised)


if __name__ == '__main__':
    main()
