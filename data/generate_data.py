import cv2
import numpy as np
from sklearn.feature_extraction import image
from matplotlib import pyplot as plt


def image_load():
    # read image
    one_image = cv2.imread('../data/dataset/images/image5.jpg')
    # convert to grey scale
    one_image = cv2.cvtColor(one_image, cv2.COLOR_BGR2GRAY)
    # extract image patches
    patches = image.extract_patches_2d(one_image, (256, 256), max_patches=20, random_state=1000)

    return patches


def noisy_image_generate(patches):
    # set mean and standard deviation
    mean = 0.0
    std = 10.0

    # loop through every image patch to add gaussian noise
    for i in range(len(patches)):
        row, col = patches[i].shape
        gauss = np.random.normal(mean, std, (row, col))
        gauss = gauss.reshape(row, col)

        # adding gaussian noise
        noisy = patches[i] + gauss
        noisy = np.clip(noisy, 0, 255)
        # save images
        plt.imsave('../data/dataset/image_patch/' + str(i) + '_image5.png', patches[i], cmap='gray')
        plt.imsave('../data/dataset/image_patch_noise/' + str(i) + '_image5.png', noisy, cmap='gray')


def main():
    patches = image_load()
    print("Loading image...")
    print("Generating image patches...")
    noisy_image_generate(patches)
    print("Generating noisy image patches...")


if __name__ == '__main__':
    main()
