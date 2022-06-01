'''
@author: afilippov
@influenced_by: bnsreenu
@date: 2022-05-08
@purpose: making batches
'''


# from tifffile import imsave, imread
import os
import numpy as np


def load_img(img_dir, img_list):
    images = []
    for i, image_name in enumerate(img_list):
        if (image_name.split('.')[1] == 'npy'):
            image = np.load(img_dir + image_name)

            images.append(image)
    images = np.array(images)

    return (images)


def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)

    # keras needs the generator infinite, so we will use while true
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)

            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield (X, Y)  # a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size


############################################

# Test the generator

from matplotlib import pyplot as plt
import random

ROOT = os.getcwd()
SPLIT_ROOT = os.path.join(ROOT, 'storage', 'input_data_128')

train_img_dir = os.path.join(SPLIT_ROOT, 'train', f'images{os.sep}')
train_mask_dir = os.path.join(SPLIT_ROOT, 'train', f'masks{os.sep}')
train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

batch_size = 2

train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size)


img, msk = train_img_datagen.__next__()

img_num = random.randint(0, img.shape[0] - 1)
test_img = img[img_num]
test_mask = msk[img_num]
test_mask = np.argmax(test_mask, axis=3)

n_slice = random.randint(0, test_mask.shape[2] - 1)
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
plt.title('FLAIR image')

plt.subplot(222)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.title('T2w image')

plt.subplot(223)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask_datagen')
plt.show()
