'''
@author: afilippov
@influenced_by: bnsreenu
@date: 2022-05-08
@purpose: to prepare imaging data from murine MRI for segmentation training with UNet
'''

import numpy as np
import nibabel as nib
import glob
import os
import logging
import random
from tensorflow.python.keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave

import mod_utils
log = mod_utils.setup_logging(level=logging.INFO)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# ACT 1: Understanding the data
# Scene 1: Load sample images and visualize
# 1.1.1 normalize images
# describe the rest later

ROOT = os.getcwd()
TRAINING_IMAGES_PATH = os.path.join(ROOT, 'trauma_project_images')

"""
after img = nibable.load(img_path), we can ask:
.shape -> (x, y, z, t)
.get_data_dtypes() -> uint16
.affine.shape -> (4,4)
hdr = img.header
hdr.get_xyzt_units() -> ('mm', 'sec')
raw = hdr.structarr; raw['xyzt_units'] -> array(10, dtype=uint8)
nib.save(img, path_and_name)
"""  # quick tutorial on nib


# test fetching t2w file
test_image_t2w_file = glob.glob(os.path.join(TRAINING_IMAGES_PATH, '.000*', '*t2w.nii.gz'))
test_image_t2w = nib.load(test_image_t2w_file[0]).get_fdata()
log.info(f' globbed file path with * for T2W image is: {test_image_t2w_file}')
log.info(f' test T2W image SHAPE is: {test_image_t2w.shape}')
log.info(f' test T2W image MAX is: {test_image_t2w.max()}')
test_image_t2w = \
    scaler.fit_transform(test_image_t2w.reshape(-1, test_image_t2w.shape[-1])).reshape(test_image_t2w.shape)
log.info(f' test T2W image SHAPE after operations is: {test_image_t2w.shape}')
log.info(f' test T2W image MAX after operations is: {test_image_t2w.max()}')

# test fetching and reading flair file
test_image_flair_file = glob.glob(os.path.join(TRAINING_IMAGES_PATH, '.000*', '*flair.nii.gz'))
test_image_flair = nib.load(test_image_flair_file[0]).get_fdata()
log.info(f' globbed file path with * for FLAIR image is: {test_image_flair_file}')
log.info(f' test FLAIR image SHAPE is: {test_image_flair.shape}')
log.info(f' test FLAIR image MAX is: {test_image_flair.max()}')
test_image_flair = \
    scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)
log.info(f' test FLAIR image SHAPE after operations is: {test_image_flair.shape}')
log.info(f' test FLAIR image MAX after operations is: {test_image_flair.max()}')

# test fetching and reading segmentation file
test_mask_file = glob.glob(os.path.join(TRAINING_IMAGES_PATH, '.000*', '*seg*'))
test_mask = nib.load(test_mask_file[0]).get_fdata().astype(np.uint8)
log.info(f' globbed file path with * for MASK is: {test_mask_file}')
log.info(f' test MASK image SHAPE is: {test_mask.shape}')
log.info(f' test MASK image labels include: {np.unique(test_mask)}')

# view a slice of t2w, flair, and mask
if log.root.level == 20:
    n_slice = round((test_mask.shape[2])/2)  # should be 8 or 9
    log.info(f' the center slice has been selected. it is: {n_slice}')

    plt.figure(figsize=(8, 8))

    plt.subplot(221)
    plt.imshow(test_image_t2w[:, :, n_slice], cmap='gray')
    plt.title('Image T2 weighted')
    plt.subplot(222)
    plt.imshow(test_image_flair[:, :, n_slice], cmap='gray')
    plt.title('Image FLAIR')
    plt.subplot(223)
    plt.imshow(test_mask[:, :, n_slice])
    plt.title('Segmentation')
    plt.subplot(224)
    plt.imshow(test_mask[:, :, n_slice+1])
    plt.show()

# ACT 2: combining images to channels and divide them to patches
# includes combining both images to 2 channel numpy array
# TODO include a way to register using FSL-FLIRT, sMRIPrep or something


# combine t2w and FLAIR
# crop or pad to a size divisible by 64 to extract 64x64(x8?) patches.
# crop strategy is to make a square 64 in every direction from the center.
# see if we lose parts of the brain
# get the coordinates, halve them to find the center
# crop the array to center +/- 64 for a 128x128x16x2 array. view it to see if brain is lost.

combined_image = np.stack([test_image_t2w, test_image_flair], axis=3)
x, y, z = test_image_t2w.shape  # should 238, 256, 16
h_x, h_y, h_z = round(x/2), round(y/2), round(z/2)  # should be 119, 128, 8

combined_image = combined_image[(h_x-64):(h_x+64), (h_y-64-32-20):(h_y+32-20)]
test_mask_crop = test_mask[(h_x-64):(h_x+64), (h_y-64-32-20):(h_y+32-20)]

n_slice = round((test_mask.shape[2])/2)  # should be 8

# view the images we've made and cropped. theres a funny thins where the head is sideways.
if log.root.level == 20:
    log.info(f' the center slice has been selected. it is: {n_slice}')

    plt.figure(figsize=(8, 12))

    plt.subplot(321)
    plt.imshow(combined_image[:, :, n_slice, 0], cmap='gray')
    plt.title('Cropped T2 weighted Image')
    plt.subplot(322)
    plt.imshow(combined_image[:, :, n_slice-1, 0], cmap='gray')

    plt.subplot(323)
    plt.imshow(combined_image[:, :, n_slice, 1], cmap='gray')
    plt.title('Cropped FLAIR Image')
    plt.subplot(324)
    plt.imshow(combined_image[:, :, n_slice-1, 1], cmap='gray')

    plt.subplot(325)
    plt.imshow(test_mask_crop[:, :, n_slice])
    plt.title('Cropped Segmentation')
    plt.subplot(326)
    plt.imshow(test_mask_crop[:, :, n_slice-1])
    plt.show()

#####
# ACT 3: for realsies
# must get data organized in a way for the AI.
# must change the way that dcm2nii names files to include 't2' and 'flair'

t2_list = sorted(glob.glob(os.path.join(TRAINING_IMAGES_PATH, '*', '*t2w.nii.gz')))
flair_list = sorted(glob.glob(os.path.join(TRAINING_IMAGES_PATH, '*', '*flair.nii.gz')))
mask_list = sorted(glob.glob(os.path.join(TRAINING_IMAGES_PATH, '*', '*seg*.nii.gz')))

for img in range(len(t2_list)):
    print('now preparing image and mask number: ', img)

    temp_image_t2 = nib.load(t2_list[img]).get_fdata()
    temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(
        temp_image_t2.shape)

    temp_image_flair = nib.load(flair_list[img]).get_fdata()
    temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(
        temp_image_flair.shape)

    temp_mask = nib.load(mask_list[img]).get_fdata()
    temp_mask = temp_mask.astype(np.uint8)
    # print(np.unique(temp_mask))

    temp_combined_images = np.stack([temp_image_flair, temp_image_t2], axis=3)

    # Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
    # cropping x, y, and z
    x, y, z = temp_image_t2.shape  # should 238, 256, 16
    h_x, h_y, h_z = round(x / 2), round(y / 2), round(z / 2)  # should be 119, 128, 8

    # change dims to 128, 128, 8
    temp_combined_images = temp_combined_images[
                           (h_x - 64):(h_x + 64),
                           (h_y - 64 - 32 - 20):(h_y + 32 - 20),
                           (h_z - 4):(h_z + 4)]
    temp_mask = temp_mask[
                (h_x - 64):(h_x + 64),
                (h_y - 64 - 32 - 20):(h_y + 32 - 20),
                (h_z - 4):(h_z + 4)]
    temp_mask[temp_mask == 2] = 0
    temp_mask[temp_mask == 3] = 0
    n_slice = round((test_mask.shape[2]) / 2)

    val, counts = np.unique(temp_mask, return_counts=True)  # this is good for counting the volumes of the segmentaoins without itk
    print(val, counts)

    save_loc = os.path.join(ROOT, 'storage', 'input_data_3channels')
    mod_utils.maybe_mkdir(os.path.join(ROOT, 'storage'), save_loc.split(os.sep)[-1])
    mod_utils.maybe_mkdir(save_loc, 'images')
    mod_utils.maybe_mkdir(save_loc, 'masks')

    # if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume with labels that are not 0
    if counts[0] > 500:  # At least 1% useful volume with labels that are not 0
        print("Save Me")
        temp_mask = to_categorical(temp_mask, num_classes=2)

        np.save(os.path.join(save_loc, 'images', f'image_{img}.npy'), temp_combined_images)
        np.save(os.path.join(save_loc, 'masks', f'mask_{img}.npy'), temp_mask)

    else:
        print("I am useless")


import splitfolders
input_folder = save_loc
output_folder = os.path.join(ROOT, 'storage', 'input_data_128')
mod_utils.maybe_mkdir(os.path.join(ROOT, 'storage'), output_folder.split(os.sep)[-1])
mod_utils.maybe_mkdir(output_folder, 'train')
mod_utils.maybe_mkdir(output_folder, 'val')
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.70, .30), group_prefix=None)


print('this is a placeholder for debugger.')

"""
test_image_t2w_file = glob.glob(os.path.join(TRAINING_IMAGES_PATH, '000*', '0000*27*.nii'))
test_image_t2w = nib.load(test_image_t2w_file[0]).get_fdata()
test_image_t2w = \
    scaler.fit_transform(test_image_t2w.reshape(-1, test_image_t2w.shape[-1])).reshape(test_image_t2w.shape)

test_image_flair_file = glob.glob(os.path.join(TRAINING_IMAGES_PATH, '000*', '0000*ir*.nii'))
test_image_flair = nib.load(test_image_flair_file[0]).get_fdata()
test_image_flair = \
    scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)

test_image_mask_file = glob.glob(os.path.join(TRAINING_IMAGES_PATH, '000*', '*seg*'))
test_image_mask = nib.load(test_image_mask_file[0]).get_fdata().astype(np.uint8)

"""
