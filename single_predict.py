import numpy as np
import logging
import os
import glob
import dicom2nifti
import nibabel as nib
from tensorflow.python.keras.utils.np_utils import to_categorical

import mod_utils
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# ROOT = os.getcwd()
ROOT = '/media/badie_group/flask'

DICOMS = '/media/badie_group/Scanner/'


def dicom_to_niigz(
        data_directory=DICOMS,
        t2_seq_num=7610,
        flair_seq_num=7611,
        output_directory=os.path.join(ROOT, 'storage')):

    for dicom in glob.glob(os.path.join(data_directory, '*', 'DICOM', '*'), recursive=True):
        if str(t2_seq_num) in dicom.split(os.sep)[-1]:
            dicom_t2w = dicom

    for dicom in glob.glob(os.path.join(data_directory, '*', 'DICOM', '*'), recursive=True):
        if str(flair_seq_num) in dicom.split(os.sep)[-1]:
            dicom_flair = dicom

    output = os.path.join(output_directory, 'temp_nifti')
    mod_utils.maybe_mkdir(output_directory, 'temp_nifti')

    dicom2nifti.convert_directory(dicom_t2w, output, compression=True, reorient=True)
    dicom2nifti.convert_directory(dicom_flair, output, compression=True, reorient=True)

    return output, t2_seq_num, flair_seq_num


def niigz_to_128_npy(
        nifti_directory=os.path.join(ROOT, 'storage', 'temp_nifti'),
        t2_seq_num=0000,
        flair_seq_num=0000,
        dims=None):

    if dims is None:
        dims = [128, 128, 8]


    for nifti in glob.glob(os.path.join(nifti_directory, '*')):
        if str(t2_seq_num) in nifti.split(os.sep)[-1]:
            nifti_t2 = nifti

    for nifti in glob.glob(os.path.join(nifti_directory, '*')):
        if str(flair_seq_num) in nifti.split(os.sep)[-1]:
            nifti_flair = nifti


    # nifti_t2 = glob.glob(os.path.join(nifti_directory, f'*{t2_seq_num}*t2w.nii.gz'))
    # nifti_flair = glob.glob(os.path.join(nifti_directory, f'*{flair_seq_num}*flair.nii.gz'))

    temp_image_t2 = nib.load(nifti_t2).get_fdata()
    temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(
        temp_image_t2.shape)

    temp_image_flair = nib.load(nifti_flair).get_fdata()
    temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(
        temp_image_flair.shape)

    temp_combined_images = np.stack([temp_image_flair, temp_image_t2], axis=3)  # shape of this? 238 256 16 2?
    print(temp_combined_images.shape)

    # Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
    # cropping x, y, and z
    x, y, z = temp_image_t2.shape  # should 238, 256, 16

    # find the center of the image/array
    h_x, h_y, h_z = round(x / 2), round(y / 2), round(z / 2)  # should be 119, 128, 8

    # change dims to 128, 128, 8
    temp_combined_images = temp_combined_images[
                           (h_x - int(dims[0]/2)):(h_x + int(dims[0]/2)),
                           (h_y - int(dims[1]/2) - int(dims[1]/4) - 20):(h_y + int(dims[1]/4) - 20),
                           (h_z - int(dims[2]/2)):(h_z + int(dims[2]/2))]

    npy_directory = os.path.join(ROOT, 'storage', f'temp_2channel_{dims[0]}_npy')
    mod_utils.maybe_mkdir(os.path.join(ROOT, 'storage'), npy_directory.split(os.sep)[-1])

    np.save(os.path.join(npy_directory, f't2w_{t2_seq_num}_flair_{flair_seq_num}_cropped{dims[0]}.npy'), temp_combined_images)

    return npy_directory, t2_seq_num, flair_seq_num, nifti_t2, nifti_flair, dims, temp_image_t2


def predict(model_directory=os.path.join(ROOT, 'models'),
            model_name='unet_3d_1000epochs_1batchsize.hdf5',
            npy_directory=os.path.join(ROOT, 'storage', 'temp_2channel_128_npy'),
            nifti_directory=os.path.join(ROOT, 'storage', 'temp_nifti'),
            t2_seq_num=0000,
            flair_seq_num=0000,
            save_plot=False,
            dims=None,
            nifti_t2='sub_ses_seq.nii.gz'
            ):

    from tensorflow.python.keras.models import load_model

    if dims is None:
        dims = [128, 128, 8]

    model_path = os.path.join(model_directory, model_name)
    my_model = load_model(model_path, compile=False)

    d = dict()

    d['t2_seq_num'] = t2_seq_num
    d['flair_seq_num'] = flair_seq_num
    d['npy_path'] = os.path.join(npy_directory, f't2w_{t2_seq_num}_flair_{flair_seq_num}_cropped{dims[0]}.npy')
    d['img'] = np.load(d['npy_path'])

    d['img_input'] = np.expand_dims(d['img'], axis=0)
    d['prediction'] = my_model.predict(d['img_input'])

    print(d['img'].shape)  # should all be the same as dims=[128,128,8, 2]
    print(d['img_input'].shape)  # is [1, 128, 128, 8, 2]
    print(d['prediction'].shape)  # is [1, 128, 128, 8, 2]

    # nifti_t2 = glob.glob(os.path.join(nifti_directory, f'*{t2_seq_num}*t2w.nii.gz'))  # this gives a list, not a pathname
    cropped_name = nifti_t2.replace(nifti_t2.split('_')[-1], 'cropped-unet-seg.npy')

    np.save(os.path.join(npy_directory, cropped_name), d['prediction']) #is prediction or argmax what I want to save?

    d['prediction_argmax'] = np.argmax(d['prediction'], axis=4)[0, :, :, :]

    d['val'], d['counts'] = np.unique(d['prediction_argmax'],
                                      return_counts=True)  # this is good ofr counting the volumes of the segmentaoins without itk

    """
    print(d['prediction'][0, 40:45, 30:35, 4, 0])
    [[9.9974185e-01 2.6024031e-03 2.4728656e-06 4.6003720e-07 4.4734775e-06]
     [9.8387939e-01 5.1414801e-05 9.9034025e-07 3.4085298e-07 3.5454195e-06]
     [9.9850607e-01 3.8475448e-03 4.3962642e-05 2.0214791e-06 7.2905646e-06]
     [9.9987471e-01 9.3986517e-01 2.9116562e-01 1.3004211e-03 1.2649920e-04]
     [9.9997866e-01 9.9948895e-01 9.9928337e-01 9.5469445e-01 7.2793872e-04]]
    
    print(np.round(d['prediction'][0, 40:45, 30:35, 4, 0], 0))
    [[1. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0.]
     [1. 1. 0. 0. 0.]
     [1. 1. 1. 1. 0.]]
    print(d['prediction_argmax'][40:45, 30:35, 4])
    [[0 1 1 1 1]
     [0 1 1 1 1]
     [0 1 1 1 1]
     [0 0 1 1 1]
     [0 0 0 0 1]]
    """


    img = nib.load(nifti_t2)
    temp_image_t2 = nib.load(nifti_t2).get_fdata()
    temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(
        temp_image_t2.shape)

    x, y, z = temp_image_t2.shape  # should 238, 256, 16

    # find the center of the image/array
    h_x, h_y, h_z = round(x / 2), round(y / 2), round(z / 2)  # should be 119, 128, 8

    resized_canvas = np.zeros((x, y, z))
    resized_canvas[
        (h_x - int(dims[0] / 2)):(h_x + int(dims[0] / 2)),
        (h_y - int(dims[1] / 2) - int(dims[1] / 4) - 20):(h_y + int(dims[1] / 4) - 20),
        (h_z - int(dims[2] / 2)):(h_z + int(dims[2] / 2))] = d['prediction_argmax']  # there was a prblem here. sahpes not the same

    resized_name = nifti_t2.replace(nifti_t2.split('_')[-1], 'resized-unet-seg.npy')

    np.save(os.path.join(npy_directory, resized_name), resized_canvas)

    new_image = nib.Nifti1Image(resized_canvas, img.affine, img.header)
    d['new_seg_name'] = nifti_t2.replace(nifti_t2.split('_')[-1], 'unet-seg.nii.gz')
    nib.save(new_image, d['new_seg_name'])

    if save_plot:
        from matplotlib import pyplot as plt

        x, y, z = d['img'].shape  # should 238, 256, 16

        # find the center of the image/array
        h_x, h_y, h_z = round(x / 2), round(y / 2), round(z / 2)  # should be 119, 128, 8

        n_slice = h_z

        plt.figure(figsize=(12, 12))
        plt.subplot(231)
        plt.title('Image')
        plt.imshow(d['img'][:, :, n_slice - 1, 1], cmap='gray')

        plt.subplot(232)
        plt.title('Prediction on test image')
        plt.imshow(d['test_prediction_argmax'][:, :, n_slice - 1])


        plt.subplot(233)
        plt.title('Image')
        plt.imshow(d['img'][:, :, n_slice, 1], cmap='gray')

        plt.subplot(234)
        plt.title('Prediction on test image')
        plt.imshow(d['test_prediction_argmax'][:, :, n_slice])


        plt.subplot(235)
        plt.title('Image')
        plt.imshow(d['img'][:, :, n_slice + 1, 1], cmap='gray')

        plt.subplot(236)
        plt.title('Prediction on test image')
        plt.imshow(d['test_prediction_argmax'][:, :, n_slice + 1])

        plt.savefig(os.path.join(ROOT, 'storage', 'plots', f"plot_{d['img_num']}_{n_slice}.png"))
        # if show_plot:
        #     plt.show()

    return d


if __name__ == '__main__':

    nifti_directory, t2_seq_num, flair_seq_num = dicom_to_niigz()

    npy_directory, t2_seq_num, flair_seq_num, nifti_t2, nifti_flair, dims, temp_image_t2 = \
        niigz_to_128_npy(nifti_directory, t2_seq_num, flair_seq_num)

    out_dict = predict(model_directory=os.path.join(ROOT, 'models'),
                       model_name='unet_3d_1000epochs_1batchsize.hdf5',
                       npy_directory=npy_directory,
                       nifti_directory=nifti_directory,
                       t2_seq_num=t2_seq_num,
                       flair_seq_num=flair_seq_num,
                       save_plot=False,
                       dims=dims,
                       nifti_t2=nifti_t2)

    print('prediction volume is ', out_dict['counts'])


