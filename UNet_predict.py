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


ROOT = os.getcwd()
SPLIT_ROOT = os.path.join(ROOT, 'storage', 'input_data_128')



def prediction(img_path=os.path.join(SPLIT_ROOT, 'train', 'images', f'image_20.npy'),
               n_slice=4,
               save_plot=False,
               show_plot=False
               ):

    from tensorflow.python.keras.models import load_model
    my_model = load_model('brats_3d_1000epochs_1batchsize.hdf5',
                          compile=False)

    d = dict()

    d['img_num'] = 7
    d['img_path'] = img_path
    d['img_path'] = os.path.join(SPLIT_ROOT, 'val', 'images', f'image_{d["img_num"]}.npy')
    d['test_img'] = np.load(d['img_path'])

    d['mask_path'] = os.path.join(SPLIT_ROOT, 'val', 'masks', f'mask_{d["img_num"]}.npy')
    d['test_mask'] = np.load(d['mask_path'])
    d['test_mask_argmax'] = np.argmax(d['test_mask'], axis=3)

    d['test_img_input'] = np.expand_dims(d['test_img'], axis=0)
    d['test_prediction'] = my_model.predict(d['test_img_input'])



    mod_utils.maybe_mkdir(SPLIT_ROOT, 'preds')
    mod_utils.maybe_mkdir(SPLIT_ROOT, 'plots')
    np.save(os.path.join(SPLIT_ROOT, 'preds', f"pred_mask_{d['img_num']}.npy"), d['test_prediction'])

    d['test_prediction_argmax'] = np.argmax(d['test_prediction'], axis=4)[0, :, :, :]

    # print(d['test_prediction_argmax'].shape)
    # print(d['test_mask_argmax'].shape)
    # print(np.unique(d['test_prediction_argmax']))


    # Plot individual slices from test predictions for verification

    import random

    # n_slice=random.randint(0, test_prediction_argmax.shape[2])

    if save_plot:
        from matplotlib import pyplot as plt
        n_slice = n_slice

        plt.figure(figsize=(12, 12))
        plt.subplot(331)
        plt.title('Testing Image')
        plt.imshow(d['test_img'][:, :, n_slice-1, 1], cmap='gray')
        plt.subplot(332)
        plt.title('Testing Label')
        plt.imshow(d['test_mask_argmax'][:, :, n_slice-1])
        plt.subplot(333)
        plt.title('Prediction on test image')
        plt.imshow(d['test_prediction_argmax'][:, :, n_slice-1])

        plt.subplot(334)
        plt.title('Testing Image')
        plt.imshow(d['test_img'][:, :, n_slice, 1], cmap='gray')
        plt.subplot(335)
        plt.title('Testing Label')
        plt.imshow(d['test_mask_argmax'][:, :, n_slice])
        plt.subplot(336)
        plt.title('Prediction on test image')
        plt.imshow(d['test_prediction_argmax'][:, :, n_slice])

        plt.subplot(337)
        plt.title('Testing Image')
        plt.imshow(d['test_img'][:, :, n_slice+1, 1], cmap='gray')
        plt.subplot(338)
        plt.title('Testing Label')
        plt.imshow(d['test_mask_argmax'][:, :, n_slice+1])
        plt.subplot(339)
        plt.title('Prediction on test image')
        plt.imshow(d['test_prediction_argmax'][:, :, n_slice+1])


        plt.savefig(os.path.join(ROOT, 'storage', 'plots', f"plot_{d['img_num']}_{n_slice}.png"))
        if show_plot:
            plt.show()
    return d


if __name__ == '__main__':
    d = prediction()
    print('outside of func: ', d['test_prediction_argmax'].shape)
    print(d['test_mask_argmax'].shape)
    print(np.unique(d['test_prediction_argmax']))
