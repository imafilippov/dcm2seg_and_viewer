"""
@author: Aleksandr Filippov
@inspiration: Sreenivas Bhattiprolu

Code to train batches of cropped T2w and FLAIR images using 3D U-net.
Please get the data ready and define custom data gnerator using the other
files in this directory.
Images are expected to be 128x128x8x2 npy data (2 corresponds to the 2 channels for test_image_flair, test_image_t2)

Masks are expected to be 128x128x8x2 npy data (2 corresponds to the 2 classes / labels: null, edema)
"""

import os
import numpy as np
from custom_datagen import imageLoader
import tensorflow.python.keras
from matplotlib import pyplot as plt
import glob
import random

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

####################################################
ROOT = os.getcwd()
SPLIT_ROOT = os.path.join(ROOT, 'storage', 'input_data_128')

train_img_dir = os.path.join(SPLIT_ROOT, 'train', f'images{os.sep}')
train_mask_dir = os.path.join(SPLIT_ROOT, 'train', f'masks{os.sep}')

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)

val_img_dir = os.path.join(SPLIT_ROOT, 'val', f'images{os.sep}')
val_mask_dir = os.path.join(SPLIT_ROOT, 'val', f'masks{os.sep}')

val_img_list = os.listdir(val_img_dir)
val_msk_list = os.listdir(val_mask_dir)

num_images = len(os.listdir(train_img_dir))

img_num = random.randint(0, num_images - 1)
test_img = np.load(train_img_dir + img_list[img_num])
test_mask = np.load(train_mask_dir + msk_list[img_num])
test_mask = np.argmax(test_mask, axis=3)

n_slice = random.randint(0, test_mask.shape[2] - 1)
plt.figure(figsize=(8, 8))

plt.subplot(221)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.title('Image t2w')
plt.subplot(223)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

#############################################################
# Optional step of finding the distribution of each class and calculating appropriate weights
# Alternatively you can just assign equal weights and see how well the model performs: 0.25, 0.25, 0.25, 0.25

import pandas as pd

columns = ['0', '1']
df = pd.DataFrame(columns=columns)

train_mask_list = sorted(glob.glob(train_mask_dir + f'{os.sep}*.npy'))
for img in range(len(train_mask_list)):
    print(img)
    temp_image = np.load(train_mask_list[img])
    temp_image = np.argmax(temp_image, axis=3)
    val, counts = np.unique(temp_image, return_counts=True)
    zipped = zip(columns, counts)
    conts_dict = dict(zipped)

    df = df.append(conts_dict, ignore_index=True)

label_0 = df['0'].sum()
label_1 = df['1'].sum()

total_labels = label_0 + label_1
n_classes = 2
# Class weights claculation: n_samples / (n_classes * n_samples_for_class)
wt0 = round((total_labels / (n_classes * label_0)), 2)  # round to 2 decimals
wt1 = round((total_labels / (n_classes * label_1)), 2)


# Weights are: 0.26, 22.53, 22.53, 26.21
# wt0, wt1, wt2, wt3 = 0.26, 22.53, 22.53, 26.21
# These weihts can be used for Dice loss

##############################################################
# Define the image generators for training and validation

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)
##################################

########################################################################




batch_size = 1
epoch=1000




train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader(val_img_dir, val_img_list,
                              val_mask_dir, val_mask_list, batch_size)

# Verify generator.... In python 3 next() is renamed as __next__()
img, msk = train_img_datagen.__next__()

img_num = random.randint(0, img.shape[0] - 1)
test_img = img[img_num]
test_mask = msk[img_num]
test_mask = np.argmax(test_mask, axis=3)

n_slice = random.randint(0, test_mask.shape[2]-1)
plt.figure(figsize=(8, 8))

plt.subplot(221)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.title('Image t2')
plt.subplot(223)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

###########################################################################
# Define loss, metrics and optimizer to be used for training
wt0, wt1 = 0.5, 0.5
os.environ["KERAS_BACKEND"] = "tensorflow"
import segmentation_models_3D as sm

dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = tensorflow.keras.optimizers.Adam(LR)
#######################################################################
# Fit the model

steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

from UNet_3D_model import simple_unet_model

model = simple_unet_model(IMG_HEIGHT=128,
                          IMG_WIDTH=128,
                          IMG_DEPTH=8,
                          IMG_CHANNELS=2,
                          num_classes=2)

model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
print(model.summary())

print(model.input_shape)
print(model.output_shape)


history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epoch,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch,
                    )
model_name = f'brats_3d_{epoch}epochs_{batch_size}batchsize.hdf5'
model.save(model_name)
##################################################################


# plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#################################################

from tensorflow.python.keras.models import load_model

#Load model for prediction or continue training

#For continuing training....
# Now, let us add the iou_score function we used during our initial training
my_model = load_model(model_name,
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score': sm.metrics.IOUScore(threshold=0.5)})

# Now all set to continue the training process.
history2 = my_model.fit(train_img_datagen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=1,
                        verbose=1,
                        validation_data=val_img_datagen,
                        validation_steps=val_steps_per_epoch,
                        )
#################################################

# For predictions you do not need to compile the model, so ...
my_model = load_model(model_name,
                      compile=False)

# Verify IoU on a batch of images from the test dataset
# Using built in keras function for IoU
# Only works on TF > 2.0
from tensorflow.python.keras.metrics import MeanIoU

batch_size = 8  # Check IoU for a batch of images
test_img_datagen = imageLoader(val_img_dir, val_img_list,
                               val_mask_dir, val_mask_list, batch_size)

# Verify generator.... In python 3 next() is renamed as __next__()
test_image_batch, test_mask_batch = test_img_datagen.__next__()

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

