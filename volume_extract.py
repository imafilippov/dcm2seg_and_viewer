import os
import pandas as pd
import numpy as np
import glob
import nibabel as nib

ROOT = os.getcwd()
TRAINING_IMAGES_PATH = os.path.join(ROOT, 'storage', 'training_images')
mask_list = sorted(glob.glob(os.path.join(TRAINING_IMAGES_PATH, '*', '*seg*.nii.gz')))

df = pd.DataFrame(columns=['file_paths', 'file_names', 'UPN', 'date', 'edema_volume'])
df['file_paths'] = mask_list
# df[['First','Last']] = df.Name.str.split("_",expand=True)

for row, val in df.iterrows():
    print(row, val)
    print(df.loc[row, 'file_paths'])
    df.loc[row, 'file_names'] = df.loc[row, 'file_paths'].split(os.sep)[-1]
    print(df.loc[row, 'file_names'])
    df.loc[row, 'UPN'] = df.loc[row, 'file_names'].split('_')[0][-3:]
    df.loc[row, 'date'] = df.loc[row, 'file_names'].split('_')[1][-10:]

    print('now preparing image and mask number: ', df.loc[row, 'file_names'])

    temp_mask = nib.load(df.loc[row, 'file_paths']).get_fdata()
    temp_mask = temp_mask.astype(np.uint8)
    print(np.unique(temp_mask))

    temp_mask[temp_mask == 2] = 0
    temp_mask[temp_mask == 3] = 0

    val, counts = np.unique(temp_mask, return_counts=True)
    print(val, counts, (counts[0] + counts[1]))

    df.loc[row, 'edema_volume'] = counts[1]

df.to_csv(os.path.join(ROOT, 'storage', 'edema_volumes.csv'))

