# dcm2seg_and_viewer
Single-click convert murine t2w and FLAIR images to edema volume segmentations and view them in-browser.

# Installation
clone the repo with
`conda install pip git`
`pip install git+https://github.com/imafilippov/dcm2seg_and_viewer`

# Use
This is a Flask webapp relying on Papaya (in JS) for viewing NifTI files in-browser
Run app.py.
Consider changing the hostname to localhost if you want to run only on your machine, or leave as 0.0.0.0 to be accessible to other devices on the network (such as a lab)

MRI images are often stored based on their seqeunce number, as in, the iterated number since the scanner started keeping count. Enter the sequence number corresponding to an axial T2w image and an axial T2w FLAIR image. The "generate prediction" button will process the two images directly from DICOM, and generate .nii.gz files for both images as well as a segmentation prediction for the edema/T2 hyperintensity volume. It will output a location on-screen for the user to locate to view by clicking on the "File" button within the Papaya viewer.

![flask_screen](https://user-images.githubusercontent.com/62579584/171479817-cef56572-8247-45ab-91b1-30278be7a149.png)
