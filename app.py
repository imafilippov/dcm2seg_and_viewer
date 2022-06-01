# https://youtu.be/bluclMxiUkA
# s/o to dsreeni above
# also a good site. need to follow this and add another one that leaves the image in the same page.
# https://thinkinfi.com/upload-and-display-image-in-flask-python/

from flask import Flask, request, render_template, Response
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import single_predict
import numpy as np

import base64
from io import BytesIO


#Create an app object using the Flask class.
app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')



@app.route('/')
def home():
    return render_template('index.html')

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission.
#Redirect to /predict page with the output


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if request.form['btn_identifier'] == 'gen_pred':
            message = "try this"
            seq_nums = [x for x in request.form.values()]
            d = dict()
            d['t2'] = str(seq_nums[0])
            d['flair'] = str(seq_nums[1])

            nifti_directory, t2_seq_num, flair_seq_num = \
                single_predict.dicom_to_niigz(data_directory=single_predict.DICOMS,
                                              t2_seq_num=d['t2'],
                                              flair_seq_num=d['flair'])

            npy_directory, t2_seq_num, flair_seq_num, nifti_t2, nifti_flair, dims, temp_image_t2 = \
                single_predict.niigz_to_128_npy(nifti_directory, t2_seq_num, flair_seq_num)

            out_dict = single_predict.predict(model_directory=os.path.join(single_predict.ROOT, 'models'),
                                              model_name='unet_3d_1000epochs_1batchsize.hdf5',
                                              npy_directory=npy_directory,
                                              nifti_directory=nifti_directory,
                                              t2_seq_num=t2_seq_num,
                                              flair_seq_num=flair_seq_num,
                                              save_plot=False,
                                              dims=dims,
                                              nifti_t2=nifti_t2)

            return render_template("index.old.html", MESSAGE_1=f"new seg at: {out_dict['new_seg_name']}")  # new segmentation can be found at...
        else:
            pass
    elif request.method == 'GET':
        return 'a get request was made'
    else:
        return 'invalid'


# this is a work in progress to export pngs for powerpoints
# @app.route('/show_slice')
# def show_slice():
#     file_path = [x for x in request.form.values()]
#
#     d = dict()
#     d['img_path'] = file_path[0]
#     d['slice_num'] = file_path[1]
#     d['test_img'] = np.load(d['img_path'])
#
#     fig = Figure()
#     ax = fig.subplots()
#
#     # plt.figure(figsize=(12, 8))
#     # plt.subplot(231)
#     ax.title('Testing Image')
#     ax.plot(d['test_img'][:, :, d['slice_num'], 1], cmap='gray')
#
#     # ax.plot([1, 2])
#     # Save it to a temporary buffer.
#     buf = BytesIO()
#     fig.savefig(buf, format="png")
#     # Embed the result in the html output.
#     data = base64.b64encode(buf.getbuffer()).decode("ascii")
#     return f"<img src='data:image/png;base64,{data}'/>"




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
