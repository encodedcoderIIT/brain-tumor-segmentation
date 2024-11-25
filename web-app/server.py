from flask import Flask, request, jsonify, render_template
import os
import gzip
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
# from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
# model = load_model('static/model/3D-UNet-2018-weights-improvement-01-0.984.hdf5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    images = []

    for file in files:
        if file.filename.endswith('.gz'):
            with gzip.open(file, 'rb') as f_in:
                nii_file = nib.Nifti1Image.from_bytes(f_in.read())
                data = nii_file.get_fdata()
                png_image = convert_nii_to_png(data)
                images.append(png_image)

    return jsonify({'images': images})

@app.route('/analyze', methods=['POST'])
def analyze_files():
    files = request.files.getlist('files')
    results = []

    # for file in files:
    #     if file.filename.endswith('.gz'):
    #         with gzip.open(file, 'rb') as f_in:
    #             nii_file = nib.Nifti1Image.from_bytes(f_in.read())
    #             data = nii_file.get_fdata()
    #             # Perform analysis using the model
    #             prediction = model.predict(np.expand_dims(data, axis=0))
    #             # Convert prediction to a displayable format
    #             result_image = convert_nii_to_png(prediction[0])
    #             results.append(result_image)

    return jsonify({'results': results})

def convert_nii_to_png(data):
    slice_2d = data[:, :, data.shape[2] // 2]
    plt.imshow(slice_2d, cmap='gray')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)