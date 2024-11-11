# from flask import Flask, request, jsonify, render_template
# import os
# import gzip
# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt
# from io import BytesIO
# import base64

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/team')
# def team():
#     return render_template('team.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     files = request.files.getlist('files')
#     images = []

#     for file in files:
#         if file.filename.endswith('.gz'):
#             with gzip.open(file, 'rb') as f_in:
#                 nii_file = nib.Nifti1Image.from_bytes(f_in.read())
#                 data = nii_file.get_fdata()
#                 png_image = convert_nii_to_png(data)
#                 images.append(png_image)

#     return jsonify({'images': images})

# def convert_nii_to_png(data):
#     # Assuming the data is 3D, take the middle slice for visualization
#     slice_2d = data[:, :, data.shape[2] // 2]
#     plt.imshow(slice_2d, cmap='gray')
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     img_base64 = base64.b64encode(buf.read()).decode('utf-8')
#     plt.close()
#     return img_base64

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import os
import gzip
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

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

def convert_nii_to_png(data):
    # Assuming the data is 3D, take the middle slice for visualization
    slice_2d = data[:, :, data.shape[2] // 2]
    plt.imshow(slice_2d, cmap='gray')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

if __name__ == '__main__':
    # Get port from environment variable or default to 10000
    port = int(os.environ.get('PORT', 10000))
    # Bind to 0.0.0.0 to make the server publicly accessible
    app.run(host='0.0.0.0', port=port, debug=False)