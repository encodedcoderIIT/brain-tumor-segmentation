import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from flask import Flask, request, jsonify, render_template
import os
import gzip
import base64
from io import BytesIO
import nibabel as nib
import matplotlib.pyplot as plt

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
    def convert_nii_to_png(data):
        slice_2d = data[:, :, data.shape[2] // 2]
        plt.imshow(slice_2d, cmap='gray')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_base64
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
    from analysis import analyze
    files = request.files.getlist('files')
    if len(files) < 2:
        return "Missing files", 400

    try:
        # Load the images using nibabel
        images = [nib.load(file).get_fdata() for file in files]
    except Exception as e:
        return f"Error loading images: {str(e)}", 400

    # Analyze the images
    try:
        result_image = analyze(images)
    except Exception as e:
        return f"Error analyzing images: {str(e)}", 500

    return jsonify(result_image)

if __name__ == '__main__':
    app.run(debug=False)