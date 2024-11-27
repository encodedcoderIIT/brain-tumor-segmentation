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


# @app.route('/analyze', methods=['POST'])
# def analyze_files():
#     import tempfile
#     if 'files' not in request.files:
#         return "No file part", 400

#     file = request.files['files']
#     if file.filename == '':
#         return "No selected file", 400

#     if file:
#         # Save the file to a temporary location
#         temp_dir = tempfile.gettempdir()
#         temp_path = os.path.join(temp_dir, file.filename)
#         file.save(temp_path)

#         # Load the file using nibabel
#         img = nib.load(temp_path)

#         # Perform your analysis here
#         # ...

#         return "File analyzed successfully", 200

@app.route('/analyze', methods=['POST'])
def analyze_files():
    print("*****************", request.files)
    import tempfile
    file = request.files.getlist('files')[0]
    print("##################", file)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)

    print(temp_path)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

    # Save the file
    file.save(temp_path)


    img = nib.load(temp_path)
    print("sdafdsfasdf", img)

    # Analyze the images
    try:
        from analysis import analyze
        result_image = analyze(img)
    except Exception as e:
        return f"Error analyzing images: {str(e)}", 500

    return jsonify(result_image)

if __name__ == '__main__':
    app.run(debug=False)