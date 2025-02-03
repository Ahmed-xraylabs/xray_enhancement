from flask import Flask, render_template, request, url_for
import os
import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 10MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'tif', 'tiff'}

def process_image(image_path):
    # 1) Read image as 16-bit if possible
    original = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError("Invalid image file")

    # 2) Convert to float32
    img_float = original.astype(np.float32)

    # 3) Percentile Clipping
    p_low, p_high = np.percentile(img_float, [0, 100])
    clipped = np.clip(img_float, p_low, p_high)
    norm = (clipped - p_low) / (p_high - p_low)
    norm = np.clip(norm, 0, 1)

    # 4) CLAHE for local contrast
    clahe_in = (norm * 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(10, 10))
    clahe_out = clahe.apply(clahe_in)
    clahe_float = clahe_out.astype(np.float32) / 65535.0

    # 5) Slightly brighten overall with gamma
    gamma_corrected = exposure.adjust_gamma(clahe_float, gamma=0.6)

    # 6) Gentle unsharp mask
    blurred = cv2.GaussianBlur(gamma_corrected, (0, 0),2.0)
    unsharp = cv2.addWeighted(gamma_corrected, 3, blurred, -2, 0)
    unsharp = np.clip(unsharp, 0, 1)

    # 7) Difference of Gaussians for more subtle edge emphasis (rather than harsh outlines)
    dog_blur_small = cv2.GaussianBlur(unsharp, (0, 0), 0.5)
    dog_blur_large = cv2.GaussianBlur(unsharp, (0, 0), 2.5)
    dog = dog_blur_small - dog_blur_large
    # Increase alpha if you want more pronounced edges
    alpha = 3
    dog_enhanced = np.clip(unsharp + alpha * dog, 0, 1)

    # 8) Final rescale to 8-bit
    enhanced = exposure.rescale_intensity(dog_enhanced, out_range=(0, 255)).astype(np.uint8)

    return enhanced, original

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file selected")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        if not allowed_file(file.filename):
            return render_template('index.html', error="Only TIFF files allowed")

        try:
            file_id = uuid.uuid4().hex
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.tif')
            result_id = uuid.uuid4().hex

            file.save(upload_path)
            enhanced_img, original_img = process_image(upload_path)

            filenames = {
                'original': f'original_{result_id}.png',
                'enhanced': f'enhanced_{result_id}.png',
                'histogram': f'histogram_{result_id}.png'
            }

            # Preview of the original
            p99_5 = np.percentile(original_img, 99.5)
            original_preview = np.clip(original_img.astype(np.float32) / p99_5, 0, 1)
            original_preview = (original_preview * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], filenames['original']), original_preview)

            # Enhanced image
            cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], filenames['enhanced']), enhanced_img)

            # Histogram
            plt.figure()
            plt.hist(enhanced_img.ravel(), bins=256, range=(0, 255))
            plt.title('Enhanced Image Histogram')
            plt.savefig(os.path.join(app.config['RESULT_FOLDER'], filenames['histogram']))
            plt.close()

            os.remove(upload_path)
            return render_template('index.html', filenames=filenames)
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
