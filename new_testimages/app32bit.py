from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tifffile
from skimage import exposure
import matplotlib.pyplot as plt
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Allow only TIFF files."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'tif', 'tiff'}

def convert_32f_to_16u(float_img):
    """
    Convert a 32-bit float image to a 16-bit unsigned image.
    
    This routine computes the minimum and maximum only on the finite values
    (ignoring any inf or NaN) and then scales the entire image (including non-finite
    pixels, which are forced to 0 or 65535) into the range [0, 65535].
    """
    # Extract finite values only.
    finite_vals = float_img[np.isfinite(float_img)]
    if finite_vals.size == 0:
        print("No finite values found in image; using fallback range [0, 1].")
        fmin, fmax = 0, 1
    else:
        fmin = finite_vals.min()
        fmax = finite_vals.max()
    print("Converting 32f -> 16u: finite min =", fmin, "finite max =", fmax)
    
    # Scale the image: map fmin to 0 and fmax to 65535.
    scaled = (float_img - fmin) / (fmax - fmin) * 65535.0
    # Replace any NaN and convert infinities to boundaries.
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=65535, neginf=0.0)
    scaled = np.clip(scaled, 0, 65535)
    return scaled.astype(np.uint16)

def process_image(image_path):
    """
    Process the TIFF image as follows:
      1) Read the image using tifffile (to properly handle 32-bit float images).
      2) If the image is 32-bit float, convert it to 16-bit using the conversion routine.
      3) Convert to float32 for further processing.
      4) Use percentile clipping (here, using the 1st and 99th percentiles) to normalize.
      5) Apply CLAHE, gamma correction, unsharp masking, and a difference of Gaussians.
      6) Rescale the final result to 8-bit.
    """
    # Load image with tifffile for better 32-bit float support.
    original = tifffile.imread(image_path)
    if original is None:
        raise ValueError("Could not read image using tifffile.")
    
    print("Original image dtype:", original.dtype)
    finite_orig = original[np.isfinite(original)]
    if finite_orig.size > 0:
        print("Original image finite min:", finite_orig.min(), "finite max:", finite_orig.max())
    else:
        print("No finite values in the original image!")
    
    # If 32-bit float, convert to 16-bit.
    if original.dtype == np.float32:
        original_16 = convert_32f_to_16u(original)
    elif original.dtype == np.uint16:
        original_16 = original
    else:
        raise ValueError("Unsupported image dtype: " + str(original.dtype))
    
    # For further processing, work in float32.
    img_float = original_16.astype(np.float32)
    print("After conversion to 16-bit (if applicable): min =", img_float.min(), "max =", img_float.max())
    
    # Percentile clipping: compute 1st and 99th percentiles.
    p_low, p_high = np.percentile(img_float, [1, 99])
    print("Percentile clipping values: 1st =", p_low, " 99th =", p_high)
    if p_high <= p_low:
        p_low, p_high = 0, 1
    clipped = np.clip(img_float, p_low, p_high)
    norm = (clipped - p_low) / (p_high - p_low)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    norm = np.clip(norm, 0, 1)
    
    # Apply CLAHE: scale normalized image to 16-bit for CLAHE.
    clahe_in = (norm * 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(20, 20))
    clahe_out = clahe.apply(clahe_in)
    clahe_float = clahe_out.astype(np.float32) / 65535.0
    
    # Gamma correction.
    gamma_corrected = exposure.adjust_gamma(clahe_float, gamma=0.5)
    
    # Unsharp masking.
    blurred = cv2.GaussianBlur(gamma_corrected, (0, 0), 2.0)
    unsharp = cv2.addWeighted(gamma_corrected, 3, blurred, -2, 0)
    unsharp = np.clip(unsharp, 0, 1)
    
    # Difference of Gaussians.
    dog_blur_small = cv2.GaussianBlur(unsharp, (0, 0), 0.5)
    dog_blur_large = cv2.GaussianBlur(unsharp, (0, 0), 2.5)
    dog = dog_blur_small - dog_blur_large
    alpha = 4  # edge enhancement factor
    dog_enhanced = np.clip(unsharp + alpha * dog, 0, 1)
    
    # Rescale the final result to 8-bit.
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
            # Save the uploaded file.
            file_id = uuid.uuid4().hex
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.tif')
            result_id = uuid.uuid4().hex
            file.save(upload_path)
            
            # Process the image.
            enhanced_img, original_img = process_image(upload_path)
            filenames = {
                'original': f'original_{result_id}.png',
                'enhanced': f'enhanced_{result_id}.png',
                'histogram': f'histogram_{result_id}.png'
            }
            
            # Create a preview of the original image.
            finite_preview = original_img[np.isfinite(original_img)]
            if finite_preview.size > 0:
                o_min = finite_preview.min()
                o_max = finite_preview.max()
            else:
                o_min, o_max = 0, 1
            print("Preview conversion: finite original min =", o_min, "finite original max =", o_max)
            preview = (original_img - o_min) / (o_max - o_min)
            preview = np.clip(preview, 0, 1)
            original_preview = (preview * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], filenames['original']), original_preview)
            
            # Save the enhanced image.
            cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], filenames['enhanced']), enhanced_img)
            
            # Save a histogram of the enhanced image.
            plt.figure()
            plt.hist(enhanced_img.ravel(), bins=256, range=(0, 255))
            plt.title('Enhanced Image Histogram')
            plt.savefig(os.path.join(app.config['RESULT_FOLDER'], filenames['histogram']))
            plt.close()
            
            # Optionally remove the uploaded TIFF.
            os.remove(upload_path)
            
            return render_template('index.html', filenames=filenames)
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
