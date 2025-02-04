from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tifffile
from skimage import exposure
import uuid
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Allow only TIFF files."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'tif', 'tiff'}

def convert_32f_to_16u(float_img):
    """
    Convert a 32-bit float image to a 16-bit unsigned image.
    Computes the min/max on finite values then scales the full image.
    """
    finite_vals = float_img[np.isfinite(float_img)]
    if finite_vals.size == 0:
        print("No finite values found in image; using fallback range [0, 1].")
        fmin, fmax = 0, 1
    else:
        fmin = finite_vals.min()
        fmax = finite_vals.max()
    print("Converting 32f -> 16u: finite min =", fmin, "finite max =", fmax)
    
    scaled = (float_img - fmin) / (fmax - fmin) * 65535.0
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=65535, neginf=0.0)
    scaled = np.clip(scaled, 0, 65535)
    return scaled.astype(np.uint16)

def process_image(image_path, tile_grid=20, clip_limit=3, gamma=0.5, alpha=4):
    """
    Process the TIFF image with adjustable parameters.
      - tile_grid: CLAHE tile grid size.
      - clip_limit: CLAHE clip limit.
      - gamma: Gamma correction factor.
      - alpha: Edge enhancement factor for Difference of Gaussians.
    """
    # Load image using tifffile for better 32-bit float support.
    original = tifffile.imread(image_path)
    if original is None:
        raise ValueError("Could not read image using tifffile.")
    
    if original.dtype == np.float32:
        original_16 = convert_32f_to_16u(original)
    elif original.dtype == np.uint16:
        original_16 = original
    else:
        raise ValueError("Unsupported image dtype: " + str(original.dtype))
    
    img_float = original_16.astype(np.float32)
    
    # Percentile clipping.
    p_low, p_high = np.percentile(img_float, [0, 100])
    if p_high <= p_low:
        p_low, p_high = 0, 1
    clipped = np.clip(img_float, p_low, p_high)
    norm = (clipped - p_low) / (p_high - p_low)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    norm = np.clip(norm, 0, 1)
    
    # Apply CLAHE.
    clahe_in = (norm * 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    clahe_out = clahe.apply(clahe_in)
    clahe_float = clahe_out.astype(np.float32) / 65535.0
    
    # Gamma correction.
    gamma_corrected = exposure.adjust_gamma(clahe_float, gamma=gamma)
    
    # Unsharp masking.
    blurred = cv2.GaussianBlur(gamma_corrected, (0, 0), 2.0)
    unsharp = cv2.addWeighted(gamma_corrected, 3, blurred, -2, 0)
    unsharp = np.clip(unsharp, 0, 1)
    
    # Difference of Gaussians.
    dog_blur_small = cv2.GaussianBlur(unsharp, (0, 0), 0.5)
    dog_blur_large = cv2.GaussianBlur(unsharp, (0, 0), 2.5)
    dog = dog_blur_small - dog_blur_large
    dog_enhanced = np.clip(unsharp + alpha * dog, 0, 1)
    
    # Rescale to 8-bit.
    enhanced = exposure.rescale_intensity(dog_enhanced, out_range=(0, 255)).astype(np.uint8)
    
    return enhanced, original

@app.route('/')
def index():
    # Render the main interface. Initially no image is uploaded.
    return render_template('index2.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return {"error": "No file selected"}, 400
        file = request.files['file']
        if file.filename == '':
            return {"error": "No file selected"}, 400
        if not allowed_file(file.filename):
            return {"error": "Only TIFF files allowed"}, 400
        
        # Save the uploaded TIFF.
        file_id = uuid.uuid4().hex
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.tif')
        file.save(upload_path)
        
        # Get slider values (or defaults).
        tile_grid = int(request.form.get("tile_grid", 20))
        clip_limit = float(request.form.get("clip_limit", 3))
        gamma = float(request.form.get("gamma", 0.5))
        alpha = float(request.form.get("alpha", 4))
        
        # Process the image.
        enhanced_img, _ = process_image(upload_path, tile_grid, clip_limit, gamma, alpha)
        
        # Encode the processed image to PNG and then base64.
        success, buffer = cv2.imencode('.png', enhanced_img)
        if not success:
            return {"error": "Image encoding failed"}, 500
        png_as_text = base64.b64encode(buffer).decode('utf-8')
        
        return {"image": png_as_text, "file_id": file_id, "tile_grid": tile_grid, 
                "clip_limit": clip_limit, "gamma": gamma, "alpha": alpha}
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/update', methods=['POST'])
def update():
    try:
        file_id = request.form.get('file_id')
        tile_grid = int(request.form.get("tile_grid", 20))
        clip_limit = float(request.form.get("clip_limit", 3))
        gamma = float(request.form.get("gamma", 0.5))
        alpha = float(request.form.get("alpha", 4))
        
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.tif')
        if not os.path.exists(upload_path):
            return {"error": "File not found"}, 400
        
        enhanced_img, _ = process_image(upload_path, tile_grid, clip_limit, gamma, alpha)
        
        success, buffer = cv2.imencode('.png', enhanced_img)
        if not success:
            return {"error": "Image encoding failed"}, 500
        png_as_text = base64.b64encode(buffer).decode('utf-8')
        
        return {"image": png_as_text}
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)
