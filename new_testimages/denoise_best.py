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
 
def process_image(image_path, tile_grid=20, clip_limit=3, gamma=0.5, alpha=4,
                 use_median=True, median_kernel=5):
    """
    Process the TIFF image with adjustable parameters.
    - Denoising is applied early in the pipeline
    - Maintains original bit depth during denoising
    """
    # Load image using tifffile for better 32-bit float support
    original = tifffile.imread(image_path)
    if original is None:
        raise ValueError("Could not read image using tifffile.")
 
    # Determine original dtype and convert to float32 for processing if necessary
    original_dtype = original.dtype
    if original_dtype == np.float32:
        img_float = original.copy()
    elif original_dtype == np.uint16:
        # Convert uint16 to float32, preserving scale for potential later use
        # We will work primarily in float32 for calculations
        img_min, img_max = np.min(original), np.max(original)
        img_float = original.astype(np.float32)
    else:
        raise ValueError("Unsupported image dtype: " + str(original_dtype))
 
    # Replace any NaN or Inf values before processing
    img_float = np.nan_to_num(img_float)
    print(f"Image loaded. Shape: {img_float.shape}, Type: {img_float.dtype}")
 
    # Denoising Stage: Median Filter
    if use_median:
        print(f"Applying Median Filter with kernel size: {median_kernel}")
        # Ensure kernel size is odd
        if median_kernel % 2 == 0:
            median_kernel += 1 # Adjust to next odd number
            print(f"Adjusted median kernel size to {median_kernel}")
 
        # Scale to uint16 for median filter
        vmin, vmax = np.percentile(img_float[np.isfinite(img_float)], [0,100 ]) # Use percentiles for robustness
        if vmax <= vmin: vmin, vmax = np.min(img_float), np.max(img_float) # Fallback
        if vmax <= vmin: vmin, vmax = 0, 1 # Final fallback
 
        # Scale to uint16 for median filter
        scaled_u16 = np.clip((img_float - vmin) / (vmax - vmin + 1e-6) * 65535.0, 0, 65535).astype(np.uint16)
 
        # Apply median filter
        median_filtered_u16 = cv2.medianBlur(scaled_u16, median_kernel)
 
        # Scale back to original float range approximate
        img_float = (median_filtered_u16.astype(np.float32) / 65535.0) * (vmax - vmin) + vmin
        print("Median filter applied.")
 
    # Enhancement Stage
    # Normalize for enhancement processing (using percentiles for robustness)
    p_low, p_high = np.percentile(img_float[np.isfinite(img_float)], [0, 100])
    if p_high <= p_low:
        p_low, p_high = np.min(img_float), np.max(img_float) # Fallback
    if p_high <= p_low: # Final fallback if image is constant
         norm = np.zeros_like(img_float, dtype=np.float32)
    else:
        norm = np.clip((img_float - p_low) / (p_high - p_low), 0, 1)
 
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    print("Normalization for enhancement done.")
 
    # Apply CLAHE. Requires uint8 or uint16 input.
    clahe_in = (norm * 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    clahe_out = clahe.apply(clahe_in)
    clahe_float = clahe_out.astype(np.float32) / 65535.0
    print("CLAHE applied.")
 
    # Gamma correction.
    gamma_corrected = exposure.adjust_gamma(clahe_float, gamma=gamma)
    print("Gamma correction applied.")
 
    # Unsharp masking.
    # Adjust blur sigma based on expectation - GaussianBlur needs ksize or sigma
    # Using sigma=2.0 as before. Kernel size (0,0) means it's derived from sigma.
    blurred = cv2.GaussianBlur(gamma_corrected, (0, 0), 2.0)
    unsharp = cv2.addWeighted(gamma_corrected, 3, blurred, -2, 0) # alpha=3, beta=-2
    unsharp = np.clip(unsharp, 0, 2)
    print("Unsharp masking applied.")
 
    # Difference of Gaussians (DoG) for edge enhancement.
    # Sigmas 0.5 and 2.5 as before.
    dog_blur_small = cv2.GaussianBlur(unsharp, (0, 0), 0.5)
    dog_blur_large = cv2.GaussianBlur(unsharp, (0, 0), 2.5)
    dog = dog_blur_small - dog_blur_large
    dog_enhanced = np.clip(unsharp + alpha * dog, 0, 1)
    print("Difference of Gaussians applied.")
 
    # Rescale final result to 8-bit for display
    # Use the dog_enhanced result which is the final step in your enhancement chain
    enhanced_display = exposure.rescale_intensity(dog_enhanced, out_range=(0, 255)).astype(np.uint8)
    print("Final scaling to uint8 for display done.")
 
    return enhanced_display, original # Return the 8-bit display image and original loaded image
 
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
 
        file_id = uuid.uuid4().hex
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.tif')
        file.save(upload_path)
 
        # Get enhancement slider values
        tile_grid = int(request.form.get("tile_grid", 20))
        clip_limit = float(request.form.get("clip_limit", 3))
        gamma = float(request.form.get("gamma", 0.5))
        alpha = float(request.form.get("alpha", 4))
 
        # Get Denoising parameters
        use_median = request.form.get("use_median", "off") == "on"
        median_kernel = int(request.form.get("median_kernel", 5))
 
        # Process the image.
        enhanced_img, _ = process_image(
            upload_path, tile_grid=tile_grid, clip_limit=clip_limit, gamma=gamma, alpha=alpha,
            use_median=use_median, median_kernel=median_kernel
        )
 
        # Encode the processed image to PNG and then base64.
        success, buffer = cv2.imencode('.png', enhanced_img)
        if not success:
            return {"error": "Image encoding failed"}, 500
        png_as_text = base64.b64encode(buffer).decode('utf-8')
 
        # Return all parameters to update the UI state
        return {
            "image": png_as_text,
            "file_id": file_id,
            # Enhancement params
            "tile_grid": tile_grid,
            "clip_limit": clip_limit,
            "gamma": gamma,
            "alpha": alpha,
            # Denoising params
            "use_median": use_median,
            "median_kernel": median_kernel,
        }
    except Exception as e:
        app.logger.error(f"Error in /upload_image: {e}", exc_info=True)
        return {"error": str(e)}, 500
 
@app.route('/update', methods=['POST'])
def update():
    try:
        file_id = request.form.get('file_id')
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.tif')
        if not file_id or not os.path.exists(upload_path):
            return {"error": "File not found or invalid file ID"}, 400
 
        # Get enhancement slider values
        tile_grid = int(request.form.get("tile_grid", 20))
        clip_limit = float(request.form.get("clip_limit", 3))
        gamma = float(request.form.get("gamma", 0.5))
        alpha = float(request.form.get("alpha", 4))
 
        # Get Denoising parameters
        use_median = request.form.get("use_median", "off") == "on"
        median_kernel = int(request.form.get("median_kernel", 5))
 
        # Process the image with updated parameters
        enhanced_img, _ = process_image(
            upload_path, tile_grid=tile_grid, clip_limit=clip_limit, gamma=gamma, alpha=alpha,
            use_median=use_median, median_kernel=median_kernel
        )
 
        # Encode the processed image
        success, buffer = cv2.imencode('.png', enhanced_img)
        if not success:
            return {"error": "Image encoding failed"}, 500
        png_as_text = base64.b64encode(buffer).decode('utf-8')
 
        # Only need to return the image for update
        return {"image": png_as_text}
 
    except Exception as e:
        app.logger.error(f"Error in /update: {e}", exc_info=True)
        return {"error": str(e)}, 500
 
@app.route('/')
def index():
    return render_template('index_denoise.html')
 
if __name__ == '__main__':
    app.run(debug=True)
 