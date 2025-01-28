from flask import Flask, render_template, request, url_for
import os
import cv2
import numpy as np
import pywt
from skimage import exposure
import matplotlib.pyplot as plt
import uuid
 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 40 * 1024 * 1024  # 10MB limit
 
# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
 
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'tif', 'tiff'}
 
def process_image(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image file")
    
    # Normalization with adaptive percentile scaling
    p99_5 = np.percentile(img, 99.5)
    img_norm = np.clip(img.astype(np.float32)/p99_5, 0, 1)

    # Wavelet transform with smoother basis
    wavelet = 'sym5'
    coeffs = pywt.wavedec2(img_norm, wavelet, level=3)
    cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

    # Adaptive noise-adaptive thresholding
    def adaptive_threshold(coeff, level):
        sigma = np.std(coeff)
        return pywt.threshold(coeff, value=sigma * (1.2 - 0.2*level), mode='soft')

    # Apply level-dependent thresholding
    cD1 = adaptive_threshold(cD1, 1)
    cD2 = adaptive_threshold(cD2, 2)
    cD3 = adaptive_threshold(cD3, 3)

    # Border-specific enhancement factors
    edge_boost = [1.5, 2.5, 4.0]
    noise_suppression = [0.8, 0.6, 0.4]

    # Multi-scale reconstruction with edge focusing
    coeffs_enhanced = [
        cA3,
        (cH3*edge_boost[0]*noise_suppression[0], 
         cV3*edge_boost[0]*noise_suppression[0], 
         cD3),
        (cH2*edge_boost[1]*noise_suppression[1], 
         cV2*edge_boost[1]*noise_suppression[1], 
         cD2),
        (cH1*edge_boost[2]*noise_suppression[2], 
         cV1*edge_boost[2]*noise_suppression[2], 
         cD1)
    ]

    # Smooth reconstruction
    img_recon = pywt.waverec2(coeffs_enhanced, wavelet)
    img_recon = np.clip(img_recon, 0, 1)

    # Convert to 8-bit for bilateral filtering
    img_recon_8bit = (img_recon * 255).astype(np.uint8)

    # Edge-preserving smoothing with corrected format
    img_smooth = cv2.bilateralFilter(img_recon_8bit, d=9, sigmaColor=75, sigmaSpace=75)
    img_smooth = img_smooth.astype(np.float32) / 255.0

    # Adaptive CLAHE with edge protection
    clahe = cv2.createCLAHE(clipLimit=0.4, tileGridSize=(16, 16))
    img_clahe = clahe.apply((img_smooth * 255).astype(np.uint8))
    img_clahe = img_clahe.astype(np.float32) / 255.0

    # Edge detection and enhancement
    edges = cv2.Canny((img_clahe * 255).astype(np.uint8), 50, 150) / 255.0
    enhanced_edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    
    # Final blending
    img_final = np.clip(img_clahe * 0.8 + enhanced_edges * 0.2, 0, 1)

    # Gentle sharpening
    blurred = cv2.GaussianBlur(img_final, (0, 0), 2.5)
    img_sharp = cv2.addWeighted(img_final, 1.8, blurred, -0.8, 0)
    img_sharp = np.clip(img_sharp, 0, 1)

    # Convert to 8-bit with histogram optimization
    enhanced_img = exposure.rescale_intensity(img_sharp, in_range=(0.05, 0.95), out_range=(0, 255)).astype(np.uint8)

    return enhanced_img, img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check file upload
        if 'file' not in request.files:
            return render_template('index.html', error="No file selected")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        
        if not allowed_file(file.filename):
            return render_template('index.html', error="Only TIFF files allowed")
        
        try:
            # Generate unique IDs
            file_id = uuid.uuid4().hex
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.tif')
            result_id = uuid.uuid4().hex
            
            # Save file
            file.save(upload_path)
            
            # Process image
            enhanced_img, original_img = process_image(upload_path)
            
            # Generate filenames
            filenames = {
                'original': f'original_{result_id}.png',
                'enhanced': f'enhanced_{result_id}.png',
                'histogram': f'histogram_{result_id}.png'
            }
            
            # Save original preview
            percentile_99_5 = np.percentile(original_img, 99.5)
            original_preview = np.clip(original_img.astype(np.float32) / percentile_99_5, 0, 1)
            original_preview = (original_preview * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], filenames['original']), original_preview)
            
            # Save enhanced image
            cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], filenames['enhanced']), enhanced_img)
            
            # Save histogram
            plt.figure()
            plt.hist(enhanced_img.ravel(), bins=256, range=(0, 255))
            plt.title('Enhanced Image Histogram')
            plt.savefig(os.path.join(app.config['RESULT_FOLDER'], filenames['histogram']))
            plt.close()
            
            # Cleanup
            os.remove(upload_path)
            
            return render_template('index.html', filenames=filenames)
        
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')
 
if __name__ == '__main__':
    app.run(debug=True)
 