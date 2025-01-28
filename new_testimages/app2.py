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
    
    # Normalization with percentile scaling
    p99 = np.percentile(img, 100)
    img_norm = np.clip(img.astype(np.float32)/p99, 0, 1)

    # Modified wavelet parameters for better edge separation
    wavelet = 'haar'  # Changed to Haar wavelet for sharper edge representation
    coeffs = pywt.wavedec2(img_norm, wavelet, level=3)  # Increased decomposition level
    cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

    # Adaptive thresholding with lower threshold factor
    threshold_factor = 0.2  # Reduced to preserve more edge information
    threshold_cD1 = pywt.threshold(cD1, threshold_factor*np.std(cD1), mode='soft', substitute=np.median(cD1))
    threshold_cD2 = pywt.threshold(cD2, threshold_factor*np.std(cD2), mode='soft', substitute=np.median(cD2))
    threshold_cD3 = pywt.threshold(cD3, threshold_factor*np.std(cD3), mode='soft', substitute=np.median(cD3))

    # Multi-scale edge enhancement
    edge_boost_level1 = 100  # Strong boost for finest details
    edge_boost_level2 = 1  # Moderate boost for mid-level details
    edge_boost_level3 = 1  # Mild boost for coarse details
    
    coeffs_enhanced = [
        cA3,
        (cH3*edge_boost_level3, cV3*edge_boost_level3, threshold_cD3),
        (cH2*edge_boost_level2, cV2*edge_boost_level2, threshold_cD2),
        (cH1*edge_boost_level1, cV1*edge_boost_level1, threshold_cD1)
    ]

    # Reconstruction with careful normalization
    img_recon = pywt.waverec2(coeffs_enhanced, wavelet)
    img_recon = np.clip(img_recon, 0, 1)

    # Adaptive CLAHE with smaller tile size
    clahe = cv2.createCLAHE(clipLimit=0.3, tileGridSize=(2, 2))  # Slightly increased clip limit
    img_clahe = clahe.apply((img_recon * 65535).astype(np.uint16))
    img_clahe = img_clahe.astype(np.float32) / 65535.0

    # Dynamic gamma correction
    p5, p95 = np.percentile(img_clahe, (2, 98))
    gamma = 0.5  # Increased contrast
    img_gamma = exposure.adjust_gamma(img_clahe, gamma=gamma)

    # Enhanced edge-preserving sharpening
    blurred = cv2.GaussianBlur(img_gamma, (0, 0), 20.0)  # Reduced sigma for finer edges
    img_sharp = cv2.addWeighted(img_gamma, 4.0, blurred, -3.0, 0)  # Stronger sharpening
    img_sharp = np.clip(img_sharp, 0, 1)

    # Final conversion with improved stretching
    enhanced_img = exposure.rescale_intensity(img_sharp, in_range=(p5, p95), out_range=(0, 255)).astype(np.uint8)

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
 