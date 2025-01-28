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
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit
 
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
 
    # Wavelet transform with different parameters
    wavelet = 'bior2.2'  # Changed wavelet for better edge preservation
    coeffs = pywt.wavedec2(img_norm, wavelet, level=2)  # Reduced decomposition level
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
 
    # Adaptive thresholding using BayesShrink
    threshold_factor = 1.5  # More conservative threshold
    threshold_cD1 = pywt.threshold(cD1, threshold_factor*np.std(cD1), mode='soft', substitute=np.median(cD1))
    threshold_cD2 = pywt.threshold(cD2, threshold_factor*np.std(cD2), mode='soft', substitute=np.median(cD2))
 
    # Enhanced edge amplification
    edge_boost = 50  # Increased edge enhancement factor
    coeffs_enhanced = [
        cA2,
        (cH2*1, cV2*1, threshold_cD2),  # Moderate enhancement for mid-level features
        (cH1*edge_boost, cV1*edge_boost, threshold_cD1)  # Stronger enhancement for fine details
    ]
 
    # Reconstruction with careful normalization
    img_recon = pywt.waverec2(coeffs_enhanced, wavelet)
    img_recon = np.clip(img_recon, 0, 1)
 
    # Adaptive CLAHE with smaller tile size
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(4, 4))  # Smaller grid for local contrast
    img_clahe = clahe.apply((img_recon * 65535).astype(np.uint16))  # Use 16-bit processing
    img_clahe = img_clahe.astype(np.float32) / 65535.0
 
    # Dynamic gamma correction
    p5, p95 = np.percentile(img_clahe, (2, 100))  # Wider percentile range
    gamma = 0.6
    img_gamma = exposure.adjust_gamma(img_clahe, gamma=gamma)
 
    # Edge-preserving sharpening using unsharp masking
    blurred = cv2.GaussianBlur(img_gamma, (0, 0), 8.0)
    img_sharp = cv2.addWeighted(img_gamma, 3, blurred, -2, 0)
    img_sharp = np.clip(img_sharp, 0, 1)
 
    # Final conversion to 8-bit with histogram stretching
    enhanced_img = exposure.rescale_intensity(img_sharp, out_range=(0, 255)).astype(np.uint8)
 
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
 