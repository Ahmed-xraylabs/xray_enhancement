import cv2
import numpy as np
import pywt

def dynamic_gamma_correction(img, gamma_base=0.7, gamma_range=0.6):
    """Adaptive gamma correction based on local intensity"""
    # Calculate local intensity using Gaussian blur
    blurred = cv2.GaussianBlur(img, (51, 51), 0)
    
    # Create gamma map with base value and intensity-dependent adjustment
    gamma_map = gamma_base + (1.0 - blurred) * gamma_range
    return np.power(img, gamma_map), gamma_map

def industrial_enhancement(image_path, output_path, 
                          gamma_base=0.7,
                          gamma_range=0.5,
                          density_threshold=0.4,
                          high_density_gain=1.8,
                          low_density_gain=1.3,
                          jpeg_quality=95):
    """
    Enhanced X-ray processing with dynamic gamma correction
    """
    
    # Read 16-bit TIFF image
    img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read the image file")

    # Normalize to [0, 1] range
    img = img.astype(np.float32)
    max_val = np.max(img)
    if max_val > 0:
        img /= max_val

    # Apply dynamic gamma correction
    img, gamma_map = dynamic_gamma_correction(img, gamma_base, gamma_range)

    # Create density mask using adaptive thresholding
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    density_mask = np.where(blurred > density_threshold, 1.0, 0.0)
    density_mask = cv2.erode(density_mask, np.ones((3,3), np.uint8), iterations=3)
    inverse_mask = 1.0 - density_mask

    # Multi-scale wavelet decomposition
    wavelet = 'bior3.3'
    levels = 4
    coeffs = pywt.wavedec2(img, wavelet, level=levels)
    
    # Multi-scale contrast enhancement
    processed_coeffs = [coeffs[0]]
    
    for level in range(1, len(coeffs)):
        H, V, D = coeffs[level]
        rows, cols = H.shape
        
        scaled_density = cv2.resize(density_mask, (cols, rows), 
                                  interpolation=cv2.INTER_AREA)
        scaled_inverse = cv2.resize(inverse_mask, (cols, rows),
                                  interpolation=cv2.INTER_AREA)

        level_weight = 1.2 + 0.3 * (len(coeffs) - level - 1)
        
        H = H * (scaled_density*high_density_gain + scaled_inverse*low_density_gain) * level_weight
        V = V * (scaled_density*high_density_gain + scaled_inverse*low_density_gain) * level_weight
        D = D * (scaled_density*high_density_gain + scaled_inverse*low_density_gain) * level_weight
        
        processed_coeffs.append((H, V, D))

    # Reconstruct image
    enhanced = pywt.waverec2(processed_coeffs, wavelet)
    enhanced = np.clip(enhanced, 0, 1)

    # Convert to 8-bit for JPEG/PNG output
    enhanced_8bit = (enhanced * 255).astype(np.uint8)

    # Industrial-optimized CLAHE
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(4,4))
    final_image = clahe.apply(enhanced_8bit)

    # Save in appropriate format
    if output_path.lower().endswith(('.jpg', '.jpeg')):
        cv2.imwrite(output_path, final_image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    elif output_path.lower().endswith('.png'):
        cv2.imwrite(output_path, final_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        cv2.imwrite(output_path, (enhanced * max_val).astype(np.uint16))

    print(f"Enhanced image saved to {output_path}")

# Usage examples
industrial_enhancement('002.tif', 'output.jpg', 
                      gamma_base=0.6, 
                      gamma_range=0.8,
                      density_threshold=0.35)