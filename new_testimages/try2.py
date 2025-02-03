import cv2
import numpy as np
import argparse

def gaussian_kernel():
    return np.array([0.05, 0.25, 0.4, 0.25, 0.05])

def downsample(image):
    kernel = gaussian_kernel()
    blurred = cv2.sepFilter2D(image, -1, kernel, kernel, borderType=cv2.BORDER_REFLECT)
    return blurred[::2, ::2]

def upsample(image):
    kernel = gaussian_kernel()
    h, w = image.shape
    upsampled = np.zeros((2*h, 2*w), dtype=image.dtype)
    upsampled[::2, ::2] = image
    upsampled = cv2.sepFilter2D(upsampled, -1, kernel, kernel, borderType=cv2.BORDER_REFLECT)
    return upsampled

def build_pyramid(image, levels):
    # Calculate required padding to make dimensions divisible by 2^levels
    h, w = image.shape
    divisor = 2 ** levels
    pad_h = (divisor - (h % divisor)) % divisor
    pad_w = (divisor - (w % divisor)) % divisor
    padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    pyramid = []
    current = padded_image.astype(np.float32)
    for _ in range(levels):
        down = downsample(current)
        up = upsample(down)
        laplacian = current - up
        pyramid.append(laplacian)
        current = down
    pyramid.append(current)  # Gaussian top
    return pyramid, (pad_h, pad_w), h, w

def modify_coefficients(pyramid, p=0.7, x_c_ratio=0.1):
    laplacian_layers = pyramid[:-1]
    original_max = max(np.max(np.abs(layer)) for layer in laplacian_layers) if laplacian_layers else 1.0
    x_c = x_c_ratio * original_max

    modified_layers = []
    for layer in laplacian_layers:
        x = layer
        abs_x = np.abs(x)
        mask = abs_x < x_c

        y_prime = np.zeros_like(x)
        y_prime[mask] = x[mask] * (x_c / original_max) ** (p - 1) if original_max != 0 else 0.0
        y_prime[~mask] = np.sign(x[~mask]) * (abs_x[~mask] ** p) / (original_max ** (p - 1)) if original_max != 0 else 0.0
        modified_layers.append(y_prime)

    new_max = max(np.max(np.abs(layer)) for layer in modified_layers) if modified_layers else 1.0
    a = original_max / new_max if new_max != 0 else 1.0
    modified_layers = [a * layer for layer in modified_layers]
    return modified_layers + [pyramid[-1]]

def reconstruct_pyramid(modified_pyramid, pad_info, original_h, original_w):
    current = modified_pyramid[-1]
    for layer in reversed(modified_pyramid[:-1]):
        up = upsample(current)
        current = up + layer
    # Remove initial padding
    current = current[:original_h, :original_w]
    return current

def main(input_path, output_path, levels=5, p=0.7, x_c_ratio=0.1):
    image = cv2.imread(input_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read the image.")

    original_dtype = image.dtype
    max_val = np.iinfo(original_dtype).max

    # Build pyramid with pre-padded image
    pyramid, pad_info, original_h, original_w = build_pyramid(image, levels)
    # Normalize pyramid
    normalized_pyramid = [layer.astype(np.float32) / max_val for layer in pyramid]
    modified_pyramid = modify_coefficients(normalized_pyramid, p, x_c_ratio)
    # Reconstruct and denormalize
    enhanced = reconstruct_pyramid(modified_pyramid, pad_info, original_h, original_w)
    enhanced = np.clip(enhanced, 0, 1) * max_val
    enhanced = enhanced.astype(original_dtype)
    
    cv2.imwrite(output_path, enhanced)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhance X-ray image using MUSICA algorithm.')
    parser.add_argument('input', help='Input TIFF image path')
    parser.add_argument('output', help='Output PNG image path')
    parser.add_argument('--levels', type=int, default=5, help='Number of pyramid levels (default: 5)')
    parser.add_argument('--p', type=float, default=0.7, help='Non-linearity exponent (0.5-0.8, default: 0.7)')
    parser.add_argument('--x_c_ratio', type=float, default=0.1, help='Crossover ratio (default: 0.1)')
    args = parser.parse_args()

    main(args.input, args.output, args.levels, args.p, args.x_c_ratio)