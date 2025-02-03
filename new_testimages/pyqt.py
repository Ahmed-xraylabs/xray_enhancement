import sys
import os
import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Force software rendering to avoid OpenGL issues (optional, uncomment if needed)
# os.environ['QT_XCB_GL_INTEGRATION'] = 'none'

# Image processing function (unchanged from original code)
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
    blurred = cv2.GaussianBlur(gamma_corrected, (0, 0), 2.0)
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

class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processor")
        self.setGeometry(100, 100, 1400, 900)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(20)

        # Styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E3440;
            }
            QLabel {
                color: #D8DEE9;
                font-size: 16px;
            }
            QPushButton {
                background-color: #5E81AC;
                color: #D8DEE9;
                border: none;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #81A1C1;
            }
        """)

        # Upload Button
        self.upload_button = QPushButton("Upload TIFF File")
        self.upload_button.setFont(QFont("Arial", 14))
        self.upload_button.clicked.connect(self.upload_file)
        self.layout.addWidget(self.upload_button, alignment=Qt.AlignCenter)

        # Horizontal layout for original and enhanced images
        self.image_layout = QHBoxLayout()
        self.image_layout.setSpacing(20)

        # Original Image Label (Very Large)
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 2px solid #4C566A; padding: 10px;")
        self.image_layout.addWidget(self.original_label, 3)  # Allocate more space to original image

        # Enhanced Image Label (Larger than histogram)
        self.enhanced_label = QLabel("Enhanced Image")
        self.enhanced_label.setAlignment(Qt.AlignCenter)
        self.enhanced_label.setStyleSheet("border: 2px solid #4C566A; padding: 10px;")
        self.image_layout.addWidget(self.enhanced_label, 2)  # Allocate less space to enhanced image

        self.layout.addLayout(self.image_layout)

        # Histogram Canvas (Very Small)
        self.figure, self.ax = plt.subplots(figsize=(8, 2))  # Reduced height for histogram
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #3B4252;")
        self.layout.addWidget(self.canvas, stretch=1)  # Allocate minimal vertical space to histogram

    def upload_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF File", "", "TIFF Files (*.tif *.tiff);;All Files (*)", options=options
        )
        if file_path:
            try:
                # Process the image
                enhanced_img, original_img = process_image(file_path)

                # Display original image preview
                p99_5 = np.percentile(original_img, 99.5)
                original_preview = np.clip(original_img.astype(np.float32) / p99_5, 0, 1)
                original_preview = (original_preview * 255).astype(np.uint8)
                self.display_image(original_preview, self.original_label)

                # Display enhanced image
                self.display_image(enhanced_img, self.enhanced_label)

                # Display histogram
                self.ax.clear()
                self.ax.hist(enhanced_img.ravel(), bins=256, range=(0, 255), color='gray')
                self.ax.set_title("Enhanced Image Histogram", color="#D8DEE9", fontsize=10)
                self.ax.tick_params(axis='x', colors="#D8DEE9", labelsize=8)
                self.ax.tick_params(axis='y', colors="#D8DEE9", labelsize=8)
                self.ax.spines['bottom'].set_color('#D8DEE9')
                self.ax.spines['top'].set_color('#D8DEE9')
                self.ax.spines['left'].set_color('#D8DEE9')
                self.ax.spines['right'].set_color('#D8DEE9')
                self.canvas.draw()

            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def display_image(self, image, label):
        height, width = image.shape
        qimage = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        # Dynamically resize images when the window is resized
        super().resizeEvent(event)

        # Check if original_label has a pixmap
        if self.original_label.pixmap() is not None:
            self.display_image(self.original_label.pixmap().toImage(), self.original_label)

        # Check if enhanced_label has a pixmap
        if self.enhanced_label.pixmap() is not None:
            self.display_image(self.enhanced_label.pixmap().toImage(), self.enhanced_label)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec_())