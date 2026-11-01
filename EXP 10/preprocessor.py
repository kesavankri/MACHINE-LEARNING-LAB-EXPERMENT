import cv2
import numpy as np
import base64

def encode_img_to_base64(img_arr):
    """Encodes an OpenCV image array to base64 PNG data URL."""
    if img_arr is None:
        return ""
    success, buffer = cv2.imencode('.png', img_arr)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

class MedicalImagePreprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def read_image(self, file_source):
        """
        Reads image from file path or bytes buffer.
        """
        if isinstance(file_source, str):
            img = cv2.imread(file_source)
        else:
            # File storage / bytes buffer
            file_bytes = np.frombuffer(file_source.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img

    def process(self, img_input):
        """
        Executes the full preprocessing pipeline on the input image:
        1. Resize
        2. Grayscale conversion
        3. Denoising / Gaussian Filtering
        4. Contrast Enhancement (CLAHE)
        5. Tissue / Lesion ROI Segmentation (Otsu Thresholding)
        6. Canny Edge Detection
        Returns:
            dict containing raw processed arrays and base64 strings for UI rendering
        """
        if isinstance(img_input, str) or hasattr(img_input, 'read'):
            raw_img = self.read_image(img_input)
        else:
            raw_img = img_input

        if raw_img is None:
            raise ValueError("Unable to decode or load the input medical image.")

        # Ensure raw_img has 3 channels for consistent visualization
        if len(raw_img.shape) == 2:
            resized = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
        else:
            resized = raw_img.copy()

        # 1. Resize & Standardization
        resized = cv2.resize(resized, self.target_size, interpolation=cv2.INTER_AREA)

        # 2. Grayscale conversion
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 3. Denoising using Gaussian Filter (removes high-frequency scanner noise)
        denoised = cv2.GaussianBlur(gray, (5, 5), 1.2)

        # 4. Contrast Enhancement with CLAHE (enhances subtle anatomical differences)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 5. Segmentation using Otsu's Thresholding & Morphological Operations
        _, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological Closing to fill small holes inside regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        segmented_mask = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)

        # 6. Canny Edge Detection (highlights anatomical boundaries)
        edges = cv2.Canny(enhanced, 50, 150)

        # 7. Create a Colored Mask Overlay on original image for medical visualization
        overlay = resized.copy()
        red_tint = np.zeros_like(resized)
        red_tint[:] = (0, 0, 240)
        blended = cv2.addWeighted(resized, 0.65, red_tint, 0.35, 0)
        overlay[segmented_mask == 255] = blended[segmented_mask == 255]

        return {
            "arrays": {
                "original": resized,
                "gray": gray,
                "denoised": denoised,
                "enhanced": enhanced,
                "segmented_mask": segmented_mask,
                "edges": edges,
                "overlay": overlay
            },
            "previews": {
                "original": encode_img_to_base64(resized),
                "gray": encode_img_to_base64(gray),
                "denoised": encode_img_to_base64(denoised),
                "enhanced": encode_img_to_base64(enhanced),
                "segmented_mask": encode_img_to_base64(segmented_mask),
                "edges": encode_img_to_base64(edges),
                "overlay": encode_img_to_base64(overlay)
            },
            "stats": {
                "dimensions": f"{resized.shape[1]}x{resized.shape[0]} px",
                "color_channels": 1 if len(gray.shape) == 2 else gray.shape[2],
                "min_intensity": int(np.min(gray)),
                "max_intensity": int(np.max(gray)),
                "mean_intensity": round(float(np.mean(gray)), 2),
                "std_intensity": round(float(np.std(gray)), 2)
            }
        }
