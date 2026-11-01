import cv2
import numpy as np
import scipy.stats as stats

class MedicalFeatureExtractor:
    def __init__(self):
        self.feature_names = [
            "mean_intensity",
            "std_intensity",
            "variance",
            "skewness",
            "kurtosis",
            "entropy",
            "glcm_contrast",
            "glcm_dissimilarity",
            "glcm_homogeneity",
            "glcm_energy",
            "glcm_correlation",
            "roi_area_ratio",
            "perimeter",
            "circularity",
            "compactness",
            "solidity",
            "hu_moment_1",
            "hu_moment_2"
        ]

    def compute_glcm_features(self, gray_img, levels=32):
        """
        Computes 5 Haralick texture features using Gray Level Co-occurrence Matrix (GLCM).
        Quantizes image to `levels` bins for speed and stability.
        """
        # Quantize image to fewer levels (e.g. 32 bins)
        quantized = (gray_img / (256.0 / levels)).astype(np.int32)
        h, w = quantized.shape
        
        # Directions: Horizontal (0 deg), Vertical (90 deg), Diagonal (45 deg)
        glcm = np.zeros((levels, levels), dtype=np.float64)
        
        # Horizontal (dx=1, dy=0)
        row1, row2 = quantized[:, :-1].flatten(), quantized[:, 1:].flatten()
        np.add.at(glcm, (row1, row2), 1)
        
        # Vertical (dx=0, dy=1)
        col1, col2 = quantized[:-1, :].flatten(), quantized[1:, :].flatten()
        np.add.at(glcm, (col1, col2), 1)
        
        # Make symmetric
        glcm = glcm + glcm.T
        total = np.sum(glcm)
        if total > 0:
            glcm = glcm / total
        else:
            glcm[0, 0] = 1.0

        # Feature calculations
        i_indices, j_indices = np.indices((levels, levels))
        
        diff = np.abs(i_indices - j_indices)
        contrast = float(np.sum((diff ** 2) * glcm))
        dissimilarity = float(np.sum(diff * glcm))
        homogeneity = float(np.sum(glcm / (1.0 + diff ** 2)))
        energy = float(np.sum(glcm ** 2))
        
        # Correlation
        i_mean = np.sum(i_indices * glcm)
        j_mean = np.sum(j_indices * glcm)
        i_var = np.sum(((i_indices - i_mean) ** 2) * glcm)
        j_var = np.sum(((j_indices - j_mean) ** 2) * glcm)
        i_std = np.sqrt(i_var) if i_var > 1e-8 else 1.0
        j_std = np.sqrt(j_var) if j_var > 1e-8 else 1.0
        
        cov = np.sum((i_indices - i_mean) * (j_indices - j_mean) * glcm)
        correlation = float(cov / (i_std * j_std))

        return {
            "glcm_contrast": round(contrast, 4),
            "glcm_dissimilarity": round(dissimilarity, 4),
            "glcm_homogeneity": round(homogeneity, 4),
            "glcm_energy": round(energy, 4),
            "glcm_correlation": round(correlation, 4)
        }

    def compute_statistical_features(self, gray_img):
        """
        Computes 1st order statistical and informational features:
        Mean, Standard Deviation, Variance, Skewness, Kurtosis, Shannon Entropy.
        """
        pixels = gray_img.flatten().astype(np.float64)
        mean_val = float(np.mean(pixels))
        std_val = float(np.std(pixels))
        variance_val = float(np.var(pixels))
        skew_val = float(stats.skew(pixels))
        kurt_val = float(stats.kurtosis(pixels))

        # Shannon Entropy
        hist, _ = np.histogram(pixels, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy_val = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0

        return {
            "mean_intensity": round(mean_val, 3),
            "std_intensity": round(std_val, 3),
            "variance": round(variance_val, 3),
            "skewness": round(skew_val, 3),
            "kurtosis": round(kurt_val, 3),
            "entropy": round(entropy_val, 3)
        }

    def compute_morphological_features(self, mask_img):
        """
        Computes shape and morphological descriptors from segmented binary mask.
        Area, Perimeter, Circularity, Compactness, Solidity, Hu Moments.
        """
        total_pixels = float(mask_img.shape[0] * mask_img.shape[1])
        white_pixels = float(np.sum(mask_img > 0))
        area_ratio = round(white_pixels / total_pixels, 4)

        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Pick largest contour (primary region of interest / anomaly)
            largest_c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_c)
            perimeter = cv2.arcLength(largest_c, True)
            
            # Circularity: 4 * pi * Area / Perimeter^2 (1.0 = perfect circle)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                compactness = (perimeter ** 2) / (area + 1e-5)
            else:
                circularity = 0.0
                compactness = 0.0
                
            # Solidity: Area / Convex Hull Area
            hull = cv2.convexHull(largest_c)
            hull_area = cv2.contourArea(hull)
            solidity = (area / hull_area) if hull_area > 0 else 0.0
            
            # Hu Moments
            moments = cv2.moments(largest_c)
            hu_raw = cv2.HuMoments(moments).flatten()
            # Log transform Hu moments for numerical scale
            hu1 = float(-1 * np.copysign(1.0, hu_raw[0]) * np.log10(abs(hu_raw[0]) + 1e-12))
            hu2 = float(-1 * np.copysign(1.0, hu_raw[1]) * np.log10(abs(hu_raw[1]) + 1e-12))
        else:
            perimeter = 0.0
            circularity = 0.0
            compactness = 0.0
            solidity = 0.0
            hu1 = 0.0
            hu2 = 0.0

        return {
            "roi_area_ratio": area_ratio,
            "perimeter": round(float(perimeter), 2),
            "circularity": round(float(circularity), 4),
            "compactness": round(float(compactness), 2),
            "solidity": round(float(solidity), 4),
            "hu_moment_1": round(hu1, 4),
            "hu_moment_2": round(hu2, 4)
        }

    def extract_all(self, gray_img, mask_img):
        """
        Extracts all 18 medical features and returns both a dictionary and numpy vector.
        """
        stat_feats = self.compute_statistical_features(gray_img)
        glcm_feats = self.compute_glcm_features(gray_img)
        morph_feats = self.compute_morphological_features(mask_img)

        combined_dict = {}
        combined_dict.update(stat_feats)
        combined_dict.update(glcm_feats)
        combined_dict.update(morph_feats)

        feature_vector = np.array([combined_dict[name] for name in self.feature_names], dtype=np.float32)

        return {
            "features_dict": combined_dict,
            "features_vector": feature_vector,
            "feature_names": self.feature_names
        }

