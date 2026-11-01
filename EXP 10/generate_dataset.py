import os
import cv2
import numpy as np

def create_synthetic_brain_mri(has_tumor=False, seed=None):
    """
    Generates a realistic synthetic Brain MRI axial slice.
    """
    if seed is not None:
        np.random.seed(seed)
    
    img = np.zeros((256, 256), dtype=np.uint8)
    
    # 1. Outer Skull
    cv2.ellipse(img, (128, 128), (95, 115), 0, 0, 360, 180, 4)
    # Scalp tissue layer
    cv2.ellipse(img, (128, 128), (91, 111), 0, 0, 360, 40, -1)
    
    # 2. Inner Brain Parenchyma (Cerebral Cortex)
    brain_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(brain_mask, (128, 128), (84, 104), 0, 0, 360, 255, -1)
    
    # Brain texture noise (gray matter / white matter variation)
    noise = np.random.normal(110, 15, (256, 256)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (9, 9), 3)
    
    # Add sulci / gyri patterns (convolutions)
    gyri = np.sin(np.linspace(0, 25, 256))[:, None] * np.cos(np.linspace(0, 25, 256))[None, :]
    gyri = ((gyri + 1) * 20).astype(np.float32)
    
    texture = np.clip(noise + gyri, 60, 160).astype(np.uint8)
    brain_parenchyma = cv2.bitwise_and(texture, texture, mask=brain_mask)
    
    # 3. Ventricles (fluid-filled dark cavities in center)
    ventricle_mask = np.zeros((256, 256), dtype=np.uint8)
    # Left and right lateral ventricles
    cv2.ellipse(ventricle_mask, (118, 125), (10, 30), -15, 0, 360, 255, -1)
    cv2.ellipse(ventricle_mask, (138, 125), (10, 30), 15, 0, 360, 255, -1)
    cv2.ellipse(ventricle_mask, (128, 145), (6, 12), 0, 0, 360, 255, -1)
    
    # Darken ventricles (CSF appears dark on T1 MRI)
    brain_parenchyma[ventricle_mask == 255] = np.random.normal(30, 5, np.sum(ventricle_mask == 255)).astype(np.uint8)
    
    # Central fissure
    cv2.line(brain_parenchyma, (128, 40), (128, 220), (35,), 1)
    
    combined = cv2.add(img, brain_parenchyma)
    
    # 4. If Pathological: Add Hyperintense Tumor Lesion with Edema
    if has_tumor:
        # Random location in left or right hemisphere
        tumor_x = int(np.random.choice([85, 170]) + np.random.randint(-10, 10))
        tumor_y = int(np.random.randint(90, 160))
        tumor_radius = int(np.random.randint(14, 24))
        
        # Surrounding vasogenic edema (moderate hyperintensity)
        edema_radius = int(tumor_radius * 1.6)
        cv2.circle(combined, (tumor_x, tumor_y), edema_radius, (170,), -1)
        
        # Dense tumor core (bright hyperintense region)
        cv2.circle(combined, (tumor_x, tumor_y), tumor_radius, (235,), -1)
        
        # Irregular necrotic core / heterogeneous center
        cv2.circle(combined, (tumor_x + 3, tumor_y - 2), int(tumor_radius * 0.45), (130,), -1)
    
    # Final smoothing and subtle sensor noise
    combined = cv2.GaussianBlur(combined, (3, 3), 0.8)
    sensor_noise = np.random.normal(0, 4, (256, 256)).astype(np.float32)
    final_img = np.clip(combined.astype(np.float32) + sensor_noise, 0, 255).astype(np.uint8)
    
    return final_img

def create_synthetic_chest_xray(has_pneumonia=False, seed=None):
    """
    Generates a realistic synthetic Chest X-Ray frontal view.
    """
    if seed is not None:
        np.random.seed(seed)
        
    img = np.zeros((256, 256), dtype=np.uint8)
    
    # Background soft tissue / thoracic wall
    cv2.rectangle(img, (20, 20), (236, 240), 90, -1)
    
    # Lungs (dark radiolucent cavities because of air)
    lung_mask = np.zeros((256, 256), dtype=np.uint8)
    # Right lung
    cv2.ellipse(lung_mask, (80, 130), (38, 70), 0, 0, 360, 255, -1)
    # Left lung
    cv2.ellipse(lung_mask, (176, 130), (38, 70), 0, 0, 360, 255, -1)
    
    # Lung texture (vascular markings / bronchial tree)
    lung_noise = np.random.normal(45, 10, (256, 256)).astype(np.float32)
    lung_noise = cv2.GaussianBlur(lung_noise, (7, 7), 2)
    lung_field = np.clip(lung_noise, 20, 85).astype(np.uint8)
    
    img[lung_mask == 255] = lung_field[lung_mask == 255]
    
    # Rib cage outlines
    for y in range(60, 220, 24):
        cv2.ellipse(img, (128, y), (100, 22), 0, 10, 170, 135, 3)
    
    # Spine (vertebral column - vertical dense strip)
    cv2.rectangle(img, (120, 20), (136, 240), 170, -1)
    
    # Mediastinum and Cardiac Silhouette (Heart shadow on left hemithorax)
    cv2.ellipse(img, (140, 155), (36, 45), -30, 0, 360, 195, -1)
    
    # Clavicles (collar bones)
    cv2.ellipse(img, (75, 48), (45, 10), -12, 0, 360, 200, 3)
    cv2.ellipse(img, (181, 48), (45, 10), 12, 0, 360, 200, 3)
    
    # Diaphragm domes
    cv2.ellipse(img, (78, 220), (45, 25), 0, 180, 360, 190, -1)
    cv2.ellipse(img, (178, 225), (45, 25), 0, 180, 360, 180, -1)
    
    # If Pneumonia: Add alveolar consolidation / opacification (cloudy white opacity)
    if has_pneumonia:
        pneumonia_mask = np.zeros((256, 256), dtype=np.uint8)
        # Infiltrate in right or left lower/middle lobe
        cx = int(np.random.choice([75, 170]))
        cy = int(np.random.randint(120, 180))
        cv2.ellipse(pneumonia_mask, (cx, cy), (26, 32), 20, 0, 360, 255, -1)
        
        # Soft fluffy consolidation opacity
        opacity = cv2.GaussianBlur(pneumonia_mask.astype(np.float32), (31, 31), 10)
        img = np.clip(img.astype(np.float32) + opacity * 0.55, 0, 255).astype(np.uint8)
        
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    return img

def build_complete_dataset(base_dir=None):
    """
    Creates dataset directories and generates:
    1. Sample test images for demonstration
    2. Training dataset of 100 images (50 Normal, 50 Pathological)
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(base_dir, "sample_data")
    dataset_dir = os.path.join(base_dir, "dataset")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "normal"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "pathological"), exist_ok=True)
    
    print("Generating Patient Sample Scans...")
    # 4 Clinical Demo Samples
    cv2.imwrite(os.path.join(samples_dir, "patient_brain_normal.png"), 
                create_synthetic_brain_mri(has_tumor=False, seed=101))
    cv2.imwrite(os.path.join(samples_dir, "patient_brain_tumor.png"), 
                create_synthetic_brain_mri(has_tumor=True, seed=202))
    cv2.imwrite(os.path.join(samples_dir, "patient_chest_normal.png"), 
                create_synthetic_chest_xray(has_pneumonia=False, seed=303))
    cv2.imwrite(os.path.join(samples_dir, "patient_chest_pneumonia.png"), 
                create_synthetic_chest_xray(has_pneumonia=True, seed=404))
    
    print("Generating 100 Training Images for ML Pipeline...")
    # Generate 50 Normal (25 Brain + 25 Chest)
    for i in range(25):
        mri = create_synthetic_brain_mri(has_tumor=False, seed=1000 + i)
        cv2.imwrite(os.path.join(dataset_dir, "normal", f"norm_mri_{i+1:03d}.png"), mri)
        cxr = create_synthetic_chest_xray(has_pneumonia=False, seed=2000 + i)
        cv2.imwrite(os.path.join(dataset_dir, "normal", f"norm_cxr_{i+1:03d}.png"), cxr)
        
    # Generate 50 Pathological (25 Tumor + 25 Pneumonia)
    for i in range(25):
        mri = create_synthetic_brain_mri(has_tumor=True, seed=3000 + i)
        cv2.imwrite(os.path.join(dataset_dir, "pathological", f"path_tumor_{i+1:03d}.png"), mri)
        cxr = create_synthetic_chest_xray(has_pneumonia=True, seed=4000 + i)
        cv2.imwrite(os.path.join(dataset_dir, "pathological", f"path_pneumo_{i+1:03d}.png"), cxr)
        
    print(f"Dataset generation complete! Files saved to {dataset_dir} and {samples_dir}")

if __name__ == "__main__":
    build_complete_dataset()

