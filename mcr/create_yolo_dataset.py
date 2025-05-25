import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import re
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm

# Configuration
DATASET_PATH = "oxford-pet/images/"
SEGMENTATION_PATH = "oxford-pet/annotations/trimaps/"
OUTPUT_DIR = "pet_yolo_dataset"

def validate_paths():
    """Validate dataset and annotation paths."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path {DATASET_PATH} does not exist.")
    if not os.path.exists(SEGMENTATION_PATH):
        raise FileNotFoundError(f"Segmentation path {SEGMENTATION_PATH} does not exist.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_breed_from_filename(filename):
    """Extract breed name from filename."""
    basename = os.path.basename(filename)
    breed = re.split(r'_\d+\.jpg$', basename)[0]
    return breed.replace('_', ' ').title()

def save_breed_list(breeds):
    """Save unique breeds to a file."""
    unique_breeds = sorted(list(set(breeds)))
    breed_list_path = f"{OUTPUT_DIR}/breed_list.txt"
    with open(breed_list_path, 'w', encoding='utf-8') as f:
        for breed in unique_breeds:
            f.write(f"{breed}\n")
    print(f"Saved breed list to {breed_list_path}")
    return unique_breeds

def create_yolo_dataset():
    """Create YOLO dataset structure and process images."""
    validate_paths()

    # Create directories
    for subset in ['train', 'val']:
        os.makedirs(f"{OUTPUT_DIR}/images/{subset}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{subset}", exist_ok=True)

    # Collect image paths
    all_images = list(Path(DATASET_PATH).glob('*.jpg'))
    image_paths = [str(p) for p in all_images if re.match(r'.+_\d+\.jpg$', p.name)]
    if not image_paths:
        raise ValueError("No valid images found in the dataset.")

    # Log missing or corrupted images
    skipped_images = []
    valid_image_paths = []
    breeds = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not load {path} (corrupted or missing)")
            skipped_images.append(path)
            continue
        valid_image_paths.append(path)
        breeds.append(get_breed_from_filename(path))

    if skipped_images:
        with open(f"{OUTPUT_DIR}/skipped_images.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(skipped_images))
        print(f"Logged {len(skipped_images)} skipped images to {OUTPUT_DIR}/skipped_images.txt")

    if not valid_image_paths:
        raise ValueError("No valid images available after filtering.")

    # Split dataset
    df = pd.DataFrame({'image_path': valid_image_paths, 'breed': breeds})
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['breed'])

    print(f"Processing {len(train_df)} training images...")
    process_images(train_df, "train")

    print(f"Processing {len(val_df)} validation images...")
    process_images(val_df, "val")

    # Create dataset.yaml
    unique_breeds = save_breed_list(breeds)
    with open(f"{OUTPUT_DIR}/dataset.yaml", 'w', encoding='utf-8') as f:
        f.write(f"path: {os.path.abspath(OUTPUT_DIR)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(unique_breeds)}\n")
        f.write(f"names: {unique_breeds}\n")

    print(f"Created dataset with {len(unique_breeds)} classes for detection and breed classification")

def process_images(df, subset):
    """Process images and create YOLO labels."""
    breed_to_id = {breed: idx for idx, breed in enumerate(sorted(list(set(df['breed']))))}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['image_path']
        img_name = os.path.basename(img_path)
        breed = row['breed']
        breed_id = breed_to_id[breed]

        try:
            shutil.copy(img_path, f"{OUTPUT_DIR}/images/{subset}/{img_name}")
        except Exception as e:
            print(f"Warning: Could not copy {img_path}, error: {e}")
            continue

        img_id = os.path.splitext(img_name)[0]
        trimap = Path(SEGMENTATION_PATH) / f"{img_id}.png"
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue

        if trimap.exists():
            mask = cv2.imread(str(trimap), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not load trimap {trimap}")
                cx, cy, nw, nh = 0.5, 0.5, 0.8, 0.8
            else:
                pet_mask = (mask == 1).astype(np.uint8)
                cnts, _ = cv2.findContours(pet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    H, W = img.shape[:2]
                    cx = (x + w/2) / W
                    cy = (y + h/2) / H
                    nw = w / W
                    nh = h / H
                else:
                    cx, cy, nw, nh = 0.5, 0.5, 0.8, 0.8
        else:
            print(f"Warning: Trimap {trimap} not found, using default bounding box")
            cx, cy, nw, nh = 0.5, 0.5, 0.8, 0.8

        lbl_path = f"{OUTPUT_DIR}/labels/{subset}/{img_id}.txt"
        with open(lbl_path, 'w', encoding='utf-8') as f:
            f.write(f"{breed_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

def apply_augmentation():
    """Apply data augmentation to training images."""
    transform = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.1),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True, min_visibility=0.5))

    imgs = list(Path(f"{OUTPUT_DIR}/images/train").glob('*.jpg'))
    print(f"Augmenting {len(imgs)} images...")
    for path in tqdm(imgs):
        img = cv2.imread(str(path))
        if img is None:
            print(f"Warning: Could not load {path} for augmentation")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lbl_file = f"{OUTPUT_DIR}/labels/train/{path.stem}.txt"
        bbs, cls = [], []
        if Path(lbl_file).exists():
            with open(lbl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Warning: Invalid label format in {lbl_file}: {line.strip()}")
                        continue
                    try:
                        class_id = int(parts[0])  # Class ID must be an integer
                        bbox = [float(x) for x in parts[1:]]  # Bounding box coordinates as floats
                        cls.append(class_id)
                        bbs.append(bbox)
                    except ValueError as e:
                        print(f"Warning: Error parsing label in {lbl_file}: {e}")
                        continue
        if not bbs:
            print(f"Warning: No valid labels found for {lbl_file}")
            continue
        for i in range(2):  # Generate 2 augmentations per image
            try:
                aug = transform(image=img, bboxes=bbs, class_labels=cls)
                if not aug['bboxes']:
                    print(f"Warning: No valid bboxes after augmentation for {path} (aug {i})")
                    continue
                out_img = cv2.cvtColor(aug['image'], cv2.COLOR_RGB2BGR)
                name = f"{path.stem}_aug{i}.jpg"
                cv2.imwrite(f"{OUTPUT_DIR}/images/train/{name}", out_img)
                with open(f"{OUTPUT_DIR}/labels/train/{path.stem}_aug{i}.txt", 'w', encoding='utf-8') as f:
                    for idx, box in enumerate(aug['bboxes']):
                        f.write(f"{aug['class_labels'][idx]} {' '.join([f'{x:.6f}' for x in box])}\n")
            except Exception as e:
                print(f"Warning: Augmentation failed for {path} (aug {i}): {e}")
                continue
    print("Augmentation completed")

if __name__ == "__main__":
    try:
        create_yolo_dataset()
        apply_augmentation()
        print("Dataset preparation completed! Next steps:")
        print("1. Train the model using: python train_model.py")
        print("2. Copy the trained model (best.pt) to the working directory")
        print("3. Run the inference script (train_yolo8_model.py)")
    except Exception as e:
        print(f"Error during dataset preparation: {e}")