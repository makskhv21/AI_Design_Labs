from ultralytics import YOLO
import os
import shutil
from pathlib import Path

def train_yolo_model():
    """Train YOLOv8 model on the pet dataset."""
    DATASET_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pet_yolo_dataset/dataset.yaml")
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Validate dataset.yaml
    if not os.path.exists(DATASET_YAML):
        print(f"Error: Dataset configuration file {DATASET_YAML} not found.")
        print("Please run prepare_dataset.py first to create the dataset.")
        return

    # Initialize model
    try:
        model = YOLO("yolov8n.pt")  # Use yolov8n.pt as the base model
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        return

    # Train model
    print("Starting YOLOv8 training...")
    try:
        model.train(
            data=DATASET_YAML,
            epochs=50,
            imgsz=640,
            batch=16,
            name="pet_breed_detector",
            project="runs/train",
            exist_ok=True
        )
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # Copy best.pt to working directory
    trained_model_path = None
    for exp_dir in sorted(Path("runs/train").glob("pet_breed_detector*"), key=lambda x: x.stat().mtime, reverse=True):
        best_pt = exp_dir / "weights/best.pt"
        if best_pt.exists():
            trained_model_path = best_pt
            break

    if trained_model_path:
        dest_path = os.path.join(OUTPUT_DIR, "best.pt")
        shutil.copy(trained_model_path, dest_path)
        print(f"Training completed! Model weights copied to {dest_path}")
    else:
        print("Error: Trained model (best.pt) not found in runs/train.")

if __name__ == "__main__":
    try:
        train_yolo_model()
    except Exception as e:
        print(f"Unexpected error: {e}")