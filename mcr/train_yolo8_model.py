import os
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

def load_breed_list():
    """Load breed list from file."""
    BREED_LIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pet_yolo_dataset/breed_list.txt")
    if not os.path.exists(BREED_LIST_PATH):
        print(f"Error: Breed list not found at {BREED_LIST_PATH}")
        return []
    with open(BREED_LIST_PATH, 'r', encoding='utf-8') as f:
        breed_list = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(breed_list)} breeds")
    return breed_list

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def apply_nms(boxes, scores, iou_threshold=0.5):
    """Apply Non-Maximum Suppression (NMS) to filter overlapping boxes."""
    if not boxes:
        return []
    indices = np.argsort(scores)[::-1]
    keep = []
    while indices.size > 0:
        current_idx = indices[0]
        keep.append(current_idx)
        ious = [calculate_iou(boxes[current_idx], boxes[idx]) for idx in indices[1:]]
        indices = np.array([idx for i, idx in enumerate(indices[1:]) if ious[i] < iou_threshold])
        indices = indices + 1
    return keep

def process_frame(frame, yolo_model, breed_list):
    """Process a single frame with YOLO model."""
    try:
        results = yolo_model(frame, conf=0.3)
        boxes = []
        scores = []
        classes = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].item()
            cls = int(box.cls[0].item())
            boxes.append(xyxy)
            scores.append(confidence)
            classes.append(cls)

        if boxes:
            keep_indices = apply_nms(boxes, scores, iou_threshold=0.5)
            boxes = [boxes[i] for i in keep_indices]
            scores = [scores[i] for i in keep_indices]
            classes = [classes[i] for i in keep_indices]

            output_frame = frame.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            font_color = (0, 255, 0)

            for xyxy, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = xyxy
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                if x2 <= x1 or y2 <= y1 or (x2 - x1) < 30 or (y2 - y1) < 30:
                    continue

                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                breed_name = breed_list[cls] if cls < len(breed_list) else "Unknown"
                text = f"{breed_name} ({score:.2f})"
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                cv2.rectangle(output_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)
                cv2.putText(output_frame, text, (x1, y1 - 5), font, font_scale, font_color, font_thickness)

            return output_frame
        return frame
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame

def process_image(image_path, output_path, yolo_model, breed_list):
    """Process an image with YOLO model."""
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return False
    processed_image = process_frame(image, yolo_model, breed_list)
    try:
        cv2.imwrite(output_path, processed_image)
        print(f"Result saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving image {output_path}: {e}")
        return False

def process_video(video_path, output_path, yolo_model, breed_list):
    """Process a video with YOLO model."""
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    try:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    except Exception as e:
        print(f"Error creating video writer for {output_path}: {e}")
        cap.release()
        return False

    frame_count = 0
    start_time = time.time()
    print(f"Starting processing of {total_frames} frames...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        processed_frame = process_frame(frame, yolo_model, breed_list)
        out.write(processed_frame)
        frame_count += 1
        if frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            fps_processing = frame_count / elapsed_time if elapsed_time > 0 else 0
            remaining_frames = total_frames - frame_count
            estimated_time = remaining_frames / fps_processing if fps_processing > 0 else 0
            print(f"Processed {frame_count}/{total_frames} frames "
                  f"({frame_count / total_frames * 100:.1f}%) - "
                  f"Estimated time remaining: {estimated_time / 60:.1f} minutes")

    cap.release()
    out.release()
    elapsed_time = time.time() - start_time
    print(f"Video processing completed in {elapsed_time / 60:.1f} minutes. Result saved to {output_path}")
    return True

def detect_media_type(file_path):
    """Detect if the file is an image or video."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    ext = os.path.splitext(file_path.lower())[1]
    if ext in image_extensions:
        return "image"
    elif ext in video_extensions:
        return "video"
    else:
        return "unknown"

def create_output_path(input_path):
    """Create output path for processed media."""
    directory = os.path.dirname(input_path)
    filename, extension = os.path.splitext(os.path.basename(input_path))
    if directory == "":
        directory = "."
    new_filename = f"{filename}_detection{extension}"
    return os.path.join(directory, new_filename)

def main():
    print("=" * 50)
    print("Universal Animal Detector and Breed Classifier")
    print("=" * 50)

    # Define model path
    YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Error: Model file {YOLO_MODEL_PATH} not found.")
        print("Please train the model using 'yolo train' or place a trained 'best.pt' file in the working directory.")
        return

    # Get input path
    input_path = input("Enter the path to an image or video: ").strip()
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found")
        return

    # Detect media type
    media_type = detect_media_type(input_path)
    if media_type == "unknown":
        print(f"Error: File format {input_path} is not supported")
        print("Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff, .gif")
        print("Supported video formats: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm")
        return

    output_path = create_output_path(input_path)

    # Load model and breed list
    print(f"Loading YOLOv8 model from {YOLO_MODEL_PATH}...")
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    breed_list = load_breed_list()
    if not breed_list:
        print("Warning: Breed list is empty. Detection will proceed without breed names.")
        breed_list = [f"Class_{i}" for i in range(100)]

    # Process media
    success = False
    if media_type == "image":
        success = process_image(input_path, output_path, yolo_model, breed_list)
    elif media_type == "video":
        success = process_video(input_path, output_path, yolo_model, breed_list)

    if success:
        print("\nTask completed successfully!")
        print(f"Result saved to: {output_path}")
    else:
        print("\nError processing media file.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")