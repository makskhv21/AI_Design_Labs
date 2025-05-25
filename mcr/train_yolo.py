from ultralytics import YOLO
import yaml
import os
import cv2

# Завантаження попередньо навченої моделі YOLOv8
model = YOLO("yolov8n.pt")  # Використовуємо стандартну модель YOLOv8 Nano

# Читання класів
with open('data/classes.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
print("Завантажені класи:", classes)

# Виведення поточного робочого каталогу для перевірки
print("Поточний робочий каталог:", os.getcwd())

# Налаштування конфігурації даних з абсолютними шляхами
current_dir = os.getcwd()
data = {
    'train': os.path.join(current_dir, 'data', 'train', 'images'),
    'val': os.path.join(current_dir, 'data', 'val', 'images'),
    'test': os.path.join(current_dir, 'data', 'test', 'images'),  # Окрема тестова вибірка
    'nc': len(classes),
    'names': classes
}

# Збереження конфігурації у файл data.yaml
file_path = 'data.yaml'
with open(file_path, 'w') as f:
    yaml.dump(data, f)

# Навчання моделі
data_path = 'data.yaml'
model.train(data=data_path, epochs=50, batch=32)
print("Навчання завершено.")

# Функція для розпізнавання об'єктів на відео
def process_video(video_path, output_path, model):
    """Обробляє відеофайл, розпізнає об'єкти та зберігає результат."""
    # Відкриття відеофайлу
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Не вдалося відкрити відеофайл")
        return
    
    # Отримання параметрів відео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Налаштування для запису вихідного відео
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Обробка відео кадр за кадром
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Виявлення об'єктів на кадрі
        results = model(frame)
        
        # Відображення анотацій на кадрі
        annotated_frame = results[0].plot()
        
        # Запис анотованого кадру у вихідне відео
        out.write(annotated_frame)
    
    # Закриття відеофайлів
    cap.release()
    out.release()
    print("Обробка відео завершена. Вихідне відео збережено у", output_path)

# Розпізнавання відео після навчання
video_path = "path/to/your/video.mp4"
output_path = "path/to/output/video.mp4"
process_video(video_path, output_path, model)