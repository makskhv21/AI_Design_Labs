import os
import requests
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

# Налаштування
PEXELS_API_KEY = "your_KEY_API"  # Вставте ваш ключ API від Pexels
HEADERS = {"Authorization": PEXELS_API_KEY}
IMG_SIZE = (224, 224)  # Розмір зображень для Xception
BATCH_SIZE = 32        # Розмір батчу
EPOCHS = 20            # Кількість епох для першого етапу

# Функція для завантаження зображень логотипів із Pexels
def download_pexels_images(query, folder_name, num_images=100):
    url = f"https://api.pexels.com/v1/search?query={query}&per_page={num_images}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        image_urls = [photo["src"]["large"] for photo in data["photos"]]
        os.makedirs(folder_name, exist_ok=True)
        for i, img_url in enumerate(image_urls):
            img_data = requests.get(img_url).content
            with open(f"{folder_name}/img_{i}.jpg", "wb") as img_file:
                img_file.write(img_data)
                print(f"Збережено: {folder_name}/img_{i}.jpg")
    else:
        print(f"Помилка для {query}")

# Завантаження зображень для брендів
brands = ["BMW", "Mercedes", "Audi", "Toyota", "Volkswagen"]
for brand in brands:
    download_pexels_images(f"{brand} logo", f"brand_images/{brand}")

# Генерація даних із аугментацією
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    "brand_images",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    "brand_images",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation"
)

class_names = list(train_generator.class_indices.keys())

# Побудова моделі Xception
base_model = Xception(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(len(class_names), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Заморожуємо базові шари для першого етапу
for layer in base_model.layers:
    layer.trainable = False

# Компіляція моделі
model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Планувальник швидкості навчання
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Перший етап навчання
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[lr_scheduler, early_stopping]
)

# Розморожуємо верхні шари для донавчання
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Перекомпіляція з меншою швидкістю навчання
model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Другий етап навчання
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=[lr_scheduler, early_stopping]
)

# Збереження моделі
model.save("brand_classifier_xception_finetuned.keras")

# Об'єднання історії навчання для графіків
train_acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
train_loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

# Створення графіків
plt.figure(figsize=(12, 5))

# Графік точності
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Тренувальна точність')
plt.plot(val_acc, label='Валідаційна точність')
plt.axvline(x=len(history.history['accuracy']), color='r', linestyle='--', label='Початок донавчання')
plt.title('Точність моделі')
plt.xlabel('Епохи')
plt.ylabel('Точність')
plt.legend()

# Графік втрат
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Тренувальні втрати')
plt.plot(val_loss, label='Валідаційні втрати')
plt.axvline(x=len(history.history['loss']), color='r', linestyle='--', label='Початок донавчання')
plt.title('Втрати моделі')
plt.xlabel('Епохи')
plt.ylabel('Втрати')
plt.legend()

plt.show()

# Функція для відображення 5 зображень по черзі без цифр
def display_predictions(model, generator, class_names, num_images=5):
    x, y = next(generator)
    predictions = model.predict(x)
    for i in range(num_images):
        img = x[i]
        true_label = class_names[int(y[i])]
        predicted_label = class_names[np.argmax(predictions[i])]
        
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"Факт: {true_label}\nПередбачено: {predicted_label}")
        plt.axis('off')
        plt.show()

print("Приклади передбачень на тренувальних даних:")
display_predictions(model, train_generator, class_names)

# Аналіз відео без цифр біля назв брендів
def analyze_video(video_path, model, class_names):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Помилка: не вдалося відкрити відео за шляхом {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    brand_intervals = {brand: [] for brand in class_names}
    current_brand = None
    start_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        time = frame_count / fps
        img = cv2.resize(frame, IMG_SIZE)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_brand = class_names[np.argmax(prediction)]

        if np.max(prediction) > 0.7:
            if current_brand != predicted_brand:
                if current_brand is not None:
                    brand_intervals[current_brand].append((start_time, time))
                current_brand = predicted_brand
                start_time = time
        else:
            if current_brand is not None:
                brand_intervals[current_brand].append((start_time, time))
                current_brand = None

        cv2.putText(frame, f"{predicted_brand}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Аналіз відео', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    if current_brand is not None:
        brand_intervals[current_brand].append((start_time, time))
    cap.release()
    cv2.destroyAllWindows()
    return brand_intervals

# Виконання аналізу відео
video_path = "vidos/video2.mp4"  # Вкажіть шлях до вашого відео
loaded_model = tf.keras.models.load_model("brand_classifier_xception_finetuned.keras")
brand_times = analyze_video(video_path, loaded_model, class_names)

# Виведення результатів
print("Логотипи з'являлися в наступних інтервалах:")
for brand, intervals in brand_times.items():
    print(f"{brand}:")
    for start, end in intervals:
        print(f"  Від {start:.2f} сек до {end:.2f} сек")