import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# --- Функція обробки зображень ---
def process_image(image, label):
    """Обробка зображень: зміна розміру та нормалізація."""
    image = tf.image.resize(image, (227, 227))  # Зміна розміру до 227x227 для AlexNet
    image = tf.cast(image, tf.float32) / 255.0  # Нормалізація значень до [0, 1]
    return image, label

# --- Завантаження і підготовка датасету ---
def fetch_and_prepare_data():
    """Завантаження і підготовка датасету Imagenette."""
    dataset, info = tfds.load('imagenette/160px-v2', with_info=True, as_supervised=True)
    train_data = dataset['train'].map(process_image).batch(64).prefetch(tf.data.AUTOTUNE)
    test_data = dataset['validation'].map(process_image).batch(64).prefetch(tf.data.AUTOTUNE)
    return train_data, test_data

# --- Перелік класів ---
categories = [
    'salmon', 'cocker spaniel', 'cd player', 'circular saw', 'cathedral',
    'trumpet', 'dump truck', 'fuel station', 'tennis ball', 'parachute'
]

# --- Візуалізація зображень ---
def display_random_images(data, categories):
    """Візуалізація 16 випадкових зображень із датасету."""
    data_shuffled = data.unbatch().shuffle(buffer_size=1000)  # Випадкове перемішування
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(data_shuffled.take(16)):  # Вибір 16 зображень
        plt.subplot(4, 4, i + 1)
        plt.imshow(image)
        plt.title(categories[label.numpy()])
        plt.axis('off')
    plt.show()

# --- Створення моделі AlexNet ---
def build_alexnet_model(classes_count):
    """Створення моделі AlexNet з заданою кількістю класів."""
    model = models.Sequential([
        layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', 
                      input_shape=(227, 227, 3)),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        layers.Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(classes_count, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Графіки результатів тренування ---
def plot_training_metrics(history):
    """Візуалізація графіків точності та втрат."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Тренувальні втрати')
    plt.plot(history.history['val_loss'], label='Валідаційні втрати')
    plt.xlabel('Епохи')
    plt.ylabel('Втрати')
    plt.title('Втрати моделі')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Тренувальна точність')
    plt.plot(history.history['val_accuracy'], label='Валідаційна точність')
    plt.xlabel('Епохи')
    plt.ylabel('Точність')
    plt.title('Точність моделі')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# --- Оцінка та передбачення ---
def assess_and_predict(model, test_data, categories):
    """Оцінка моделі та візуалізація передбачень."""
    test_loss, test_acc = model.evaluate(test_data)
    print(f'Тестова точність: {test_acc:.4f}')
    print(f'Тестові втрати: {test_loss:.4f}')

    # Візуалізація передбачень
    test_data_shuffled = test_data.unbatch().shuffle(buffer_size=1000)
    plt.figure(figsize=(12, 12))
    for i, (image, label) in enumerate(test_data_shuffled.take(9)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image)
        predictions = model.predict(np.expand_dims(image, axis=0), verbose=0)
        predicted_label = np.argmax(predictions)
        plt.title(f"Справжній: {categories[label.numpy()]}\n"
                  f"Передбачений: {categories[predicted_label]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- Основна функція ---
def main():
    """Основна логіка програми."""
    train_data, test_data = fetch_and_prepare_data()
    
    display_random_images(test_data, categories)
    
    model = build_alexnet_model(classes_count=10)
    history = model.fit(train_data, epochs=10, validation_data=test_data)
    
    plot_training_metrics(history)
    
    model.save('alexnet_trained_model.h5')
    
    assess_and_predict(model, test_data, categories)

if __name__ == "__main__":
    main()
