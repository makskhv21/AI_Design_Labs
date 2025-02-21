import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from PIL import Image, ImageDraw
import tkinter as tk

# Оголошення назв класів
class_labels = [str(i) for i in range(10)]

# Завантаження та обробка даних
def load_and_process_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_labels, test_labels = to_categorical(train_labels, 10), to_categorical(test_labels, 10)
    return (train_images, train_labels), (test_images, test_labels)

# Функція для показу випадкових зображень
def show_random_images(images, labels, count=25):
    plt.figure(figsize=(10, 10))
    random_indices = np.random.choice(images.shape[0], count, replace=False)
    for i, idx in enumerate(random_indices):
        plt.subplot(5, 5, i + 1)
        display_image(images[idx], labels[idx])
    plt.show()

# Функція для відображення одного зображення
def display_image(image, label):
    plt.xticks([]) 
    plt.yticks([]) 
    plt.grid(False) 
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_labels[np.argmax(label)])

# Створення моделі
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Процес навчання моделі
def train_neural_network(model, train_images, train_labels, epochs=50, batch_size=100):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    training_history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[early_stopping])
    return training_history

# Побудова графіків для тренувальних метрик
def plot_metrics(training_history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(training_history.history['loss'], label='Train Loss')
    plt.plot(training_history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Progress')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(training_history.history['accuracy'], label='Train Accuracy')
    plt.plot(training_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Progress')
    plt.legend()

    plt.show()

# Оцінка точності моделі
def assess_model(model, test_images, test_labels):
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Loss on test data: {test_loss:.4f}, Accuracy on test data: {test_accuracy:.4f}")

# Перевірка передбачень моделі
def verify_predictions(model, test_images, test_labels, sample_count=5):
    random_indices = np.random.choice(test_images.shape[0], sample_count, replace=False)
    for idx in random_indices:
        display_image(test_images[idx], np.argmax(test_labels[idx]))
        prediction = model.predict(test_images[idx:idx+1])
        plt.title(f'Predicted: {np.argmax(prediction)}')
        plt.show()

# Прогнозування цифри з нового зображення
def predict_digit_from_image(model, img):
    img = img.resize((28, 28)).convert('L')
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28)
    prediction = np.argmax(model.predict(img))
    return prediction

# GUI для розпізнавання цифр
def run_digit_recognition_gui(model):
    root_window = tk.Tk()
    root_window.title("Digit Recognition")

    canvas = tk.Canvas(root_window, width=280, height=280, bg='black')
    canvas.grid(row=0, column=0, columnspan=2)

    img_canvas = Image.new('RGB', (280, 280), 'black')
    drawing_canvas = ImageDraw.Draw(img_canvas)

    def draw_on_canvas(event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        drawing_canvas.ellipse([x1, y1, x2, y2], fill='white', outline='white')

    def reset_canvas():
        drawing_canvas.rectangle((0, 0, 280, 280), fill='black')
        canvas.delete("all")

    def predict_digit():
        digit = predict_digit_from_image(model, img_canvas)
        result_label.config(text=f"Recognized: {digit}")

    canvas.bind("<B1-Motion>", draw_on_canvas)

    tk.Button(root_window, text="Recognize", command=predict_digit).grid(row=1, column=0)
    tk.Button(root_window, text="Clear", command=reset_canvas).grid(row=1, column=1)

    result_label = tk.Label(root_window, text="Recognized: ")
    result_label.grid(row=2, column=0, columnspan=2)

    root_window.mainloop()

# Основна логіка програми
if __name__ == "__main__":
    # Завантаження і обробка даних
    (train_images, train_labels), (test_images, test_labels) = load_and_process_data()

    # Показати випадкові зображення
    show_random_images(train_images, train_labels)

    # Створення та компіляція моделі
    neural_network_model = create_model()
    neural_network_model.summary()

    # Навчання моделі
    history = train_neural_network(neural_network_model, train_images, train_labels)

    # Показ графіків результатів тренування
    plot_metrics(history)

    # Оцінка результатів на тестових даних
    assess_model(neural_network_model, test_images, test_labels)

    # Перевірка передбачень
    verify_predictions(neural_network_model, test_images, test_labels)

    # Запуск GUI для розпізнавання цифр
    run_digit_recognition_gui(neural_network_model)
