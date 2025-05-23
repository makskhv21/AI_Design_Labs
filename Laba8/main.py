import os
import sys
import requests
import tarfile
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer

# ### Конфігурація
DATASET_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
DATASET_DIR = "LJSpeech-1.1"
WAVS_DIR = os.path.join(DATASET_DIR, "wavs")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")
BATCH_SIZE = 32
EPOCHS = 8
RNN_UNITS = 512
FRAME_LENGTH = 256
FRAME_STEP = 160
FFT_LENGTH = 384

# ### Функція для завантаження та розпакування датасету
def download_and_extract_dataset():
    """Завантажує та розпаковує датасет LJSpeech-1.1, якщо його ще немає."""
    if not os.path.exists(DATASET_DIR):
        print(f"Директорія {DATASET_DIR} не знайдена. Завантажуємо датасет...")
        local_filename = "LJSpeech-1.1.tar.bz2"
        try:
            print("Завантаження датасету...")
            response = requests.get(DATASET_URL, stream=True)
            response.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Датасет завантажено.")
        except requests.exceptions.RequestException as e:
            print(f"Помилка завантаження: {e}")
            sys.exit(1)
        try:
            print("Розпакування датасету...")
            with tarfile.open(local_filename, "r:bz2") as tar:
                tar.extractall(path=".")
            print("Датасет розпаковано.")
        except tarfile.TarError as e:
            print(f"Помилка розпакування: {e}")
            sys.exit(1)
        os.remove(local_filename)
    else:
        print(f"Датасет знайдено в {DATASET_DIR}.")

# ### Завантаження датасету
download_and_extract_dataset()

# ### Читання та підготовка метаданих
metadata_df = pd.read_csv(METADATA_PATH, sep="|", header=None, quoting=3)
metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
metadata_df = metadata_df[["file_name", "normalized_transcription"]]
metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

# Додавання повного шляху до файлів аудіо
metadata_df["file_path"] = metadata_df["file_name"].apply(lambda x: os.path.join(WAVS_DIR, x + ".wav"))

# Розділення на тренувальний і валідаційний набори
split_index = int(len(metadata_df) * 0.90)
train_df = metadata_df[:split_index]
val_df = metadata_df[split_index:]

# ### Визначення набору символів для кодування
characters = list("abcdefghijklmnopqrstuvwxyz'?! ")
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# ### Функція для кодування одного зразка
def encode_single_sample(file_path, transcription):
    """Кодує аудіофайл у спектрограму та транскрипцію у числовий формат."""
    # Читання аудіофайлу
    file = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)
    
    # Обчислення спектрограми
    spectrogram = tf.signal.stft(audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, fft_length=FFT_LENGTH)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    
    # Нормалізація спектрограми
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    
    # Кодування тексту
    transcription = tf.strings.lower(transcription)
    transcription = tf.strings.unicode_split(transcription, input_encoding="UTF-8")
    transcription = char_to_num(transcription)
    
    return spectrogram, transcription

# ### Створення датасетів
train_dataset = tf.data.Dataset.from_tensor_slices((train_df["file_path"], train_df["normalized_transcription"]))
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val_dataset = tf.data.Dataset.from_tensor_slices((val_df["file_path"], val_df["normalized_transcription"]))
val_dataset = (
    val_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# ### Візуалізація одного зразка
fig, ax = plt.subplots(2, 1, figsize=(8, 5))
for batch in train_dataset.take(1):
    spectrogram, label = batch[0][0], batch[1][0]
    spectrogram = spectrogram.numpy()
    spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
    label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
    
    ax[0].imshow(spectrogram, vmax=1)
    ax[0].set_title(f"Спектрограма: {label}")
    ax[0].axis("off")
    
    file_path = train_df["file_path"].iloc[0]
    file = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(file)
    audio = audio.numpy()
    
    ax[1].plot(audio)
    ax[1].set_title("Звукова хвиля")
    ax[1].set_xlim(0, len(audio))
    display.display(display.Audio(np.transpose(audio), rate=16000))

plt.savefig('sample_visualization.png')

# ### Функція втрат CTC
def ctc_loss(y_true, y_pred):
    """Обчислює втрати CTC для батчу."""
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.math.count_nonzero(y_true, axis=1, keepdims=True)
    label_length = tf.cast(label_length, dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# ### Побудова моделі DeepSpeech2
def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=RNN_UNITS):
    """Створює модель, подібну до DeepSpeech2."""
    input_spectrogram = layers.Input((None, input_dim), name="input")
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    
    # Перший згортковий шар
    x = layers.Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False, name="conv_1")(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    
    # Другий згортковий шар
    x = layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False, name="conv_2")(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    
    # Перетворення до послідовності
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    
    # RNN шари
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(units=rnn_units, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, reset_after=True, name=f"gru_{i}")
        x = layers.Bidirectional(recurrent, name=f"bidirectional_{i}", merge_mode="concat")(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
    
    # Повнозв'язний шар
    x = layers.Dense(units=rnn_units * 2)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(rate=0.5)(x)
    
    # Вихідний шар
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=ctc_loss)
    return model

# ### Створення моделі
model = build_model(input_dim=FFT_LENGTH // 2 + 1, output_dim=char_to_num.vocabulary_size())
model.summary(line_length=150)

# ### Функція декодування передбачень
def decode_batch_predictions(pred):
    """Декодує передбачення моделі у текст."""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = [tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8") for result in results]
    return output_text

# ### Callback для оцінки після кожної епохи
class EvaluationCallback(keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model.predict(X, verbose=0)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 4):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)

# ### Тренування моделі
validation_callback = EvaluationCallback(val_dataset)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[validation_callback],
)

# ### Тестування моделі
predictions = []
targets = []
for batch in val_dataset:
    X, y = batch
    batch_predictions = model.predict(X, verbose=0)
    batch_predictions = decode_batch_predictions(batch_predictions)
    predictions.extend(batch_predictions)
    for label in y:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        targets.append(label)

wer_score = wer(targets, predictions)
print("-" * 100)
print(f"Word Error Rate: {wer_score:.4f}")
print("-" * 100)
for i in np.random.randint(0, len(predictions), 8):
    print(f"Target    : {targets[i]}")
    print(f"Prediction: {predictions[i]}")
    print("-" * 100)