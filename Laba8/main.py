import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer
import requests
import tarfile
import os

# Константи для конфігурації
DATA_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
LOCAL_TAR_PATH = "LJSpeech-1.1.tar.bz2"
EXTRACT_DIR = "./LJSpeech-1.1"
AUDIO_DIR = EXTRACT_DIR + "/wavs/"
METADATA_PATH = EXTRACT_DIR + "/metadata.csv"
FRAME_LENGTH = 256
FRAME_STEP = 160
FFT_LENGTH = 384
BATCH_SIZE = 32

# Завантаження та розпакування датасету
def download_and_extract_dataset():
    """Завантажує та розпаковує датасет LJSpeech, якщо його ще немає."""
    if not os.path.exists(LOCAL_TAR_PATH):
        print("Завантаження датасету LJSpeech...")
        response = requests.get(DATA_URL, stream=True)
        with open(LOCAL_TAR_PATH, "wb") as f:
            f.write(response.content)
        print("Завантаження завершено.")

    if not os.path.exists(EXTRACT_DIR):
        print("Розпакування датасету...")
        with tarfile.open(LOCAL_TAR_PATH, "r:bz2") as tar:
            tar.extractall(path=".")
        print("Розпакування завершено.")

# Підготовка метаданих
def prepare_metadata():
    """Читає метадані, перемішує їх і розділяє на тренувальну та валідаційну вибірки."""
    df = pd.read_csv(METADATA_PATH, sep="|", header=None, quoting=3)
    df.columns = ["audio_file", "transcription", "normalized_transcript"]
    df = df[["audio_file", "normalized_transcript"]]
    df = df.sample(frac=1).reset_index(drop=True)
    
    split_idx = int(len(df) * 0.90)
    return df[:split_idx], df[split_idx:]

# Перетворення символів у числа
CHARACTERS = list("abcdefghijklmnopqrstuvwxyz'?! ")
char_to_num = keras.layers.StringLookup(vocabulary=CHARACTERS, oov_token="")
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# Обробка одного зразка
def process_sample(audio_file, transcript):
    """
    Обробляє один аудіофайл та його транскрипцію для моделі.
    
    Args:
        audio_file (str): Базове ім'я аудіофайлу (без розширення).
        transcript (str): Текст транскрипції.
    
    Returns:
        tuple: (спектограма, числова транскрипція).
    """
    # Читання та декодування аудіо
    file_path = AUDIO_DIR + audio_file + ".wav"
    audio_content = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_content)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)
    
    # Обчислення спектограми
    spectrogram = tf.signal.stft(audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, fft_length=FFT_LENGTH)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    
    # Нормалізація по частотних бін
    means = tf.math.reduce_mean(spectrogram, axis=1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, axis=1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    
    # Обробка транскрипції
    transcript = tf.strings.lower(transcript)
    transcript = tf.strings.unicode_split(transcript, input_encoding="UTF-8")
    transcript = char_to_num(transcript)
    
    return spectrogram, transcript

# Створення датасетів
def create_datasets(train_df, val_df):
    """Створює тренувальний та валідаційний датасети з метаданих."""
    train_dataset = tf.data.Dataset.from_tensor_slices((train_df["audio_file"].tolist(), train_df["normalized_transcript"].tolist()))
    train_dataset = (
        train_dataset.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_df["audio_file"].tolist(), val_df["normalized_transcript"].tolist()))
    val_dataset = (
        val_dataset.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    return train_dataset, val_dataset

# Візуалізація зразка (опціонально)
def visualize_sample(dataset, audio_dir):
    """Візуалізує спектограму та звукову хвилю одного зразка з датасету."""
    for batch in dataset.take(1):
        spectrogram = batch[0][0].numpy()
        spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
        label = tf.strings.reduce_join(num_to_char(batch[1][0])).numpy().decode("utf-8")
        
        plt.figure(figsize=(8, 5))
        ax1 = plt.subplot(2, 1, 1)
        ax1.imshow(spectrogram, vmax=1)
        ax1.set_title(label)
        ax1.axis("off")
        
        audio_file = audio_dir + train_df["audio_file"].iloc[0] + ".wav"
        audio_content = tf.io.read_file(audio_file)
        audio, _ = tf.audio.decode_wav(audio_content)
        audio = audio.numpy().flatten()
        
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(audio)
        ax2.set_title("Звукова хвиля")
        ax2.set_xlim(0, len(audio))
        
        display.display(display.Audio(audio, rate=16000))
        plt.show()

# CTC втрати
def ctc_loss(y_true, y_pred):
    """
    Обчислює CTC (Connectionist Temporal Classification) втрати для задачі розпізнавання мови.
    
    Args:
        y_true: Реальні мітки.
        y_pred: Передбачені логіти моделі.
    
    Returns:
        float: Значення CTC втрати.
    """
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    
    return keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

# Побудова моделі
def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    """
    Створює модель, подібну до DeepSpeech2, для розпізнавання мови.
    
    Args:
        input_dim (int): Кількість частотних бін у спектограмі.
        output_dim (int): Розмір словника (кількість символів).
        rnn_layers (int): Кількість рекурентних шарів.
        rnn_units (int): Кількість одиниць у кожному рекурентному шарі.
    
    Returns:
        keras.Model: Скомпільована модель.
    """
    inputs = layers.Input(shape=(None, input_dim), name="input_spectrogram")
    x = layers.Reshape((-1, input_dim, 1), name="expand_dims")(inputs)
    
    # Згорткові шари
    x = layers.Conv2D(32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False, name="conv_1")(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    
    x = layers.Conv2D(32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False, name="conv_2")(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    
    # Переформатування для рекурентних шарів
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]), name="reshape_for_rnn")(x)
    
    # Рекурентні шари
    for i in range(1, rnn_layers + 1):
        gru = layers.GRU(rnn_units, return_sequences=True, name=f"gru_{i}")
        x = layers.Bidirectional(gru, name=f"bidirectional_{i}")(x)
        if i < rnn_layers:
            x = layers.Dropout(0.5, name=f"dropout_{i}")(x)
    
    # Щільні шари
    x = layers.Dense(rnn_units * 2, activation="relu", name="dense")(x)
    x = layers.Dropout(0.5, name="dropout_dense")(x)
    outputs = layers.Dense(output_dim + 1, activation="softmax", name="output")(x)
    
    model = keras.Model(inputs, outputs, name="DeepSpeech_2")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=ctc_loss)
    return model

# Основний потік виконання
if __name__ == "__main__":
    # Перевірка доступності GPU
    print("----- Запуск -----")
    
    # Завантаження та підготовка даних
    download_and_extract_dataset()
    train_df, val_df = prepare_metadata()
    train_dataset, validation_dataset = create_datasets(train_df, val_df)
    
    # Візуалізація (опціонально)
    visualize_sample(train_dataset, AUDIO_DIR)
    
    # Побудова моделі
    input_dim = FFT_LENGTH // 2 + 1  # Кількість частотних бін
    output_dim = char_to_num.vocabulary_size()  # Розмір словника
    model = build_model(input_dim, output_dim, rnn_units=512)
    model.summary()