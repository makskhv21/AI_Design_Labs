import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models
from tensorflow.keras.layers import LSTM, Dense, Embedding
import matplotlib.pyplot as plt
import numpy as np

# Тестові речення для перевірки моделі
test_sentences = [
    "The customer support was exceptional, they resolved my issue quickly.",  # Позитивне
    "The storyline was interesting but had some predictable moments.",         # Нейтральне
    "The software update introduced more bugs than fixes, very disappointing.", # Негативне
    "The concert was an unforgettable experience, pure magic!",               # Позитивне
    "The book had a decent pace, but nothing too exciting happened."           # Нейтральне
]

def plot_accuracy_history(history):
    """Малює графіки точності тренування та валідації на основі об'єкта історії."""
    plt.plot(history.history['accuracy'], label='Accuracy', color='green')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.xlabel('Епоха')
    plt.ylabel('Точність')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

class SentimentAnalysisModel:
    """Клас для створення, тренування та передбачення моделі аналізу настроїв."""

    def __init__(self, vocab_size=20000, max_sequence_length=200):
        """Ініціалізує модель із розміром словника та максимальною довжиною послідовності."""
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
        self.model = None

    def fit_tokenizer(self, texts):
        """Навчає токенізатор на наданих текстах."""
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.tokenizer.fit_on_texts(texts)

    def transform_texts(self, texts):
        """Перетворює тексти в послідовності з вирівнюванням довжини."""
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        return padded_sequences

    def create_model(self):
        """Створює та компілює модель аналізу настроїв."""
        self.model = models.Sequential([
            Embedding(self.vocab_size, 8),
            LSTM(16),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        self.model.compile(optimizer="RMSprop", loss="binary_crossentropy", metrics=["accuracy"])

    def train_model(self, train_texts, train_labels, test_texts, test_labels, epochs=10):
        """Тренує модель на тренувальних даних і валідує на тестових."""
        train_sequences = self.transform_texts(train_texts)
        test_sequences = self.transform_texts(test_texts)
        history = self.model.fit(
            train_sequences, train_labels,
            epochs=epochs,
            validation_data=(test_sequences, test_labels)
        )
        return history

    def predict_sentiment(self, sentences):
        """Передбачає настрій для наданих речень і виводить результати."""
        for sentence in sentences:
            print(sentence)
            input_seq = self.transform_texts([sentence])
            prediction = self.model.predict(input_seq, verbose=0)
            prediction_score = prediction[0][0]
            if prediction_score < 0.45:
                sentiment = 'Negative'
            elif prediction_score > 0.85:
                sentiment = 'Positive'
            else:
                sentiment = 'Neutral'
            print(f"Prediction: {sentiment} ({prediction_score:.2f})")

def main():
    """Основна функція для запуску аналізу настроїв."""
    # Завантаження даних
    data = tfds.load("yelp_polarity_reviews", as_supervised=True)
    train_set, test_set = data['train'], data['test']

    # Підготовка тренувальних даних
    train_texts, train_labels = [], []
    for element in train_set:
        train_texts.append(element[0].numpy().decode())
        train_labels.append(int(element[1].numpy()))

    # Підготовка тестових даних
    test_texts, test_labels = [], []
    for element in test_set:
        test_texts.append(element[0].numpy().decode())
        test_labels.append(int(element[1].numpy()))

    # Перетворення міток у масиви NumPy
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Ініціалізація та тренування моделі
    model = SentimentAnalysisModel()
    model.fit_tokenizer(train_texts)
    model.create_model()
    history = model.train_model(train_texts, train_labels, test_texts, test_labels)

    # Візуалізація результатів тренування
    plot_accuracy_history(history)

    # Передбачення настроїв для тестових речень
    model.predict_sentiment(test_sentences)

if __name__ == "__main__":
    main()