import tensorflow as tf
import numpy as np

# XOR data generation
def generate_xor_data():
    X = []
    Y = []
    for i in range(16):
        inputs = [int(x) for x in format(i, '04b')]
        X.append(inputs)
        result = inputs[0] ^ inputs[1] ^ inputs[2] ^ inputs[3]
        Y.append([result])
    return np.array(X), np.array(Y)

# Model creation
X, Y = generate_xor_data()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_dim=4, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
model.fit(X, Y, epochs=1000, batch_size=8, verbose=1)

# Model evaluation
accuracy = model.evaluate(X, Y)
print(f'✅ Model accuracy: {accuracy[1]*100:.2f}%\n')

# Check results
predictions = model.predict(X)
binary_predictions = (predictions > 0.5).astype(int)

print("Predictions for input data:\n")
print(" Input data  | Prediction | Expected result | Status")
print("-" * 60)
for i in range(len(X)):
    match_status = "✅" if binary_predictions[i][0] == Y[i][0] else "❌"
    print(f' {X[i]}    |    {binary_predictions[i][0]}      |          {Y[i][0]}      |     {match_status}')
print("-" * 60)