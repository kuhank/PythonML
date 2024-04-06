# CNN

import numpy as np
import tensorflow as tf

# Generate some example image data
n_samples = 1000
img_size = 28
n_channels = 1

# Generate random images
X = np.random.randn(n_samples, img_size, img_size, n_channels)

# Generate random labels (binary classification)
y = np.random.randint(0, 2, size=(n_samples,))

# Split data into training and testing sets
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define the CNN model
model_cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, n_channels)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_cnn.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model_cnn.evaluate(X_test, y_test)
print("CNN Test Loss:", loss)
print("CNN Test Accuracy:", accuracy)


# RNN
# Generate some example text data
text_corpus = ["hello world", "good morning", "have a nice day"]
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(text_corpus)
sequences = tokenizer.texts_to_sequences(text_corpus)
X = tf.keras.preprocessing.sequence.pad_sequences(sequences)

# Generate random labels (binary classification)
y = np.random.randint(0, 2, size=(len(text_corpus),))

# Split data into training and testing sets
split = int(0.8 * len(text_corpus))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define the RNN model
model_rnn = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_rnn.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model_rnn.evaluate(X_test, y_test)
print("RNN Test Loss:", loss)
print("RNN Test Accuracy:", accuracy)
