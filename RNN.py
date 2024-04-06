import numpy as np
import tensorflow as tf

# Generate some example data
n_samples = 1000
time_steps = 10
input_dim = 1
output_dim = 1

# Generate input data
X = np.random.randn(n_samples, time_steps, input_dim)

# Generate output data (simple addition of input)
y = np.sum(X, axis=1)

# Split data into training and testing sets
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define the RNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=(time_steps, input_dim), return_sequences=False),
    tf.keras.layers.Dense(output_dim)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
