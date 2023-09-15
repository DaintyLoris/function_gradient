import tensorflow as tf
import numpy as np


# Definiendo la funcion f(x1, x2)
def custom_function(x):
    return 10 - tf.exp(-x[:, 0] ** 2 + 3 * x[:, 1] ** 2)


#Crenado el dataset con valores aleatorios
X = np.random.rand(100, 2)  # 100 data points con 2 caracteristicas
y = custom_function(X)

# DDefiniendo un perceptron multicapa
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Capa de salida con una neurona
])

# Compilando el modelo
model.compile(optimizer='adam', loss='mse')  # Usando "mean squared error loss"

# Entrenando el modelo
model.fit(X, y, epochs=100, verbose=2)

#Obteniendo los gradientes de la salida del modelo con respecto a las entradas
with tf.GradientTape() as tape:
    inputs = tf.constant(X, dtype=tf.float32)
    tape.watch(inputs)
    predictions = model(inputs)

gradients = tape.gradient(predictions, inputs)

#Imprimiendo los gradientes
print("Gradients:")
print(gradients)
