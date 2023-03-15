import numpy
import matplotlib.pyplot as plt

x = numpy.zeros((6, 6))
for i in range(6):
    for j in range(6):
        x[i][j] = float(input(f"Ingrese el elemento ({i+1},{j+1}): "))

y = numpy.linalg.inv(x)

weights = numpy.random.randint(1, 100, size=(36, 6*6)).astype(float)
biases = numpy.random.randint(1, 100, size=36).astype(float)

def floor(x):
    return numpy.floor(x)

def predict(x):
    x = x.flatten()
    y = numpy.dot(weights, x) + biases
    return floor(y).astype(int).reshape(6,6)

def loss(y_pred, y_true):
    return numpy.sum((y_pred - y_true) ** 2)

learning_rate = 0.0001
num_iterations = 1207

losses = []
for i in range(num_iterations):

    y_pred = predict(x)

    l = loss(y_pred, y)
    losses.append(l)

    d_weights = 2 * numpy.outer((y_pred - y).flatten(), x.flatten())
    d_biases = 2 * (y_pred - y).flatten()

    weights -= learning_rate * d_weights.reshape(weights.shape)
    biases -= learning_rate * d_biases

    if i % 1000 == 0:
        print(f"Iteration {i}, loss: {l}")

print("Matriz original:\n", x)
print("Inversa de la matriz (calculada por la red neuronal):\n", predict(x))
print("Inversa de la matriz (calculada por numpy):\n", y)

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()