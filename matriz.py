import numpy

x = numpy.random.randint(1, 10, size=(8, 8))

weights = numpy.random.randint(1, 10, size=(64, 8*8))
biases = numpy.random.randint(1, 10, size=64)

def floor(x):
    return numpy.floor(x)

def predict(x):
    x = x.flatten()
    y = numpy.dot(weights, x) + biases
    return floor(y).astype(int).reshape(8,8)

prediction = predict(x)

print("Matriz original:\n", x)
print("Predicción de la matriz por la red neuronal:\n", prediction)