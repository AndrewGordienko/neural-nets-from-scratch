import matplotlib.pyplot as plt

lr = 0.8
epochs = 20
weight = 0.5
input = 2
ground_truth = 0.8

epoch_number = []
error_number = []

for epoch in range(epochs):
    value = weight * input

    delta = value - ground_truth
    error = (delta) ** 2

    weight -= lr * (value * delta)

    print(f"error {error} predicted {value}")

    epoch_number.append(epoch)
    error_number.append(error)

plt.plot(epoch_number, error_number)
plt.show()