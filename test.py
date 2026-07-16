import numpy as np
import matplotlib.pyplot as plt
import __init__ as Euclid


# Function to learn
def f(x):
    return np.sin(x)#np.exp(-(x**2))


# Create training dataset
x_train = np.linspace(-10, 10, 1000).reshape(-1, 1)
y_train = f(x_train)


# Build neural network
model = Euclid.Model(
    network=Euclid.Sequential([
        Euclid.Dense(1, 16, activation=Euclid.Sigmoid()),
        Euclid.Dense(16, 32, activation=Euclid.Sigmoid()),
        Euclid.Dense(32, 1)
    ]),
    learning=Euclid.SupervisedLearning(
        loss=Euclid.MSE(),
        optimizer=Euclid.Adam(0.001)
    )
)


# Train
model.learning.train(
    x_train,
    y_train,
    epochs=20000
)


# Test predictions on training data
predictions = model.predict(x_train)

print("First 10 predictions:")
print(predictions[:10])

print("\nActual:")
print(y_train[:10])


# Plot beyond training range
x_test = np.linspace(-20, 20, 300).reshape(-1, 1)
y_test = f(x_test)

predictions_test = model.predict(x_test)


plt.figure(figsize=(10, 6))

plt.scatter(
    x_train,
    y_train,
    label="Training data",
    s=15
)

plt.plot(
    x_test,
    y_test,
    label="True f(x)",
    linewidth=2
)

plt.plot(
    x_test,
    predictions_test,
    label="NN prediction",
    linewidth=2
)

plt.axvline(
    x_train.min(),
    linestyle="--",
    label="Training boundary"
)

plt.axvline(
    x_train.max(),
    linestyle="--"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Neural Network Function Approximation")
plt.legend()
plt.grid(True)
plt.show()