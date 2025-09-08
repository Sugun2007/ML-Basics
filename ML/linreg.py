import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data.csv")

# Loss function (Mean Squared Error)
def loss_fn(m, b, points):
    total_loss = 0
    for i in range(len(points)):
        x = points.iloc[i].study_time
        y = points.iloc[i].marks
        total_loss += (y - (m * x + b)) ** 2
    return total_loss / float(len(points))

# Gradient Descent Step
def gradient_des(m_now, b_now, points, l):
    m_grad = 0
    b_grad = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].study_time
        y = points.iloc[i].marks
        m_grad += -(2/n) * x * (y - (m_now * x + b_now))
        b_grad += -(2/n) * (y - (m_now * x + b_now))
    m = m_now - l * m_grad
    b = b_now - l * b_grad
    return m, b

# Initialize parameters
m = 0
b = 0
l = 0.01   # learning rate
eph = 300  # epochs

# Training loop
for i in range(eph):
    m, b = gradient_des(m, b, data, l)
    if i % 50 == 0:
        print(f"Epoch {i}, Loss: {loss_fn(m, b, data)}")

print("Final slope (m):", m)
print("Final intercept (b):", b)

# Plot data and fitted line
plt.scatter(data.study_time, data.marks, color="black")
plt.plot(list(np.arange(0, 1, 0.1)), [m*x+b for x in np.arange(0, 1, 0.1)], color="red")
plt.xlabel("Study Time")
plt.ylabel("Marks")
plt.show()
