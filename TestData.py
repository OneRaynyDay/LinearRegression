import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

arr = np.array([[1,0.7,0.5],[2,0.8,2.5],[3,0.9,3],[4,1.1,4.5],[5,1.4,4.5]])

LR = LinearRegression.LinearRegression(arr)
for i in range(10):
    LR.gradDescent(0.1)

vals = np.dot(LR.X, LR.Theta)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], c='r')
print(vals)
ax.plot(arr[:, 0], arr[:, 1], vals, c='b')
plt.show()

print(LR.Theta)