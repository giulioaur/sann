from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

px, py, tx, ty = np.loadtxt('../files/points.csv', unpack = True, delimiter = ',')

plt.scatter(px, py, color='red', label="My result")
plt.scatter(tx, ty, color='blue', label="True result")

plt.xlabel('x')
plt.ylabel('y')

plt.legend()
plt.show()