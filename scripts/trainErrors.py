from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import sys

style.use('ggplot')

size = 200000

if(len(sys.argv) > 1):
    trainFilename = sys.argv[1]
    if(len(sys.argv) > 2):
        testFilename = sys.argv[2]
        if(len(sys.argv) > 3):
            size = int(sys.argv[3])
else:
    sys.exit('You must insert at least the file name with training error.')

x0, y0, z0 = np.loadtxt('../files/' + trainFilename + '.csv', unpack = True, delimiter = ',')
if 'testFilename' in locals():
    x1, y1, z1 = np.loadtxt('../files/' + testFilename + '.csv', unpack = True, delimiter = ',')

plt.subplot(2, 1, 1)
plt.plot(x0[:size], y0[:size], 'r-', label = "Training Error")
if 'testFilename' in locals():
    plt.plot(x1[:size], y1[:size], 'b--', label = "Test Error")
plt.title("Error curve")
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x0[:size], z0[:size], 'r-', label = "Training accuracy")
if 'testFilename' in locals():
    plt.plot(x1[:size], z1[:size], 'b--', label = "Test accuracy")
plt.title("Accuracy curve")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.legend()
plt.show()