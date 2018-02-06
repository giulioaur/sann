import os
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import sys
import math

def shouldRead(filename):
    i = 0
    for line in open(filename):
        i = i + 1
        if(i > 1):
            return True
    return False


style.use('ggplot')
lastEpoch = 2000000
figures = 1

if(len(sys.argv) >= 2):
    figures = int(sys.argv[1])
if(len(sys.argv) >= 3):
    lastEpoch = int(sys.argv[2])

# Retrieve all the directory of validation
mainDir = '../files/validation'
dirs = os.listdir(mainDir)

sizeC = int(math.ceil(math.sqrt(len(dirs) / figures)))
sizeR = sizeC #len(dirs) / sizeC + 2
step = len(dirs) / figures

# Support multiple plot
for f in range(figures):
    i = 0
    plt.figure(f)
    fig, axarr = plt.subplots(sizeR, sizeC)
    end = (f + 1) * step if f < figures - 1 else len(dirs)

    for k in range(f * step, end):
        fullDirName = mainDir + '/' + dirs[k]
        if os.path.isdir(fullDirName):
            row = int(i / sizeR)
            col = int(i % sizeC)
            mean = np.array([])
            epoch = []
            files = os.listdir(fullDirName)
            #axarr[row, col].set_ylim(0, 1)
        
            # For each files in the directory plot all the graph and compute the mean
            for filename in files:
                fullname = fullDirName + '/' + filename
                if shouldRead(fullname):
                    epoch, y, z = np.loadtxt(fullname, unpack = True, delimiter = ',')
                    
                    if mean.size == 0:
                        mean = np.zeros(epoch.size)
                    
                    if '.vd' not in filename:
                        axarr[row, col].plot(epoch[:lastEpoch], y[:lastEpoch], '-', color='xkcd:light red')
                    else:
                        axarr[row, col].plot(epoch[:lastEpoch], y[:lastEpoch], '-', color='xkcd:light blue')
                    
                    # mean += np.array(y)
            
            # Plot the mean
            # mean /= len(files)
            # axarr[row, col].plot(epoch, mean, '-', color='xkcd:crimson')
            axarr[row, col].set_title(dirs[k], fontsize=6)
            i += 1
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4, wspace=0.2, left=0.03, right=0.98, top=0.97, bottom=0.04)

plt.show()