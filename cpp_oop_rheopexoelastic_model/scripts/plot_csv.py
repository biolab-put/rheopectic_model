import numpy as np
from matplotlib import pyplot as plt

filename = 'estimated_force.csv'

data = []
with open(filename,'r') as csvfile:
    for line in csvfile.readlines():
        value = line.rstrip().split(',')
        data.append(np.float32(value))
data_r = data[0]
#print(data_r)
plt.plot(data_r)
plt.show()
