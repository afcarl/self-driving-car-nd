from numpy import genfromtxt
import matplotlib.pyplot as plt
per_data=genfromtxt('highway_map.csv',delimiter=' ')
plt.xlabel ('x')
plt.ylabel ('y')
x = per_data[:,0]
y = per_data[:,1]
plt.plot(x,y, 'ro')
plt.plot(x[0], y[0], 'bo')
plt.plot(x[180], y[180], 'go')
plt.plot()
plt.show()
