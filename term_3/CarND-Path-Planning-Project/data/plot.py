from numpy import genfromtxt
import matplotlib.pyplot as plt
per_data=genfromtxt('logger.csv',delimiter=',')
plt.figure(1)
plt.xlabel ('x')
plt.ylabel ('y')
x = per_data[:,0]
y = per_data[:,1]
plt.plot(x,y, 'ro')
plt.show()

plt.figure(2)
plt.xlabel ('x')
plt.ylabel ('y')
x = per_data[:,2]
y = per_data[:,3]
plt.plot(x,y, 'ro')
#plt.plot(x, 'ro')
plt.plot()
plt.show()
