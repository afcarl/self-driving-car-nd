from numpy import genfromtxt
import matplotlib.pyplot as plt
per_data=genfromtxt('lake_track_waypoints.csv',delimiter=',')
plt.xlabel ('x')
plt.ylabel ('y')
x = per_data[:,0]
y = per_data[:,1]
plt.plot(x,y, 'ro')
plt.show()
