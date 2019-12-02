import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

data=pd.read_csv("data2.txt")
nor_data=(data-data.mean())/data.std()
nor_data=np.array(nor_data)
x=np.array(nor_data[:,0:2])
y=np.array(nor_data[:,2])
x_one=np.append(np.ones([np.size(x,0),1]),x,axis=1)

#plotting the data
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.scatter3D(x[:,0],x[:,1],y)
plt.show()

#initializing gradient descent
iterations=1500
alpha=0.01
m=np.size(data,0)
y=y.reshape(-1,1)
theta=np.zeros([np.size(x,1)+1,1])
temp_theta=np.zeros([np.size(x,1)+1,1])
for i in range(iterations):
	h=np.inner(theta.transpose(),x_one).transpose()
	temp_theta=theta-alpha*(1/m)*np.inner(x_one.transpose(),(h-y).transpose())
	theta=temp_theta

print("the values of theta are "+str(theta))

y_pred=np.inner(theta.transpose(),x_one).transpose()
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.scatter3D(x[:,0],x[:,1],y,color='green')
ax.scatter3D(x[:,0],x[:,1],y_pred,color='red')
plt.show()


	
