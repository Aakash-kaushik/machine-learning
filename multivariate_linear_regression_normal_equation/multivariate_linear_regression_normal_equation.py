import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

data=pd.read_csv("data2.txt")
nor_data=data#(data-data.mean())/data.std()
nor_data=np.array(nor_data)
x=np.array(nor_data[:,0:2])
y=np.array(nor_data[:,2])
x_one=np.append(np.ones([np.size(x,0),1]),x,axis=1)

#plotting the data
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.scatter3D(x[:,0],x[:,1],y)
plt.show()

#solving with normal equation
y=y.reshape(-1,1)
theta=np.zeros([np.size(x,1)+1,1])
theta=np.matmul(np.matmul(np.linalg.inv(np.matmul(x_one.transpose(),x_one)),x_one.transpose()),y)


print("the values of theta are "+str(theta))

y_pred=np.inner(theta.transpose(),x_one).transpose()
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.scatter3D(x[:,0],x[:,1],y,color='green')
ax.scatter3D(x[:,0],x[:,1],y_pred,color='red')
plt.show()


	
