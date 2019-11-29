import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#getting the data into the program
data=pd.read_csv("ex1data1.txt")
data=np.array(data)
zero=np.zeros([np.size(data,0),1],int)
one=zero+1
x=np.array(data[:,0]).reshape(-1,1)
y=np.array(data[:,1]).reshape(-1,1)

#plotting a graph of the data
plt.scatter(x,y)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#initializing the gradient descent
iterations=1500
alpha=0.01
theta0,theta1=0,0
m=np.size(x,0)	
for i in range(iterations):
	h=theta0+theta1*x	#hypothesis
	temp0=theta0-alpha*((1/m)*sum(h-y))
	temp1=theta1-alpha*(1/m)*sum((h-y)*x)
	theta0,theta1=temp0,temp1
print("the value of theta0:"+str(theta0)+" and theta1:"+str(theta1))

#plotting graph with out fitted model
plt.scatter(x,y)
plt.plot([min(x),max(x)],[min(h),max(h)],color='red') #regression line
plt.show()




	


