import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sci

data=pd.read_csv("data.txt")
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
admitted=data.loc[y==1]
not_admitted=data.loc[y==0]

#plotting the data
plt.scatter(admitted.iloc[:,0],admitted.iloc[:,1],color='red',marker='X')
plt.scatter(not_admitted.iloc[:,0],not_admitted.iloc[:,1],color='green',marker='o')
plt.show()

X=np.c_[np.ones((X.shape[0],1)),X]
y=y[:,np.newaxis]
theta=np.zeros((X.shape[1],1))

def sigmoid(x):
	return 1/(1+np.exp(-x))

def net_input(theta,x):
	return np.matmul(x,theta.transpose())

def probability(theta,x):
	return sigmoid(net_input(theta,x))

def cost_func(theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(y * np.log(probability(theta, x)) + (1 - y) * np.log(1 - probability(theta, x)))
    return total_cost

def gradient(theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)

def fit(x, y, theta):
	opt_weights=sci.fmin_tnc(func=cost_func,x0=theta,fprime=gradient,args=(x,y.flatten()))
	return opt_weights[0]

parameters = fit(X, y, theta)

print('The value of parameters are '+str(parameters))

x_values = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]
y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]
plt.plot(x_values, y_values,color='black')
plt.scatter(admitted.iloc[:,0],admitted.iloc[:,1],color='red',marker='X')
plt.scatter(not_admitted.iloc[:,0],not_admitted.iloc[:,1],color='green',marker='o')
plt.show()
