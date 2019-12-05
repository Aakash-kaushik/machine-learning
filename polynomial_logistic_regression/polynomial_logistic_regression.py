import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sci
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_csv("data2.txt")
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
admitted=data.loc[y==1]
not_admitted=data.loc[y==0]

#plotting the data
plt.scatter(admitted.iloc[:,0],admitted.iloc[:,1],color='red',marker='X')
plt.scatter(not_admitted.iloc[:,0],not_admitted.iloc[:,1],color='green',marker='o')
plt.show()

#feature mapping
pol=PolynomialFeatures(degree=6)
x_pol=pol.fit_transform(X)

x_pol=np.c_[np.ones((x_pol.shape[0],1)),x_pol]
model=LogisticRegression()
model.fit(x_pol,y)
predicted= model.predict(x_pol)
accuracy = accuracy_score(y,predicted)
theta=model.coef_
accuracy=accuracy*100
print("the calculated values of theta are "+str(theta))
print(accuracy)






