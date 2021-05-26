import numpy as np
import pandas as pd
def normalEqn(X,y):
    inner = X.T*X
    theta = inner.I*X.T*y
    return theta
"""单特征数据"""
data1 = pd.read_csv("ex1data1.txt",names = ['population','profit'])
x = data1.population
y = data1.profit
df = data1.copy()
df.insert(0,"one",1)
X = df.iloc[:,0:df.shape[1]-1]
y = df.iloc[:,df.shape[1]-1:df.shape[1]]
y = np.matrix(y)
X = np.matrix(X)
x = np.matrix(x)
theta1 = normalEqn(X,y)
print('theta1:','\n',theta1)

"""多特征数据"""
data2 = pd.read_csv('ex1data2.txt',names=['square', 'bedrooms', 'price'])

"""特征缩放"""
x_2 = data2.iloc[:,0:data2.shape[1]-1]
y_2 = data2.iloc[:,data2.shape[1]-1:]
x_2 = (x_2 - np.average(x_2,axis = 0))/np.std(x_2,axis = 0,ddof = 1)#ddof = 1,有偏和无偏？
y_2 = (y_2 - np.average(y_2,axis = 0))/np.std(y_2,axis = 0,ddof = 1)


X_2 = x_2.copy()
X_2.insert(0,'one',1)
X_2 = np.matrix(X_2)
y_2 = np.matrix(y_2)
theta2 = normalEqn(X_2,y_2)
print('theta2:','\n',theta2)






