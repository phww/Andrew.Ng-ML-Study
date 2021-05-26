import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt


def feature_map(x1,x2,power):
    for i in np.arange(power+1):
        for j in np.arange(power+1):
            df2['F' + str(i) + str(j)] = np.power(x1, i) * np.power(x2, j)


def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g



def costfunction(theta,X,y,):
    cost_pos = -y * np.log(sigmoid(X @ theta.T))
    cost_neg = -(1-y) * np.log(1 - sigmoid(X @ theta))
    cost = 1/len(X)*(cost_pos + cost_neg)
    return np.sum(cost)


def gradient(theta,X,y):
    h = X @ theta.T
    erro = sigmoid(h) - y
    grad = (erro.T @ X)/len(X)
    return grad


def costfunctionreg(theta,X,y,c=1):
    reg = c/len(X)*np.sum(theta[1:]**2)
    return costfunction(theta,X,y)+reg


def gradientreg(theta,X,y,c=1):
    reg = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        if i == 0:
            reg[i] = 0
        else:
            reg[i] = c/len(X)*theta[i]
    return gradient(theta,X,y) + reg



data2 = pd.read_csv("ex2data2.txt",names = ["text1","text2","quality"])
df2 = data2.copy()
x1 = df2["text1"]
x2 = df2["text2"]
x = np.array(df2.iloc[:,0:2])
# feature_map(x1,x2,power = 6)
# df2.drop("text1",axis = 1,inplace = True)
# df2.drop("text2",axis = 1,inplace = True)



col = df2.shape[1]
X = df2.iloc[:,0:col-1]
X.insert(0,'one',1)
X = np.array(X)
y = np.array(df2.iloc[:,col-1:col]).ravel()
theta = np.zeros(X.shape[1])



classifier = lm.LogisticRegression(solver='liblinear', C=100)
classifier.fit(X,y)



x_min,x_max = min(X[:,1]) - 1.0, max(X[:,1]) + 1.0
y_min,y_max = min(X[:,2]) - 1.0, max(X[:,2]) + 1.0
z_min,z_max = min(X[:,0]) - 1.0, max(X[:,0]) + 1.0
x_values,y_values = np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.1))
y_values,z_values = np.meshgrid(np.arange(y_min,y_max,0.01),np.arange(z_min,z_max,0.1))
mesh_output = classifier.predict(np.c_[x_values.ravel(),y_values.ravel(),z_values.ravel()])
mesh_output = mesh_output.reshape(x_values.shape)
plt.figure()
plt.pcolormesh(x_values,y_values,mesh_output,cmap=plt.cm.gray)
plt.scatter(pos_x_1,pos_x_2,label = 'admitted',marker = 'o')
plt.scatter(neg_x_1,neg_x_2,label = 'not admitted',marker= 'x')
plt.show()




