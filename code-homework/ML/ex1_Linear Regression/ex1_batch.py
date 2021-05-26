import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("ex1data1.txt",names = ['population','profit'])
x = data.population
y = data.profit
"初始化,所有变量都是ｍａｔｒｉｘ"
df = data.copy()#因为ｉｎｓｅｒｔ会改变原数组，所以先复制一份，坑１．
df.insert(0,"one",1)
X = df.iloc[:,0:df.shape[1]-1]
y = df.iloc[:,df.shape[1]-1:df.shape[1]]#df.iloc[:,-1]是个一维数组(series)，ｒｅｓｈａｐｅ（９７，１）都不行，坑２．
theta = np.zeros(X.shape[1])
y = np.matrix(y)
X = np.matrix(X)
x = np.matrix(x)
x = x.T #行向量／列向量傻傻分不清　坑３
theta = np.matrix(theta)
H = X*(theta.T)
"""计算代价"""
def costfunction(X,y,H):
    n = np.power((H-y),2)
    return np.sum(n)/(2*len(X))
"""批量梯度下降"""
alpha = 0.01
m = len(X)
times = 1000
def gradient_descent(theta,X,y,alpha,m,H,times):
    thetas_0 = [0]
    thetas_1 = [0]
    cost = [costfunction(X,y,H)]
    for i in range(times):
        H = X*theta.T
        erro = H - y
        temp = np.matrix([0,0])
        temp = theta - erro.T * X * alpha/m #矩阵运算是精髓，临时变量很重要．坑４
        thetas_0.append(temp[0,0])
        thetas_1.append(temp[0,1])
        theta = temp
        cost.append(costfunction(X,y,H))
    return theta,cost,thetas_0,thetas_1
final_theta,cost,thetas_0,thetas_1= gradient_descent(theta,X,y,alpha,m,H,times)
print(final_theta,'\n',cost,'\n',thetas_0,'\n',thetas_1)
"""绘图"""
fig,(ax1,ax2) = plt.subplots(2,1)
H = final_theta * X.T
H = H.T
ax1.plot(x,H,c = 'r',label = 'Prediction')
ax1.scatter(data.population,data.profit,label = 'data')
ax1.legend(loc = 2)
ax2.plot(cost)
ax1.set_xlabel('population')
ax1.set_ylabel('profit')
ax1.set_title('relationship between population and profit'.title())
ax2.set_xlabel('times')
ax2.set_ylabel('cost')
ax2.set_title('how does cost changed'.title())
fig.subplots_adjust(hspace = 0.8)
plt.show()