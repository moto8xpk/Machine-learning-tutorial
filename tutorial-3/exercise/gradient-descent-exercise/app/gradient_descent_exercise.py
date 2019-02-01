import numpy as np
import pandas as pd
import math
import sklearn.linear_model
from sklearn.linear_model import LinearRegression


def gradient_descent(x,y):
    m_curr=b_curr=0
    iteration=1000000
    n=len(x)
    learning_rate=0.0002
    cost_pre=0
    m_pre=0
    b_pre=0
    for i in range(iteration):
        y_predicted=m_curr*x +b_curr
        cost=(1/n)*sum([val**2 for val in (y-y_predicted)])
        md=-(2/n)*sum(x*(y-y_predicted))
        bd=-(2/n)*sum(y-y_predicted)

        m_curr=m_curr - learning_rate*md
        b_curr=b_curr - learning_rate*bd

        if(i==0):
            cost_pre=cost
            m_pre=m_curr
            b_pre=b_curr
        if(i>0):
            if(math.isclose(cost,cost_pre,rel_tol=1e-20)):
                print("iteration:{}, m:{}, b:{}, cost:{}".format(i,m_pre,b_pre,cost_pre))
                print("iteration:{}, m:{}, b:{}, cost:{}".format(i,m_curr,m_curr,cost))
                break
            else:
                cost_pre=cost
                m_pre=m_curr
                b_pre=b_curr

def linear_regresion(x,y):
    lr=LinearRegression()
    lr.fit(x,y);
    return lr





df=pd.read_csv("test_scores.csv")
# print(df)

df=df.drop(['name'],axis=1)

print(df['math'].values)
# print(type(df['math'].values))
x_math=np.array(df['math'].values)
y_cs=np.array(df['cs'].values)
# x_math=x_math.astype(float)
# y_cs=y_cs.astype(float)
# print(type(x_math))

gradient_descent(x_math,y_cs)
x_math=[x_math]
lr=linear_regresion(df[['math']],df['cs'])
print(lr.coef_)
print(lr.intercept_)
