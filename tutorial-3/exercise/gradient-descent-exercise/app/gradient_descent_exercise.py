import numpy as np
import pandas as pd

def gradient_descent(x,y):
    m_curr=b_curr=0
    iteration=10000
    n=len(x)
    learning_rate=0.0002

    for i in range(iteration):
        y_predicted=m_curr*x +b_curr
        cost=(1/n)*sum([val**2 for val in (y-y_predicted)])

        md=-(2/n)*sum(x*(y-y_predicted))
        bd=-(2/n)*sum(y-y_predicted)

        m_curr=m_curr - learning_rate*md
        b_curr=b_curr - learning_rate*bd

        print("iteration:{}, m:{}, b:{}, cost:{}".format(i,m_curr,b_curr,cost))



df=pd.read_csv("test_scores.csv")
print(df)

df=df.drop(['name'],axis=1)

print(df['math'].values)
# print(type(df['math'].values))
x_math=np.array(df['math'].values)
y_cs=np.array(df['cs'].values)
# x_math=x_math.astype(float)
# y_cs=y_cs.astype(float)
# print(type(x_math))

gradient_descent(x_math,y_cs)
