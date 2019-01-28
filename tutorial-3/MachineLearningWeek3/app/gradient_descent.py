import  numpy as np
# gradient_descent to find a best fit line of our appoaches
def gradient_descent(x,y):
    m_curr=b_curr=0
    #repeat the loop to find value
    iterations=1000
    #length of data training
    n=len(x)
    #It need to define the learning rate and people are used to start with 0.01 and then you can improve it value like remove some zeros 
    #or increacing the a little to find the diffence
    learning_rate=0.08
    for i in range(iterations):
        #find the label of the data traning
        y_predicted=m_curr*x+b_curr
        #find the Mean square error  value
        cost=(1/n)*sum(val**2 for val in (y-y_predicted))
        
        #find M derivative when B is a constain and B derivative  when M is a constain
        md=-(2/n)*sum(x*(y-y_predicted))
        bd=-(2/n)*sum((y-y_predicted))
        
        
        m_curr=m_curr - learning_rate * md
        b_curr=b_curr - learning_rate * bd
        print("m {},b {},cost {}, iteration {}".format(m_curr,b_curr,cost,i))

x=np.array([1,2,3,4,5])
y= np.array([5,7,9,11,13])
gradient_descent(x,y)
