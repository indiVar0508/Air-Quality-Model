import matplotlib.pyplot as plt
import numpy as np

def get_graph(X,Y,x_temp,regr):
    plt.scatter(np.array(X).reshape(-1,1),np.array(Y).reshape(-1,1).ravel(),color = 'red')
    plt.plot(x_temp,regr.predict(x_temp),color = 'blue')
    plt.xlabel('time(min)')
    plt.ylabel('MQ Sensor ')
    plt.show()


'''
print('Linear Regression : ')
y_pred = []
from sklearn.linear_model import LinearRegression
for i in range(10):
    regr = LinearRegression()
    regr.fit(np.array(x_train[i]).reshape(-1,1),np.array(y_train[i]).reshape(-1,1)) 
    y_pred.append([regr.predict(np.array(x_test[i]).reshape(-1,1))])
    print('accuracy for MQ',i+1,' : ',r2_score(np.array(y_test[i]).reshape(-1,1).ravel(),np.array(y_pred[i]).reshape(-1,1).ravel()))
    #choice = input('Plot the Graph ? [Y/N] : ')
    choice = 'Y'
    if(choice == 'Y' or choice == 'y'):
        #print(np.array(x_test[i]).reshape(-1,1).shape,np.array(y_test[i]).reshape(-1,1).ravel().shape)
        get_graph(np.array(x_test[i]).reshape(-1,1),np.array(y_test[i]).reshape(-1,1).ravel(),np.array(y_pred[i]).reshape(-1,1).ravel())
        #input('Press Enter To Continue...')
del i,regr    
    

print('Support Vector Regression : ')
y_pred = []
from sklearn.svm import SVR
for i in range(10):
    regr = SVR()
    regr.fit(np.array(x_train[i]).reshape(-1,1),np.array(y_train[i]).reshape(-1,1)) 
    y_pred.append([regr.predict(np.array(x_test[i]).reshape(-1,1))])
    print('accuracy for MQ',i+1,' : ',r2_score(np.array(y_test[i]).reshape(-1,1).ravel(),np.array(y_pred[i]).reshape(-1,1).ravel()))
    #choice = input('Plot the Graph ? [Y/N] : ')
    choice = 'Y'
    if(choice == 'Y' or choice == 'y'):
        #print(np.array(x_test[i]).reshape(-1,1).shape,np.array(y_test[i]).reshape(-1,1).ravel().shape)
        get_graph(np.array(x_test[i]).reshape(-1,1),np.array(y_test[i]).reshape(-1,1).ravel(),np.array(y_pred[i]).reshape(-1,1).ravel())
        #input('Press Enter To Continue...')
del i,regr

print('Decision Tree Regression : ')
y_pred = []
from sklearn.tree import DecisionTreeRegressor
for i in range(10):
    regr = DecisionTreeRegressor()
    regr.fit(np.array(x_train[i]).reshape(-1,1),np.array(y_train[i]).reshape(-1,1)) 
    y_pred.append([regr.predict(np.array(x_test[i]).reshape(-1,1))])
    print('accuracy for MQ',i+1,' : ',r2_score(np.array(y_test[i]).reshape(-1,1).ravel(),np.array(y_pred[i]).reshape(-1,1).ravel()))
    #choice = input('Plot the Graph ? [Y/N] : ')
    choice = 'Y'
    if(choice == 'Y' or choice == 'y'):
        #print(np.array(x_test[i]).reshape(-1,1).shape,np.array(y_test[i]).reshape(-1,1).ravel().shape)
        x_temp,y_temp = np.array(x_test[i]).reshape(-1,1),np.array(y_test[i]).reshape(-1,1).ravel()
        x_new = np.arange(x_temp.min(),x_temp.max(),0.01)
        x_temp = x_new.reshape(-1,1)
        #get_graph(x_temp,y_temp,y_pred_temp)
        plt.scatter(np.array(x_test[i]).reshape(-1,1),np.array(y_test[i]).reshape(-1,1).ravel(),color = 'red')
        plt.plot(x_temp,regr.predict(x_temp),color = 'blue')
        plt.xlabel('time(min)')
        plt.ylabel('MQ Sensor')
        plt.show()
        #input('Press Enter To Continue...')
del i,regr
''' 
