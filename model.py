from Data_preprocess_ import *
import model_manipulator_ as mm
print('Random Forest Regression : ')
y_pred = []
from sklearn.ensemble import RandomForestRegressor
Predictor = []
for i in range(8):
    regr = mm.train_model(np.array(x_train[i]).reshape(-1,1),np.array(y_train[i]).reshape(-1,1))
    Predictor.append(regr)
    y_pred.append([regr.predict(np.array(x_test[i]).reshape(-1,1))])
    print('accuracy for MQ',i+1,' : ',r2_score(np.array(y_test[i]).reshape(-1,1).ravel(),np.array(y_pred[i]).reshape(-1,1).ravel()))
    choice = 'Y'
    choice = input('Plot the Graph ? [Y/N]')
    if(choice == 'Y' or choice == 'y'):
        x_temp,y_temp = np.array(x_test[i]).reshape(-1,1),np.array(y_test[i]).reshape(-1,1).ravel()
        x_new = np.arange(x_temp.min(),x_temp.max(),0.01)
        x_temp = x_new.reshape(-1,1)
        a.get_graph(x_test[i],y_temp,x_temp,regr)
del i

X_T = Y
x_t_train,x_t_test,y_t_train,y_t_test = train_test_split(Y,Y_TGS,test_size = 0.45)
regr = RandomForestRegressor(n_estimators = 10)
regr.fit(x_t_train,y_t_train)
y_pred_tgs = regr.predict(x_t_test)
print('accuracy for tgs  : ',r2_score(y_t_test,y_pred_tgs))
 
