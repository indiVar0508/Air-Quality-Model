import pandas as pd 
import numpy as np
from sklearn.metrics import r2_score
import warnings
import Graphy as a

#path = input('Give path or URL : ')
warnings.simplefilter('ignore')
dataset = pd.read_csv('Dataset\\airtable.csv',header = None)

for i in range(len(dataset)):
    temp = dataset.values[i,0].split()
    x = temp[1].split(':')
    dataset.iloc[i,0] = float(x[0])*60 + float(x[1])
del i,x,temp
dataset_l = []
for i in range(10):
    dataset_l.append(dataset.iloc[i])

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
dataset = std.fit_transform(dataset.iloc[:,:-1])

X = dataset[:,0:1]
Y = dataset[:,1:-1]
Y_TGS = dataset[:,-1:]
Y_val = []
for i in range(8):
    Y_val.append(Y[:,i])
x_train ,y_train,x_test,y_test= [],[],[],[]
del i

from sklearn.model_selection import train_test_split
for i in range(8):
    x_train_,x_test_,y_train_,y_test_ = train_test_split(X,Y_val[i],test_size = 0.45)
    x_train.append([x_train_])
    x_test.append([x_test_])
    y_train.append([y_train_])
    y_test.append([y_test_])
    del x_test_,x_train_,y_train_,y_test_
del i

print('Data Preprocessed')
