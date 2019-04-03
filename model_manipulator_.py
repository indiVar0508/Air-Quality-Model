from sklearn.ensemble import RandomForestRegressor
def train_model(x,y):
    regr = RandomForestRegressor(n_estimators = 10)
    regr.fit(x,y)
    return regr
def manipulate_time(tim):
    x = tim.split(':')
    print(float(x[0])*60 + float(x[1]))
    return float(x[0])*60 + float(x[1])
