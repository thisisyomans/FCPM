import pandas as pd
import numpy as np
from sklearn import svm, linear_model

data = pd.read_csv("Cancer_Rates.csv")

classifiers = [
    svm.SVR(gamma = 'scale'), #1
    linear_model.SGDRegressor(), #2
    linear_model.LassoLars(), #3
    linear_model.ARDRegression(), #4
    linear_model.PassiveAggressiveRegressor(), #5
    linear_model.TheilSenRegressor(), #6
    linear_model.LinearRegression() #7
]

X = np.asarray(data["Breast_Can"])
y = np.asarray(data["All_Cancer"])
predictionData = [[100, 200, 300, 400, 500, 600]]
predictionData1 = [[100]]

for model in classifiers:
    print(model)
    clf = model
    clf.fit(X.reshape(-1, 1), y)
    result = clf.predict(predictionData1)
    print(result, '\n\n')

#Result reporting
#1 = [2581.9454382], model = svm.SVR(gamma = 'scale')
#2 = [-3.89458487e+13], model = linear_model.SGDRegressor()
#3 = [1428.48210477], model = linear_model.LassoLars()
#4 = [1456.86804605], model = linear_model.ARDRegression()
#5 = [592.97724255], model = linear_model.PassiveAggressiveRegressor()
#6 = [1046.87568591], model = linear_model.TheilSenRegressor()
#7 = [1415.80666987], model = linear_model.LinearRegression()
