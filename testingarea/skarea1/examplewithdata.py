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
predictionData2 = [[200]]
predictionData3 = [[300]]
predictionData4 = [[400]]
predictionData5 = [[500]]
predictionData6 = [[600]]

for model in classifiers:
    print(model)
    clf = model
    clf.fit(X.reshape(-1, 1), y)
    result1 = clf.predict(predictionData1)
    result2 = clf.predict(predictionData2)
    result3 = clf.predict(predictionData3)
    result4 = clf.predict(predictionData4)
    result5 = clf.predict(predictionData5)
    result6 = clf.predict(predictionData6)
    print(result1, '\n')
    print(result2, '\n')
    print(result3, '\n')
    print(result4, '\n')
    print(result5, '\n')
    print(result6, '\n\n')

#Result reporting
#1 = [2581.9454382], model = svm.SVR(gamma = 'scale')
#2 = [-3.89458487e+13], model = linear_model.SGDRegressor()
#3 = [1428.48210477], model = linear_model.LassoLars()
#4 = [1456.86804605], model = linear_model.ARDRegression()
#5 = [592.97724255], model = linear_model.PassiveAggressiveRegressor()
#6 = [1046.87568591], model = linear_model.TheilSenRegressor()
#7 = [1415.80666987], model = linear_model.LinearRegression()
