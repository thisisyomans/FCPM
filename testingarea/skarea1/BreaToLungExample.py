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
    linear_model.LinearRegression(), #7
    linear_model.Ridge() #8
]

X = np.asarray(data["Breast_Can"])
y = np.asarray(data["Lung_Bronc"])
predictionData = [[100, 200, 300, 400, 500, 600]]
predictionData1 = [[100]] #NOTE: there is no breast_can index below 200
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
    print(result1, '\n\n')

#Result reporting

#1 model = svm.SVR(gamma = 'scale')
# (100, 279.70279933)
# (200, 280.9292097)
# (300, 280.9292097)
# (400, 282.08352754)
# (500, 280.72103253)
# (600, 279.70279933)

#2 model = linear_model.SGDRegressor()
# (100, 5.93700662e+12)
# (200, 1.18625492e+13)
# (300, 1.77880918e+13)
# (400, 2.37136344e+13)
# (500, 2.9639177e+13)
# (600, 5.93700662e+12)

#3 model = linear_model.LassoLars()
# (100, 262.06846862)
# (200, 276.91967738)
# (300, 291.77088614)
# (400, 306.6220949)
# (500, 321.47330366)
# (600, 262.06846862)

#4 model = linear_model.ARDRegression()
# (100, 291.86483757)
# (200, 296.54115678)
# (300, 301.217476)
# (400, 305.89379521)
# (500, 310.57011443)
# (600, 291.86483757)

#5 model = linear_model.PassiveAggressiveRegressor()
# (100, 55.20047296)
# (200, 110.36494027)
# (300, 165.52940757)
# (400, 220.69387488)
# (500, 275.85834219)
# (600, 55.20047296)

#6 model = linear_model.TheilSenRegressor()
# (100, 182.02013831)
# (200, 183.71588866)
# (300, 185.411639)
# (400, 187.10738935)
# (500, 188.80313969)
# (600, 182.02013831)

#7 model = linear_model.LinearRegression()
# (100, 249.39303372)
# (200, 268.57266083)
# (300, 287.75228793)
# (400, 306.93191504)
# (500, 326.11154215)
# (600, 249.39303372)

#8 model = linear_model.Ridge()
# (100, 249.39317806)
# (200, 268.57275588)
# (300, 287.7523337)
# (400, 306.93191151)
# (500, 326.11148933)
# (600, 249.39317806)

#Data Notes
# value duplication at 100 and 600