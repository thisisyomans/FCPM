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
# (100, 2581.9454382)
# (200, 2580.90193209)
# (300, 2582.22904749)
# (400, 2581.61052286)
# (500, 2583.00734397)
# (600, 2581.9454382)

#2 model = linear_model.SGDRegressor()
# (100, 6.09340285e+13)
# (200, 1.21875995e+14)
# (300, 1.82817962e+14)
# (400, 2.43759928e+14)
# (500, 3.04701895e+14)
# (600, 6.09340285e+13)

#3 model = linear_model.LassoLars()
# (100, 1428.48210477)
# (200, 1838.18976033)
# (300, 2247.8974159)
# (400, 2657.60507146)
# (500, 3067.31272703)
# (600, 1428.48210477)

#4 model = linear_model.ARDRegression()
# (100, 1456.86804605)
# (200, 1856.88244611)
# (300, 2256.89684618)
# (400, 2656.91124624)
# (500, 3056.92564631)
# (600, 1456.86804605)

#5 model = linear_model.PassiveAggressiveRegressor()
# (100, 624.32722891)
# (200, 1248.49238896)
# (300, 1872.657549)
# (400, 2496.82270905)
# (500, 3120.98786909)
# (600, 624.32722891)

#6 model = linear_model.TheilSenRegressor()
# (100, 1046.87568591)
# (200, 1483.31613256)
# (300, 1919.75657921)
# (400, 2356.19702585)
# (500, 2792.6374725)
# (600, 1046.87568591)

#7 model = linear_model.LinearRegression()
# (100, 1415.80666987)
# (200, 1829.84274378)
# (300, 2243.87881769)
# (400, 2657.91489161)
# (500, 3071.95096552)
# (600, 1415.80666987)

#Data Notes
# value duplication at 100 and 600