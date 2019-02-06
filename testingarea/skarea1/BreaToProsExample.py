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
y = np.asarray(data["Prostate_C"])
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
# (100, 303.86311274)
# (200, 302.65503928)
# (300, 302.37201293)
# (400, 302.36845186)
# (500, 304.92626686)
# (600, 303.86311274)

#2 model = linear_model.SGDRegressor()
# (100, -7.77726691e+13)
# (200, -1.55530286e+14)
# (300, -2.33287903e+14)
# (400, -3.1104552e+14)
# (500, -3.88803137e+14)
# (600, -7.77726691e+13)

#3 model = linear_model.LassoLars()
# (100, 144.87879227)
# (200, 204.70664196)
# (300, 264.53449165)
# (400, 324.36234134)
# (500, 384.19019103)
# (600, 144.87879227)

#4 model = linear_model.ARDRegression()
# (100, 136.83689516)
# (200, 199.41089875)
# (300, 261.98490234)
# (400, 324.55890594)
# (500, 387.13290953)
# (600, 136.83689516)

#5 model = linear_model.PassiveAggressiveRegressor()
# (100, 47.09388482)
# (200, 94.17414537)
# (300, 141.25440592)
# (400, 188.33466648)
# (500, 235.41492703)
# (600, 47.09388482)

#6 model = linear_model.TheilSenRegressor()
# (100, 115.67439056)
# (200, 180.68104995)
# (300, 245.68770933)
# (400, 310.69436872)
# (500, 375.7010281)
# (600, 115.67439056)

#7 model = linear_model.LinearRegression()
# (100, 132.20335737)
# (200, 196.35962541)
# (300, 260.51589345)
# (400, 324.67216149)
# (500, 388.82842952)
# (600, 132.20335737)

#8 model = linear_model.Ridge()
# (100, 132.20384021)
# (200, 196.35994337)
# (300, 260.51604653)
# (400, 324.67214968)
# (500, 388.82825284)
# (600, 132.20384021)

#Data Notes
# value duplication at 100 and 600
