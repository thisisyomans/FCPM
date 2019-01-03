import matplotlib.pyplot as pt
import pandas as pd

data = pd.read_csv("Cancer_Rates.csv")
data = data.head(20)

x = len(data[data.Breast_Can >= 400])
x1 = len(data[(data.Breast_Can >= 300) & (data.Breast_Can < 400)])
x2 = len(data[data.Breast_Can < 300])

pt.axis('equal')

pt.pie([x, x1, x2], colors = ['yellow', 'red', 'blue'], labels = ['400 +', '300s', 'below 300'])

pt.legend(title = 'Description')

pt.show()
