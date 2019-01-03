import matplotlib.pyplot as pt
import pandas as pd
fields = ['Breast_Can']

data = pd.read_csv("Cancer_Rates.csv", skipinitialspace = True, usecols = fields)

bins = [200, 300, 400, 500, 600, 700]

pt.hist(data, bins, histtype = 'bar', rwidth = 0.8)
pt.title('Breast Cancer Index Spread')
pt.xlabel('Index')
pt.ylabel('People')
pt.show()