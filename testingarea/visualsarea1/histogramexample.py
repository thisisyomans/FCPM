import matplotlib.pyplot as pt
import pandas as pd

age_data = [20, 30, 54, 66, 70, 10, 45, 65, 77, 99, 120, 130, 29, 40, 80, 75, 90]
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80,90, 100, 110, 120]

pt.hist(age_data, bins, histtype = 'bar', rwidth = 0.8)
pt.title('Age Distribution')
pt.xlabel('Age')
pt.ylabel('People')
pt.show()
