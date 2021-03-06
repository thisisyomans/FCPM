import matplotlib.pyplot as pt
import pandas as pd

data = pd.read_csv("Cancer_Rates.csv")
data = data.head(20)

pt.scatter(data["ZIP"], data["Breast_Can"], color = "blue", label = "scatter")

pt.xlabel("ZIP", color = "green")
pt.ylabel("Breast Cancer Index", color = "blue")
pt.title("Breast Cancer Index by Area", color = "green")

pt.plot(data["ZIP"], data["Breast_Can"], color = "red", label = "line graph")
pt.legend()

pt.show()
