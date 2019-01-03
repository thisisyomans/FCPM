import matplotlib.pyplot as pt
import pandas as pd

data = pd.read_csv("Cancer_Rates.csv")
data = data.head(20)

pt.bar(data["ZIP"], data["Breast_Can"], color = ["green", "blue", "pink", "red"])

pt.xlabel("ZIP", color = "green")
pt.ylabel("Breast Cancer Index", color = "blue")
pt.title("Breast Cancer Index by Area", color = "green")
pt.show()
