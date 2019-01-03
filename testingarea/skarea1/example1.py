from sklearn import datasets
from sklearn import svm, linear_model, neighbors

iris = datasets.load_iris()

#setting up classifiers
clf = svm.LinearSVC()
reg = linear_model.LinearRegression()
knn = neighbors.KNeighborsClassifier()

#fitting, aka training
#use x to predict y in (x, y)
clf.fit(iris.data, iris.target)
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
knn.fit(iris.data, iris.target)

#calling predict after training classifiers as they are now usable
clf.predict([[5.0, 3.6, 1.3, 0.25]])
result = knn.predict([[0.1, 0.2, 0.3, 0.4]])

print("Linear SVC")
print(clf.coef_)
print("\nLinear Regression")
print(reg.coef_)
print("\nK Neihgbors")
print(result)