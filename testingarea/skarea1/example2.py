from sklearn import cluster, datasets

iris = datasets.load_iris()

#create clusters for k = 3
k=3
k_means = cluster.KMeans(k)

#fitting, aka training
k_means.fit(iris.data)

#printing results (clustered) - ny k_means and iris target
print(k_means.labels_[::10])
print(iris.target[::10])