import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metric

from time import time
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='Experiment 03')
parser.add_argument('-dim', dest='n_components', type=int, required=True)
parser.add_argument('-k', dest='n_clusters', type=int, required=True)

N_NEIGHBORS = 15

def init_dataset():
	# Read csv
    df = pd.read_csv('bags.csv', header=None)

    return df.values

def pca(data, n_components):

	pca = PCA(n_components=n_components)

	pca.fit(data)

	print("Variance:", np.sum(pca.explained_variance_ratio_))

	data_reduced = pca.fit_transform(data)

	return data_reduced

def clustering(data, n_clusters):

	# Create model
	km = KMeans(n_clusters=n_clusters, init="random", n_init=100, max_iter=500, n_jobs=8, algorithm="full")

	print("Clustering data with %s clusters..." % n_clusters)

	t0 = time()
	km.fit(data)
	tf = time()
	print("Done in %0.3fs" % (tf - t0))

	print("Inertia: %0.4f" % km.inertia_)

	return km

def analysis(data, km):
	# Load tweets
	df_tweets = pd.read_csv('health.txt', delimiter='|')

	# Number of clusters
	n_clusters = km.cluster_centers_.shape[0]

	for k in range(0, 3):
		# Select cluster for analysis
		idx = random.randint(0, n_clusters-1)

		print("Cluster: %d" % idx)

		# Filter points by label
		points = np.where(km.labels_ == idx)[0]

		# Remove duplicates
		tweets = df_tweets.iloc[points[:], 2].duplicated()
		duplicated_idxs = np.where(tweets == True)
		points_ = np.delete(points, duplicated_idxs)

		# Find medoid
		min_idx = 0
		min_dist = distance.euclidean(km.cluster_centers_[idx,:], data[points_[0],:])
		for i in range(1, len(points_)):
			dist = distance.euclidean(km.cluster_centers_[idx,:], data[points_[i],:])
			if dist < min_dist:
				min_idx = points_[i]
				min_dist = dist

		medoid_idx = min_idx
		print("\tMedoid: " + df_tweets.iloc[medoid_idx, 2])

		# Find closest neighbors
		dist = []
		indices = []
		for i in range(0, len(points_)):
			if medoid_idx != points_[i]:
				indices.append(points_[i])
				dist.append(distance.euclidean(data[medoid_idx,:], data[points_[i],:]))

		sorted_idxs = np.argsort(dist)
		for i in range(0, min(N_NEIGHBORS, len(sorted_idxs))):
			print("\tNeighbor " + str(i) + ": " + df_tweets.iloc[indices[sorted_idxs[i]], 2] + " (%0.3f)" % dist[sorted_idxs[i]])

	# Compute histogram of clusters
	hist = []
	for i in range(0, n_clusters):
		# Filter points by label
		points = np.where(km.labels_ == i)[0]
		hist.append(len(points))

	# Plot histogram
	fig = plt.figure()
	width = 1/1.5
	plt.bar(range(n_clusters), hist, width, color="blue")
	plt.xlabel("Group")
	plt.ylabel("Frequency")
	plt.show()
	fig.savefig("histogram.png")

	# Metrics of quality
	print("Silhouette Coefficient: %0.3f" % metric.silhouette_score(data, km.labels_, metric='euclidean'))
	print("Davies-Bouldin Index: %0.3f" % metric.davies_bouldin_score(data, km.labels_))
	print("Calinski Index: %0.3f" % metric.calinski_harabaz_score(data, km.labels_))

def main():
	args = parser.parse_args()

	# Load dataset
	data = init_dataset()

	# Reduce dimensionality using PCA
	data_reduced = pca(data, args.n_components)

	# Clustering data
	km = clustering(data_reduced, args.n_clusters)

	# Clustering analysis
	analysis(data_reduced, km)


if __name__ == '__main__':
    main()