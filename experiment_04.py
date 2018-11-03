import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metric

from time import time
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering

parser = argparse.ArgumentParser(description='Experiment 04')
parser.add_argument('-k', dest='n_clusters', type=int, required=True)

def init_dataset():
	# Read csv
    df = pd.read_csv('bags.csv', header=None)

    return df.values

def clustering(data, n_clusters):

	ac = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity='euclidean')

	t0 = time()
	ac.fit(data)
	tf = time()
	print("Done in %0.3fs" % (tf - t0))

	return ac

def analysis(data, n_clusters, ac):
	
	# Compute histogram of clusters
	hist = []
	for i in range(0, n_clusters):
		# Filter points by label
		points = np.where(ac.labels_ == i)[0]
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
	print("Silhouette Coefficient: %0.3f" % metric.silhouette_score(data, ac.labels_, metric='euclidean'))
	print("Davies-Bouldin Index: %0.3f" % metric.davies_bouldin_score(data, ac.labels_))
	print("Calinski Index: %0.3f" % metric.calinski_harabaz_score(data, ac.labels_))

def main():
	args = parser.parse_args()

	# Load dataset
	data = init_dataset()

	# Clustering data
	ac = clustering(data, args.n_clusters)

	# Clustering analysis
	analysis(data, args.n_clusters, ac)


if __name__ == '__main__':
    main()