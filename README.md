# Clustering
Clustering analysis for Health News in Twitter

## Experiment 01

Executes the K-Means algorithm for a range of k values and plots the Cost curve per value of k.

### To execute:

python3 experiment_01.py [-maxk -step]

	**-maxk - maximum value of k

	**-step - step to skip between each value of k

## Experiment 02

Executes the K-Means algorithm for a value of k and analyze the medoids and their nearest neighbors for 3 random clusters. In addition computes the quality metrics such as Silhouette Coeficient and return a histogram of the clusters distribuition.

### To execute:

python3 experiment_02.py [-k]

	**-k - number of clusters

## Experiment 03

Executes the PCA to reduce the data dimensionality and executes after that the K-Means algorithm for a value of k and analyze the medoids and their nearest neighbors for 3 random clusters. In addition computes the quality metrics such as Silhouette Coeficient and return a histogram of the clusters distribuition.

### To execute:

python3 experiment_03.py [-k]

	**-dim - number of dimensions to keep

	**-k - number of clusters

## Experiment 04

Executes the Agglomerative Clustering algorithm for a value of k and analyze the quality metrics such as Silhouette Coeficient and return a histogram of the clusters distribuition.

### To execute:

python3 experiment_04.py [-k]

	**-k - number of clusters