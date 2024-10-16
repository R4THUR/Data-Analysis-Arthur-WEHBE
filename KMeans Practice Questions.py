# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 19:24:31 2024

@author: arthu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline

#1: Loading the Iris dataset (only petal width and length)

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target



#2: the elbow plot

def plot_elbow(X, scaler=None):
    if scaler:
        X = scaler.fit_transform(X)
    distortions = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, distortions, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    if scaler:
        plt.title('Elbow Method (after scaling)')
    else:
        plt.title('Elbow Method (before scaling)')
    plt.show()


#3: Drawing the elbow plot to find optimal value of k (before scaling)

print("Elbow plot before scaling:")
plot_elbow(X)


#4: Applying KMeans clustering with the optimal value of k and plotting the clusters (before and after scaling)

k = 3  # (we know it from the elbow plot)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

#before scaling

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='r', label='Centroids')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('KMeans Clustering (before scaling)')
plt.legend()
plt.show()

#after scaling

print("Elbow plot after MinMaxScaler:")
plot_elbow(X, scaler=MinMaxScaler())


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

#after scaling

plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='r', label='Centroids')
plt.xlabel('Scaled Petal Length')
plt.ylabel('Scaled Petal Width')
plt.title('KMeans Clustering (after MinMaxScaler)')
plt.legend()
plt.show()

# Conclusion: 
print("Conclusion:")
print("Based on the elbow plots and cluster visualizations, scaling did not appear to significantly improve the clustering results in this case. The elbow plots and cluster formations before and after scaling were quite similar. Therefore, for this specific dataset and clustering task, scaling may not be necessary.")
