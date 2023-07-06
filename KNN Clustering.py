#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import re
import random
import seaborn as sns
from collections import Counter
np.random.seed(2)
df = pd.read_csv('dataset', header=None)
newd = df.copy()

# define a lambda function to extract letters and numbers from each array
extract_letters_numbers = lambda arr: ([re.findall('[a-zA-Z]', s) for s in arr],
                                        [re.findall('-?\d*\.?\d+', s) for s in arr])
# apply the lambda function to the 'col' column
newd['letters_numbers'] = newd.apply(lambda arr: extract_letters_numbers(arr))
nums = newd['letters_numbers'][1]
letters = newd['letters_numbers'][0]

#joining list of characters to form words
words = []
for i in letters:
    word = (''.join(i))
    words.append(word)

#create dataframe for words and numbers
dict= {'words': words, 'numbers':nums}
newdf = pd.DataFrame(dict)

#extract features
features = pd.DataFrame(newdf['numbers'].apply(pd.Series).values.tolist())
feature = features.drop(features.columns[[300, 301]], axis=1)

#fulldataset
final = pd.concat([newdf, feature], axis=1)
dt = final.drop(['numbers'], axis=1)

#classification
dt['class'] = ''
# Assign the value 'animals' to the 'class' column for rows 0 to 49
dt.iloc[0:50, dt.columns.get_loc('class')] = 'animals'
dt.iloc[50:211, dt.columns.get_loc('class')] = 'countries'
dt.iloc[211:269, dt.columns.get_loc('class')] = 'fruit'
dt.iloc[269:327, dt.columns.get_loc('class')] = 'vegetables'

#encoding
class_mapping = {'animals': 1, 'countries': 2, 'fruit': 3, 'vegetables': 4}
dt['class'] = dt['class'].replace(class_mapping)

#X and Y
X = dt.drop(['words','class'], axis=1)
y = dt['class']
train_size = int(0.7 * len(dt))
train_set = dt[:train_size]
test_set = dt[train_size:]
X_train = train_set.drop(['words','class'], axis=1)
y_train = train_set['class']
X_test = test_set.drop(['words','class'], axis=1)
y_test = test_set['class']
X_trainp = X_train.to_numpy()
X_testp = X_test.to_numpy()
y_trainp = y_train.to_numpy()
y_testp = y_test.to_numpy()
X_trainf = X_train.astype(float)
X_testf = X_test.astype(float)

#Preprocessed
X = dt.drop(['words','class'], axis=1)
y = dt['class']
Xt = X.to_numpy()
Xt = [[float(val) for val in row] for row in Xt]
Xtarr = np.array(Xt)
labes = y.to_numpy()

#Silhouette Score
def silhouette_score(X, labels):
    n_clusters = len(np.unique(labels))
    if n_clusters == 1:
        return None
    n_samples = X.shape[0]

    _, counts = np.unique(labels, return_counts=True)

    # Calculate the mean distance between each point and all other points in its cluster
    cluster_means = np.zeros((n_clusters, X.shape[1]))
    for i in range(n_clusters):
        cluster_means[i] = np.mean(X[labels.astype(int) == i], axis=0)

    # Calculate the distance between each point and all other points in its cluster
    cluster_distances = np.zeros((n_samples, n_clusters))
    for i in range(n_clusters):
        cluster_distances[:, i] = np.sqrt(np.sum((X - cluster_means[i])**2, axis=1))

    # Calculate the distance between each point and all other points in the nearest cluster
    nearest_cluster_distances = np.zeros((n_samples, n_clusters))
    for i in range(n_clusters):
        mask = np.ones(n_clusters, dtype=bool)
        mask[i] = False
        nearest_cluster_distances[:, i] = np.min(cluster_distances[:, mask], axis=1)

    # Calculate the silhouette score for each point
    s = np.zeros(n_samples)
    for i in range(n_samples):
        a = cluster_distances[i, int(labels[i])]
        b = nearest_cluster_distances[i, int(labels[i])]
        s[i] = (b - a) / np.maximum(a, b)

    # Return the mean silhouette score
    return np.mean(s)



#Algorithm 1: KMeans
class KMeansClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def Assignment(self, X_train):
        min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        self.centroids = random.sample(list(X_train), self.n_clusters)
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any():  # limit number of iterations to prevent infinite loops
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = np.linalg.norm(np.array(self.centroids) - x, axis=None)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

            
    def Optimisation(self, X_train):
        self.Assignment(X_train)
        centroids = np.array(self.centroids)
        centroid_idxs = np.zeros(len(X_train))
        for i, x in enumerate(X_train):
            dists = np.linalg.norm(centroids - x, axis=1)
            centroid_idx = np.argmin(dists)
            centroid_idxs[i] = centroid_idx
        return centroids, centroid_idxs

#KMeans Plot
n_clusters_range = range(1, 10)
for n_clusters in n_clusters_range:
    kmeans = KMeansClustering(n_clusters=n_clusters)
    kmeans.Assignment(Xt)
    clust, classification = kmeans.Optimisation(Xt)

sns.scatterplot(x=[X[0] for X in Xt],
                y=[X[1] for X in Xt],
                hue=labes,
                style=classification,
                palette="deep",
                legend=None
                )
plt.plot(np.array(kmeans.centroids)[:, 0],
         np.array(kmeans.centroids)[:, 1],
         '+',
         markersize=10,
         )
plt.title('K-Means Clustering on dataset')
plt.show()

# Compute silhouette scores for different values of k
silhouette_scores = []
n_clusters_range = range(1, 10)
for n_clusters in n_clusters_range:
    class_centers, classification = kmeans.Optimisation(Xt)
    cluster_labels = np.array(classification).astype(float)
    score = silhouette_score(Xtarr , cluster_labels)
    silhouette_scores.append(score)

plt.plot(n_clusters_range, silhouette_scores, '-o',
         markersize=10)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette score for K-Means Clustering')
plt.show()

#Algorithm 2: KMeans Plus
class KMeansPlus:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        
        
    def Assignment(self, X_train):
        self.centroids = [random.choice(X_train)]
        
        for _ in range(self.n_clusters-1):
            # Calculate distances from points to each centroids
            #axis=2 is the norm along the last axis of the distances array with shape num_centroids, num_points
            dists = np.linalg.norm(np.array(self.centroids)[:,np.newaxis, :] - X_train, axis=2)
            #use norm again to calc norm across all distances
            #result with shape num_points
            norms = np.linalg.norm(dists, axis=0)
            norms /= np.sum(norms)
            # Choose remaining points based on their distances
            new_centroid_idx = np.random.choice(range(len(X_train)), size=1, p=norms)
            self.centroids += [X_train[new_centroid_idx[0]]]
            
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any():  # limit number of iterations to prevent infinite loops
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = np.linalg.norm(np.array(self.centroids) - x, axis=None)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
                
   
    def Optimisation(self, X_train):
        self.Assignment(X_train)
        centroids = np.array(self.centroids)
        centroid_idxs = np.zeros(len(X_train))
        for i, x in enumerate(X_train):
            dists = np.linalg.norm(centroids - x, axis=1)
            centroid_idx = np.argmin(dists)
            centroid_idxs[i] = centroid_idx
        return centroids, centroid_idxs

#Plotting the Clusters
kmeansp = KMeansPlus(n_clusters=n_clusters)
kmeansp.Assignment(Xt)
class_centers, classification = kmeansp.Optimisation(Xt)
sns.scatterplot(x=[X[0] for X in Xt],
                y=[X[1] for X in Xt],
                hue=labes,
                style=classification,
                palette="deep",
                legend=None
                )
plt.plot(np.array(kmeansp.centroids)[:, 0],
         np.array(kmeansp.centroids)[:, 1],
         '+',
         markersize=10,
         )
plt.title('K-Means Plus Clustering on dataset')
plt.show()

#Silhouette Score for KMeans+
silhouette_scores = []
n_clusters_range = range(1, 10)
for n_clusters in n_clusters_range:
    kmeansp.Assignment(Xt)
    class_centers, classification = kmeansp.Optimisation(Xt)
    cluster_labels = np.array(classification).astype(float)
    score = silhouette_score(Xtarr , cluster_labels)
    silhouette_scores.append(score)

plt.plot(n_clusters_range, silhouette_scores, '-o',
         markersize=10)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette score for KMeans++')
plt.show()

#Algorithm 3: Bisecting KMeans 
class BKMeansC:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        
    def Assignment(self, X_train):
        min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        if self.n_clusters > len(X_train):
            self.n_clusters = len(X_train)
        self.centroids = random.sample(list(X_train), self.n_clusters)
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any():  # limit number of iterations to prevent infinite loops
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = np.linalg.norm(np.array(self.centroids) - x, axis=None)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
   
    def Optimisation(self, X_train):
        self.Assignment(X_train)
        centroids = np.array(self.centroids)
        centroid_idxs = np.zeros(len(X_train))
        for i, x in enumerate(X_train):
            dists = np.linalg.norm(centroids - x, axis=1)
            centroid_idx = np.argmin(dists)
            centroid_idxs[i] = centroid_idx
        return centroids, centroid_idxs.astype(int)
    
    def Bisecting(self, X_train):
        clusters = [X_train]
        while len(clusters) < self.n_clusters:
            maxscore = -np.inf
            for i, cluster in enumerate(clusters):
                if len(cluster) <= 1:
                    continue
                class_centers, centroid_idxs = self.Optimisation(cluster)
                # Calculate the total sum of squared distances of points from their cluster center
                score = np.sum((cluster - class_centers[centroid_idxs])**2)
                # Invert the score if it's negative, so larger scores are always better
                if score < 0:
                    score = -score
                if score > maxscore:
                    maxscore = score
                    maxclusteridx = i
            # Split the cluster with the highest score in two
            clusterr = clusters.pop(maxclusteridx)
            class_centers, centroid_idxs = self.Optimisation(clusterr)
            # Create new clusters by indexing into the original cluster array
            cluster1 = [clusterr[i] for i in range(len(clusterr)) if centroid_idxs[i] == 0]
            cluster2 = [clusterr[i] for i in range(len(clusterr)) if centroid_idxs[i] == 1]
            clusters.extend([cluster1, cluster2])

        # Call the optimization function on each of the resulting clusters
        optimized_clusters = []
        for cluster in clusters:
            class_centers, centroid_idxs = self.Optimisation(cluster)
            optimized_clusters.append(class_centers[centroid_idxs])

        return optimized_clusters


    
#Plotting Bisecting KMeans Clustering
kmeansb = BKMeansC(n_clusters=n_clusters)
class_centers, classification = kmeansb.Optimisation(Xt)
sns.scatterplot(x=[X[0] for X in Xt],
                y=[X[1] for X in Xt],
                hue=labes,
                palette="deep",
                legend=None
                )
plt.plot(np.array(kmeansb.centroids)[:, 0],
         np.array(kmeansb.centroids)[:, 1],
         '+',
         markersize=10,
         )
plt.title('Bisecting K-Means Clustering on dataset')
plt.show()

#Silhouette Score for Bisecting KMeans
silhouette_scores = []
n_clusters_range = range(1, 10)
for n_clusters in n_clusters_range:
    kmeansb = BKMeansC(n_clusters=n_clusters)
    clusters = kmeansb.Bisecting(Xt)
    cluster_labels = np.zeros(Xtarr.shape[0], dtype=int)
    for i, cluster in enumerate(clusters):
        cluster_idxs = kmeansb.Optimisation(cluster)[1]
        cluster_labels[cluster_idxs] = i
    # Remove any empty clusters
    non_empty_clusters = np.unique(cluster_labels)
    if len(non_empty_clusters) == 1:
        score = 0
    else:
        non_empty_labels = np.zeros_like(cluster_labels)
        for i, c in enumerate(non_empty_clusters):
            non_empty_labels[cluster_labels == c] = i
        score = silhouette_score(Xtarr, non_empty_labels.ravel())
    silhouette_scores.append(score)
    
plt.plot(n_clusters_range, silhouette_scores, '-o', markersize=10)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette score for Bisecting KMeans')
plt.show()


# In[ ]:




