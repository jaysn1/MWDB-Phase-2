import numpy as np
import matplotlib.pyplot as plt
import copy
from random import randint

class K_Means:
    def __init__(self, k, max_iter=1000):
        # Number of clusters
        self.k = k
        # Maximum iterations willing to do
        self.max_iter = max_iter
        # Centroids
        self.centroids = None

    # Strategy 1
    def random_centroids(self, data):
        self.centroids = data.sample(n=self.k, random_state = 42)

    def fit(self, data):
        self.random_centroids(data)
        is_converged = False
        
        iter = 0
        cluster = {}
        prev_cluster = {}
        is_converged = False
        while iter<self.max_iter and (not is_converged):
            for m in range(self.k):
                # Initialize empty list for each cluster
                cluster[m] = []

            # For each point in dataset, calculate distance to every chosen centroid and update
            # the cluster list of the centroid
            for index, row in data.iterrows():
                # Calculate distance of gesture to the centroid values
                distances = [np.linalg.norm(np.array(row)-np.array(self.centroids.iloc[centroid_index])) for centroid_index in range(0, len(self.centroids))]
                # Chose the centoid with the least distance
                centroid_to_choose = distances.index(min(distances))
                # Add the gesture to the centroid
                cluster[centroid_to_choose].append(index)

            # Calculate mean of all gesture column values under each centroid
            # to update the new centroid
            for centroid_index in cluster.keys():
                # Get all gestures under specified centroid
                rows_under_centroid_index = data.loc[cluster[centroid_index]]
                # Retrieve all the columns
                columns = data.columns
                # Update the centroid values
                self.centroids.iloc[centroid_index] = list(rows_under_centroid_index[columns].mean())
            
            # Count to check if the clusters have converged
            outer_count = 0
            for centroid_index in cluster.keys():
                if centroid_index not in prev_cluster:
                    break
                if len(cluster[centroid_index]) != len(prev_cluster[centroid_index]):
                    break
                # Sort both centroid clusters
                cluster[centroid_index].sort()
                prev_cluster[centroid_index].sort()
                inner_count = 0
                
                for i in range(0, len(cluster[centroid_index])):
                    if cluster[centroid_index][i] != prev_cluster[centroid_index][i]:
                        break
                    else:
                        inner_count += 1
                if inner_count == len(cluster[centroid_index]):
                    outer_count += 1

            # Make sure all the gestures under the centroid are the same
            # as they were in the previous iteration
            if outer_count == len(cluster.keys()):
                # print("Converged at {} iteration".format(iter))
                is_converged = True 

            # Update previous cluster
            prev_cluster = copy.deepcopy(cluster)
            iter += 1

        return cluster
    
