import numpy as np
import pandas as pd
from k_means import K_Means

class SpectralClustering:
    def __init__(self, k, max_iter=1000):
        # Number of clusters
        self.k = k
        # Maximum iterations willing to do
        self.max_iter = max_iter
        # Centroids
        self.centroids = None
    
    def zero_if_negative(self, x):
        if x < 0:
            return 0
        return x

    def convert_to_adjacency_matrix(self, arr):
        for i in range(0, arr.shape[0]):
            for j in range(0, arr.shape[1]):
                arr[i][j] = self.zero_if_negative(arr[i][j])
        return arr   

    def fit(self, data):
        # save list of index values for k-means
        index_list = list(data.index)
        similarity_matrix = np.array(data.values, dtype=float)

        # convert similarity matrix to adjacency matrix
        adjacency_matrix = self.convert_to_adjacency_matrix(similarity_matrix)

        # compute the degree matrix
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        # print(degree_matrix)

        # compute the laplacian matrix
        laplacian_matrix = degree_matrix - adjacency_matrix
        # print(laplacian_matrix)

        x, V = np.linalg.eig(laplacian_matrix)
        # # eigenvalues
        # print('eigenvalues:')
        # print(x)
        # # eigenvectors
        # print('eigenvectors:')
        # print(V)

        # sort eigen values
        ind = np.argsort(np.linalg.norm(np.reshape(x, (1, len(x))), axis=0))

        # chose first k eigen vectors
        V_K = np.real(V[:, ind[:self.k]])

        # convert eigen vector np array into dataframe for k means
        V_K_column_data = {}
        for i in range(0,V_K.shape[1]):
            V_K_column_data['eigen_value_{}'.format(i+1)] = V_K[:, i]
        V_K_column_data['index'] = index_list

        V_K_df = pd.DataFrame(V_K_column_data)
        V_K_df = V_K_df.set_index('index')

        # perform k-means with the first K eigenvectors as columns
        k_means = K_Means(self.k, max_iter=500)
        
        return k_means.fit(V_K_df)