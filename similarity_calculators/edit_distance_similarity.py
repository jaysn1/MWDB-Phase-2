# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:00:53 2020

@author: Jaysn
"""
import numpy as np

def edit_distance_similarity(query_words, sensor_words):
    distance = 0
    row = len(sensor_words) + 1
    col = len(query_words) + 1
    ED_matrix = np.array([[0]*col]*row)

    for i in range(row):
        ED_matrix[i][0] = i
    
    for j in range(col):
        ED_matrix[0][j] = j
    
    for j in range(1,col):
        for i in range(1,row):
            if sensor_words[i-1] == query_words[j-1]:
                cost = 0
            else:
                cost = 1
            ED_matrix[i][j] = min(ED_matrix[i-1][j] + 1,
                                  ED_matrix[i][j-1] + 1,
                                  ED_matrix[i-1][j-1] + cost)
    
    distance = ED_matrix[i][j]
    return distance