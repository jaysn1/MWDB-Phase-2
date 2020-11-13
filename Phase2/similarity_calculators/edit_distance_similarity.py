# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:00:53 2020

@author: Jaysn
"""
import numpy as np

sensor_cost = {
    #1	HipCenter
    1:1,
    #2	Spine
    2:1,
    #3	ShoulderCenter
    3:2,
    #4	Head
    4:2,
    #5	ShoulderLeft
    5:3,
    #6	ElbowLeft
    6:4,
    #7	WristLeft
    7:5,
    #8	HandLeft
    8:5,
    #9	ShoulderRight
    9:3,
    #10	ElbowRight
    10:4,
    #11	WristRight
    11:5,
    #12	HandRight
    12:5,
    #13	HipLeft
    13:1,
    #14	KneeLeft
    14:2,
    #15	AnkleLeft
    15:2,
    #16	FootLeft
    16:3,
    #17	HipRight
    17:1,
    #18	KneeRight
    18:2,
    #19	AnkleRight
    19:2,
    #20	FootRight
    20:3,
}

def edit_distance_similarity(query_words, sensor_words):
    distance = 0
    row = len(sensor_words) + 1
    col = len(query_words) + 1
    ED_matrix = np.array([[0]*col]*row)

    for i in range(row):
        ED_matrix[i][0] = i
    
    for j in range(col):
        ED_matrix[0][j] = j
    
    for i in range(1,row):
        for j in range(1,col):
            if sensor_words[i-1] == query_words[j-1]:
                ED_matrix[i][j] = ED_matrix[i-1][j-1]
            else:
                cost = 1
                ED_matrix[i][j] = min(ED_matrix[i-1][j] + 1,
                                ED_matrix[i][j-1] + 1,
                                ED_matrix[i-1][j-1] + cost)
    
    distance = ED_matrix[row-1][col-1]
    # since all words will have same sensor value
    # retrieving sensor_id from the first symbol
    # of the first sensor word
    sensor_id = sensor_words[0][0]
    return sensor_cost[sensor_id] * distance