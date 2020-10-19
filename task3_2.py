# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:27:31 2020

@author: Jaysn
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, PCA, TruncatedSVD as SVD, LatentDirichletAllocation as LDA

def SVD_gesture_gesture():
    gesture_gesture_similarity = pd.read_json("intermediate/gesture_gesture_similarity_dictionary.json")
    
    k=50
    gestures = np.array(gesture_gesture_similarity)
    svd = SVD(n_components=k, algorithm = "randomized", n_iter=7, random_state=42)
    try:
        svd.fit(gestures)
    except ValueError as e:
        raise ValueError(e)
    eigen_vectors = svd.components_
    transformed_data = svd.transform(gestures)
    
    gesture_scores = []
    gestures = list(gesture_gesture_similarity.columns)
    for eigen_vector in eigen_vectors:
        gesture_score = sorted(zip(gestures, eigen_vector), key=lambda x: x[1])
        gesture_scores.append(gesture_score)
    
    transformed_gestures_gestures_dict = {}
    for i in range(len(gestures)):
        transformed_gestures_gestures_dict[gestures[i]] = list(transformed_data[i])
    return transformed_gestures_gestures_dict, gesture_scores

transformed_gestures_gestures_dict, gesture_scores = SVD_gesture_gesture()

