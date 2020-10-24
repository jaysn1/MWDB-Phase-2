# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:27:31 2020

@author: Jaysn
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, PCA, TruncatedSVD as SVD, LatentDirichletAllocation as LDA


def store_gesture_gesture_score_dict(gesture_gesture_score_matrix,
                                     gesture_gesture_score_dir="intermediate/gesture_gesture_score.txt"):
    with open(gesture_gesture_score_dir, "w") as f:
        for row in range(len(gesture_gesture_score_matrix)):
            for col in range(len(gesture_gesture_score_matrix[row])):
                score, gesture_gesture = gesture_gesture_score_matrix[row][col]
                f.write("{} : {} \t".format(gesture_gesture, score))
            f.write("\n")


def SVD_gesture_gesture(p, gesture_gesture_dir="intermediate/gesture_gesture_similarity_dictionary.json"):
    gesture_gesture_similarity = pd.read_json(gesture_gesture_dir)
    gestures = np.array(gesture_gesture_similarity)
    svd = SVD(n_components=p, algorithm="randomized", n_iter=7, random_state=42)
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


def NMF_gesture_gesture(p, gesture_gesture_dir="intermediate/gesture_gesture_similarity_dictionary.json"):
    gesture_gesture_similarity = pd.read_json(gesture_gesture_dir)
    gestures = np.array(gesture_gesture_similarity)
    nmf = NMF(n_components=p, init='random', random_state=0, max_iter=500)
    try:
        nmf.fit(gestures)
    except ValueError as e:
        raise ValueError(e)
    eigen_vectors = nmf.components_
    transformed_data = nmf.transform(gestures)

    gesture_scores = []
    gestures = list(gesture_gesture_similarity.columns)
    for eigen_vector in eigen_vectors:
        gesture_score = sorted(zip(gestures, eigen_vector), key=lambda x: x[1])
        gesture_scores.append(gesture_score)

    transformed_gestures_gestures_dict = {}
    for i in range(len(gestures)):
        transformed_gestures_gestures_dict[gestures[i]] = list(transformed_data[i])
    return transformed_gestures_gestures_dict, gesture_scores
