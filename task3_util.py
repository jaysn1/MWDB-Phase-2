# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:27:31 2020

@author: Jaysn
"""
import pandas as pd
import numpy as np
import os
import fnmatch
from sklearn.decomposition import NMF, TruncatedSVD as SVD

# Delete gesture_gesture_score.txt file if present
def delete_gesture_gesture_score_text_file(directory):
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*gesture_gesture_score*.txt'):
            os.remove(os.path.join(root,filename))

def store_gesture_gesture_score_dict(gesture_gesture_score_matrix, directory, file_name):
    delete_gesture_gesture_score_text_file(directory)
    with open(os.path.join(directory, file_name), "w") as f:
        for row in range(len(gesture_gesture_score_matrix)):
            for col in range(len(gesture_gesture_score_matrix[row])):
                score, gesture = gesture_gesture_score_matrix[row][col]
                f.write("{} : {} \t".format(gesture, score))
            f.write("\n")

def SVD_gesture_gesture(p, gesture_gesture_dir="intermediate/gesture_gesture_similarity_dictionary.json"):
    gesture_gesture_similarity = pd.read_json(gesture_gesture_dir)
    gestures = np.array(gesture_gesture_similarity, dtype=float)

    svd = SVD(n_components=p, algorithm = "randomized", n_iter=7, random_state=42)
    # algorithm: To computes a truncated randomized SVD
    # n_iter: Number of iterations for randomized SVD solver
    # random_states: Used during randomized svd. To produce reproducible results
    try:
        svd.fit(gestures)
    except ValueError as e:
        raise ValueError(e)
    eigen_vectors = svd.components_
    transformed_data = svd.transform(gestures)
    
    gesture_ids = list(gesture_gesture_similarity.columns)
    gesture_scores = []
    for eigen_vector in eigen_vectors:
        gesture_score = sorted(zip(abs(eigen_vector), gesture_ids), key=lambda x: -x[0])
        gesture_scores.append(gesture_score)

    transformed_gestures_dict = {}
    for i in range(len(gesture_ids)):
        transformed_gestures_dict[gesture_ids[i]] = list(transformed_data[i])

    return transformed_gestures_dict, gesture_scores


def NMF_gesture_gesture(p, gesture_gesture_dir="intermediate/gesture_gesture_similarity_dictionary.json"):
    gesture_gesture_similarity = pd.read_json(gesture_gesture_dir)
    gestures = np.array(gesture_gesture_similarity, dtype=float)
    nmf = NMF(n_components=p, init='random', random_state=0)
    try:
        nmf.fit(gestures)
    except ValueError as e:
        raise ValueError(e)
    eigen_vectors = nmf.components_
    transformed_data = nmf.transform(gestures)
    
    gesture_ids = list(gesture_gesture_similarity.columns)
    gesture_scores = []
    for eigen_vector in eigen_vectors:
        gesture_score = sorted(zip(abs(eigen_vector), gesture_ids), key=lambda x: -x[0])
        gesture_scores.append(gesture_score)

    transformed_gestures_dict = {}
    for i in range(len(gesture_ids)):
        transformed_gestures_dict[gesture_ids[i]] = list(transformed_data[i])

    return transformed_gestures_dict, gesture_scores
