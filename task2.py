# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 22:52:18 2020

@author: Jaysn
"""
from Phase2 import task1
import pandas as pd



def read_labels(labels_dir):
    labels = pd.read_excel(labels_dir, header=None)
    labels.columns = ['id', 'label']
    return labels

def calculate_distance_knn(query_values, gesture_values):
    distance = 0
    for query_word, gesture_word in zip(query_values, gesture_values):
        distance += (query_word - gesture_word) ** 2
    return distance ** 0.5

def calculate_distances_from_all_gestures_knn(query_values, gestures):
    distances = {}
    minimum_distance = 100
    similar_gesture = None
    for gesture_id, gesture_values in gestures.items():
        distance = calculate_distance_knn(query_values, gesture_values)
        if distance < minimum_distance:
            minimum_distance = distance
            similar_gesture = gesture_id
        distances[gesture_id] = distance
    return similar_gesture

def KNN(unlabeled_gestures, gestures):
    predictions = {}
    for query_id, query_values in unlabeled_gestures.items():
        prediction_id = calculate_distances_from_all_gestures_knn(query_values, gestures)
        predictions[query_id] = prediction_id
    return predictions

def split_data(gestures, all_labels, training_labels):
    unlabeled_gestures = {}
    labeled_gestures = gestures.copy()
    for gesture_id in training_labels['id']:
        unlabeled_gestures[gesture_id] = gestures[str(gesture_id)]
        del labeled_gestures[str(gesture_id)]
    return unlabeled_gestures, labeled_gestures

def make_shift_accuracy(predictions):
    correct = 0
    for truth, prediction in predictions.items():
        if abs(truth - int(prediction)) < 100:
            correct += 1
    accuracy = correct / len(predictions)
    return accuracy
        
# def main(): 
training_labels_dir = 'data/sample_training_labels.xlsx'
all_labels_dir = 'data/all_labels.xlsx'
training_labels = read_labels(training_labels_dir)
all_labels = read_labels(all_labels_dir)

gestures = task1.calculate_pca(1, 4)[0]
unlabeled_gestures, labeled_gestures = split_data(gestures, all_labels, training_labels)
predictions = KNN(unlabeled_gestures, labeled_gestures)
accuracy = make_shift_accuracy(predictions)
print("KNN accuracy: ", accuracy)

# if __name__ == '__main__':
#     main()