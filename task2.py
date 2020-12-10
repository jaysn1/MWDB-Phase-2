# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 22:52:18 2020

@author: Jaysn
"""
from Phase2 import task1
from task1 import task1_initial_setup
from ppr_classification import classifier, read_files
import pandas as pd
import numpy as np
import json


def read_labels(labels_dir):
    labels = pd.read_excel(labels_dir)
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

def KNN(unlabeled_gestures, gestures, training_labels):
    predictions = []
    for query_id, query_values in unlabeled_gestures.items():
        prediction_id = calculate_distances_from_all_gestures_knn(query_values, gestures)
        temp = [query_id, training_labels[training_labels['id'] == prediction_id].label.values[0], training_labels[training_labels['id'] == prediction_id].label_code.values[0]]
        predictions.append(temp)
    return pd.DataFrame(predictions, columns=('id', 'label', 'label_code'))

def split_data(gestures, training_labels):
    labeled_gestures = {}
    unlabeled_gestures = gestures.copy()
    for gesture_id in training_labels['id']:
        labeled_gestures[gesture_id] = gestures[str(gesture_id)]
        del unlabeled_gestures[str(gesture_id)]
    return unlabeled_gestures, labeled_gestures

def make_shift_accuracy(predictions):
    correct = 0
    for truth, prediction in predictions.items():
        if abs(int(truth) - int(prediction)) < 100:
            correct += 1
    accuracy = correct / len(predictions)
    return accuracy

class Node:
    def __init__(self, y_pred):
        self.y_pred = y_pred
        self.f_i = 0
        self.threshold = 0
        self.l = None
        self.r = None
    

class DecisionTreeClassifier:
    def __init__(self, max_depth=0):
        self.max_depth = max_depth
        self.y_count = 0
        self.dimensionality = 0
        self.root = None
    
    def fit(self, X, y):
        self.y_count = len(set(y))
        self.dimensionality = X.shape[1]
        self.root = self.extend(X, y)
        self.mapping = {0:"combinato", 1:"daccordo", 2:"vattene"}

    def predict(self, X):
        
        return pd.DataFrame([[key, self.mapping[self.predict_single(value)], self.predict_single(value)] for key, value in X.items()], columns=('id', 'label', 'label_code'))
    
    def get_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.y_count)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.dimensionality):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.y_count
            num_right = num_parent[:]
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] = num_left[c] + 1
                num_right[c] = num_right[c] - 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.y_count))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.y_count))
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr
    
    def extend(self, X, y, depth=0):
        y_pred = np.argmax([np.sum(y == i) for i in range(self.y_count)])
        node = Node(y_pred=y_pred)
        if depth < self.max_depth:
            idx, thr = self.get_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.f_i = idx
                node.threshold = thr
                node.l = self.extend(X_left, y_left, depth + 1)
                node.r = self.extend(X_right, y_right, depth + 1)
        return node
    
    def predict_single(self, x):
        node = self.root
        while node.l:
            node = node.l if x[node.f_i] < node.threshold else node.r
        return node.y_pred


def main(): 
    K = 25
    training_labels_dir = 'data/labels.xlsx'
    training_labels = read_labels(training_labels_dir)
    
    training_labels['label'] = training_labels['label'].astype('category')
    training_labels['label_code'] = training_labels['label'].cat.codes
    gestures = task1.calculate_pca(1, 10)[0]    

    unlabeled_gestures, labeled_gestures = split_data(gestures, training_labels)
    X_train, X_test, y_train = pd.DataFrame(labeled_gestures.values()).to_numpy(), pd.DataFrame(unlabeled_gestures.values()).to_numpy(),training_labels['label_code'].to_numpy()
    
    predictions_knn = KNN(unlabeled_gestures, labeled_gestures, training_labels)
    predictions_knn.to_csv("output/KNN_predictions.csv")
    print(f"\nkNN Results(first {K}): ")
    results = predictions_knn.groupby("label").apply(lambda x: list(sorted(x['id'])))
    for k,v in results.to_dict().items():
        print(f"Gesture:- {k} (total {len(v)}): ", v[:K])
    print("kNN results saved in output/KNN_predictions.csv\n")

    # vector_model = 1
    # ges, gids = task1.read_vectors(vector_model)
    # gestures = {i:g for i,g in zip(gids, ges)}
    # unlabeled_gestures, labeled_gestures = split_data(gestures, training_labels)
    # X_train, X_test, y_train = pd.DataFrame(labeled_gestures.values()).to_numpy(), pd.DataFrame(unlabeled_gestures.values()).to_numpy(),training_labels['label_code'].to_numpy()
        
    clf = DecisionTreeClassifier(max_depth=6)
    clf.fit(X_train, y_train)
    clf.mapping = {code:label for code, label in zip(training_labels['label_code'], training_labels['label'])}
    y_pred_test = clf.predict(unlabeled_gestures)
    y_pred_test.to_csv("output/DT_predictions.csv")
    print(f"\nDecision Tree Results(first {K}): ")
    results = y_pred_test.groupby("label").apply(lambda x: list(sorted(x['id'])))
    for k,v in results.to_dict().items():
        print(f"Gesture:- {k} (total {len(v)}): ", v[:K])
    print("Decision Tree results saved in output/DT_predictions.csv\n\n")
    
    # Code to call PPR classifier
    print("For PPR based classification.")
    user_option = int(input("\n 1: DOT\n 2: PCA\n 3: SVD\n 4: NMF\n 5: LDA\n What to use for gesture gesture matrix? "))
    if user_option not in [6,7]:
        vector_model = int(input("What vector model to use (TF: 0, TF-IDF:1)?: "))
    
    data = task1_initial_setup(user_option, vector_model, False)
    graph = pd.DataFrame.from_dict(data).values
    labels = set()
    # Labelled dataset for training
    input_image_label_pair = []
    tr_labels = read_files(training_labels_dir)
    
    d = {}
    for i in tr_labels[0]:
        d[tr_labels[0][i]] = str(tr_labels[1][i])
    # removing extra that has been read from file
    del d['Filename']

    mapping = {v:k for k,v in d.items()}#{"combinato":0, "daccordo":1, "vattene":2}
    for line in d:
        image_id = str(line)
        label = d[line]
        labels.add(label)
        input_image_label_pair.append([image_id, label, mapping[label]])
    # print("PPR======================")
    
    accuracy = classifier(graph, labels, input_image_label_pair)
    with open("output/PPR_res_labels.json") as f:
        dic = json.load(f)
        print(f"\nPPR classification Results(first {K}): ")
        for k,v in dic.items():
            print(f"Gesture:- {k} (total {len(v)}): ", v[:K])
    print("PPR results saved in output/PPR_res_labels.json")
    
    # print("Personalised PageRank Classifier accuracy: " + str(accuracy))
        
if __name__ == '__main__':
    main()