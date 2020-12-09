from task1 import pageRank, task1_initial_setup
import json 

import numpy as np
import pandas as pd
import networkx as nx

from os import listdir, makedirs
from os.path import isfile, join


def map_files():
    mypath = "data/W/"

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    f = {}
    t = 0

    for i in onlyfiles:
        f[t] = i[:-4]
        t += 1

    return f

# Different PPR Code for classification
def pageRank(graph, input_images, beta=0.85, epsilon=0.000001):
    nodes = len(graph)

    # Keeping record of index and gestures IDs (Since gesture ID values may not be in order)
    # training_labels_dir = 'data/all_labels.xlsx'
    # tr_labels = read_files(training_labels_dir)
    tr_labels = map_files()

    # print(nodes)
    # index_img_dict = tr_labels[0]
    img_index_dict = {str(y):str(x) for x,y in tr_labels.items()} # Step to make a dict with the gestures IDs as key and their serial numbers as values

    M = graph / np.sum(graph, axis=0)

    # Initializing Teleportation matrix and Page Rank Scores with Zeros for all images
    teleportation_matrix = np.zeros(nodes)
    pageRankScores = np.zeros(nodes)

    # print(len(teleportation_matrix))

    # Updating Teleportation and Page Rank Score Matrices with 1/num_of_input images for the input images.
    for image_id in input_images:
        teleportation_matrix[int(img_index_dict[image_id])] = 1 / len(input_images)
        pageRankScores[int(img_index_dict[image_id])] = 1 / len(input_images)

    # Calculating Page Rank Scores
    while True:
        oldPageRankScores = pageRankScores
        pageRankScores = (beta * np.dot(M, pageRankScores)) + ((1 - beta) * teleportation_matrix)
        if np.linalg.norm(pageRankScores - oldPageRankScores) < epsilon:
            break

    # Normalizing & Returning Page Rank Scores
    return pageRankScores / sum(pageRankScores)

def read_files(labels_dir):
    labels = pd.read_excel(labels_dir, header=None)
    return labels.to_dict()

# Check accuracy of PPR Classifier
def acc_check(results, tr_labels):
    correct_labels = {}
    for i in tr_labels:
        correct_labels[tr_labels[i]] = str(tr_labels[i])
    
    # Calculating accuracy 
    acc = 0
    for i in correct_labels:
        for j in results:
            if j == correct_labels[i]:
                # print(final_output[j])
                if str(i) in results[j]:
                    acc = acc + 1
    
    # print(len(correct_labels))
    acc = (acc / len(correct_labels)) * 100
    return acc

def labelling(output):
    # Assigning label to each image based on the highest Page Rank score value
    image_labels = []   
    for i in range(0, len(output[0][1])):
        compare = []
        for item in output:
            compare.append([item[0], i, item[1][i]])
        image_labels.append(sorted(compare, reverse=True, key=lambda x: x[2])[0])

    # Dataset with the correct label for all the datasets

    check = map_files()

    # actual_labels = 'data/all_labels.xlsx'
    # tr_labels = read_files(actual_labels)
    # index_img_dict = tr_labels
    index_img_dict = {str(x):str(y) for x,y in check.items()} # Step to convert values to string type

    final = []
    for elem in image_labels:
        final.append(index_img_dict[str(elem[1])] + ": " + elem[0])

    # Grouping images for each labels
    final_output = dict()
    for elem in image_labels:
        if elem[0] not in final_output:
            final_output[elem[0]] = []
        final_output[elem[0]].append(index_img_dict[str(elem[1])])

    # File with all predicted labels for the input gestures
    with open('output/PPR_res_labels.json', 'w') as f:
        f.write(json.dumps(final_output, indent = 2))
 
    return final_output

def classifier(graph, labels, input_image_label_pair):
    # Calculating Personalized Page Rank for each label once.
    # print("Running PPR for each label")
    output = []
    # output = {}
    label_dict = dict()
    count = 0
    for label in labels:
        label_dict[str(count)] = label
        count += 1
        input_images = [str(item[0]) for item in input_image_label_pair if item[1] == label]
        # l = work(gr_m, input_images, 0.85)
        output.append([label, pageRank(graph, input_images, beta = 0.85).tolist()])
        # output.append([label, l])
        # output[label] = l

    final = labelling(output)
    # final = label_m(output)

    # actual_labels = 'data/all_labels.xlsx'
    # tr_labels = read_files(actual_labels)
    tr_labels = map_files()
    index_img_dict = tr_labels
    index_img_dict = {str(x):str(y) for x,y in tr_labels.items()} # Step to convert values to string type

    # print(index_img_dict)

    # accuracy = acc_check(final, check_label)
    
    # print(accuracy)

    # return accuracy
    


def main():
    data = task1_initial_setup(1, 0, False)
    # print(data)
    graph = pd.DataFrame.from_dict(data).values

    labels = set()

    # Labelled dataset for training
    training_labels_dir = 'data/labels.xlsx'
    # actual_labels = 'data/all_labels.xlsx'
    input_image_label_pair = []
    tr_labels = read_files(training_labels_dir)
    d = {}
    for i in tr_labels[0]:
        d[tr_labels[0][i]] = str(tr_labels[1][i])

    for line in d:
        image_id = str(line)
        label = d[line]
        labels.add(label)
        input_image_label_pair.append([image_id, label])

    classifier(graph, labels, input_image_label_pair)

if __name__ == '__main__':
    main()