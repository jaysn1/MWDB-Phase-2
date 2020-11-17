from task1 import pagerank, task1_initial_setup
import json 
# import time 

import numpy as np
import pandas as pd
import networkx as nx

def work(gesture_gesture_matrix, seed_nodes, beta):
    G = {}
    for node in gesture_gesture_matrix:
        G[node] = [_[0] for _ in sorted(gesture_gesture_matrix[node].items(), key=lambda x: x[1], reverse=True)]
    node_map = {node:i for i,node in enumerate(G.keys())}
    node_inv_map = {i:node for i,node in enumerate(G.keys())}

    graph = {}
    for k in G.keys():
        graph[node_map[k]] = []
        for neighbour in G[k]:
            graph[node_map[k]].append(node_map[neighbour])

    topic = [node_map[_] for _ in seed_nodes]
    G = np.zeros((len(graph), len(graph)))
    for node, neighbours in graph.items():
        for neighbour in neighbours:
            G[node][neighbour] = 1
    graph = nx.convert_matrix.from_numpy_matrix(G)
    pr = pagerank(graph, beta, topic)
    ppr_score = {node_inv_map[i]: pr[i] for i in range(len(G))}
    return ppr_score

def pageRank(graph, input_images, beta=0.85, epsilon=0.000001):
    nodes = len(graph)

    # Dictionaries mapping index number of adjacency matrix to the image IDs and vice-a-versa
    # index_img_dict = dict(csv.reader(open('IndexToImage.csv', 'r')))
    # img_index_dict = dict([[v, k] for k, v in index_img_dict.items()])
    
    training_labels_dir = 'data/all_labels.xlsx'
    tr_labels = read_labels(training_labels_dir)
    index_img_dict = tr_labels[0]
    img_index_dict = {str(y):str(x) for x,y in tr_labels[0].items()}

    M = graph / np.sum(graph, axis=0)

    # Initializing Teleportation matrix and Page Rank Scores with Zeros for all images
    teleportation_matrix = np.zeros(nodes)
    pageRankScores = np.zeros(nodes)

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

def read_labels(labels_dir):
    labels = pd.read_excel(labels_dir, header=None)
    return labels.to_dict()

def acc_check(results, tr_labels):
    correct_labels = {}
    for i in tr_labels[0]:
        correct_labels[tr_labels[0][i]] = str(tr_labels[1][i])
    
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


def main():
    print('Loading Image-Image Similarity Matrix...')
    data = task1_initial_setup(1, 0, False)
    graph = pd.DataFrame.from_dict(data).values

    labels = set()

    # Labelled dataset for training
    training_labels_dir = 'data/sample_training_labels.xlsx'
    input_image_label_pair = []
    tr_labels = read_labels(training_labels_dir)
    d = {}
    for i in tr_labels[0]:
        d[tr_labels[0][i]] = str(tr_labels[1][i])

    for line in d:
        image_id = str(line)
        label = d[line]
        labels.add(label)
        input_image_label_pair.append([image_id, label])

    # Calculating Personalized Page Rank for each label once.
    output = []
    label_dict = dict()
    count = 0
    for label in labels:
        label_dict[str(count)] = label
        count += 1
        print('Calculating Personalised Page Rank for label : ' + label)
        input_images = [str(item[0]) for item in input_image_label_pair if item[1] == label]
        # gr, topic = work(graph, input_images)
        # # graph = nx.convert_matrix.from_numpy_matrix(graph)
        output.append([label, pageRank(graph, input_images, beta = 0.85).tolist()])

    # Assigning label to each image based on the highest Page Rank score value
    image_labels = []   
    for i in range(0, len(output[0][1])):
        compare = []
        for item in output:
            compare.append([item[0], i, item[1][i]])
        image_labels.append(sorted(compare, reverse=True, key=lambda x: x[2])[0])

    # Dataset with the correct label for all the datasets
    actual_labels = 'data/all_labels.xlsx'
    tr_labels = read_labels(actual_labels)
    index_img_dict = tr_labels[0]
    index_img_dict = {str(x):str(y) for x,y in tr_labels[0].items()}

    # Grouping images for each labels
    final_output = dict()
    for elem in image_labels:
        if elem[0] not in final_output:
            final_output[elem[0]] = []
        final_output[elem[0]].append(index_img_dict[str(elem[1])])
    # print(final_output)
    
    with open('res_labels.json', 'w') as f:
        f.write(json.dumps(final_output, indent = 2))

    accuracy = acc_check(final_output, tr_labels)
    print(accuracy)


if __name__ == '__main__':
    main()