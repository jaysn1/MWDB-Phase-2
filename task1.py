"""
Phase3 - task 1

Task 1: Representative gesture identification Implement a program which, given a value k, creates a gesture-gesture
similarity graph, such that from each gesture, there are k outgoing edges to k most similar/related imagesgestures to
it. Given n user specified gestures on the graph, the program identifies and visualizes m most dominant gestures using
Personalized Page Rank (PPR) for a user supplied m. See

– “J.-Y. Pan, H.-J. Yang, C. Faloutsos, and P. Duygulu. Automatic multimedia cross-modal correlation discovery. In
KDD, pages 653658, 2004”

for a personalized PageRank formulation based on RandomWalks with re-start.

author: Monil
TODO: Visualise < m > dominant gestures?
"""
from Phase2 import task0a, task0b, task1 as phase2task1
from Phase2.similarity_calculators.dot_product_similarity import dot_product_similarity
from Phase2.similarity_calculators.edit_distance_similarity import edit_distance_similarity
from Phase2.similarity_calculators.distance_for_PCA_SVD_NMF_LDA import calculate_similarity
from Phase2.similarity_calculators.dtw_similarity import dynamic_time_warping
from Phase2.read_word_average_dict import read_word_average_dict
from Phase2.read_sensor_average_std_dict import read_sensor_average_std_dict
from Phase1.helper import min_max_scaler, load_data
from visualize import visualize
from tqdm import tqdm
import networkx as nx
import concurrent.futures

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os, json, pickle


def get_query_for_edit_distance(components, sensors, gesture_id, word_store, s):
    query = {}
    for component_id in components:
        component_words = []
        for sensor_id in range(1, sensors + 1):
            time = 0
            word = (component_id, gesture_id, sensor_id, time)
            sensor_words = []
            # Get all the words in the sensor (done this way as partial search doesn't guarantee order)
            while word_store.get(str(word)):
                sensor_words.append(word_store[str(word)])
                time += s
                word = (component_id, gesture_id, sensor_id, time)
            component_words.append(sensor_words)
        query[component_id] = component_words
    return query

def edit_distance_thread(p_vectors, vectors, components, sensors, mega_data_store, s):
    thread_gestures = {}
    for gesture_id in tqdm(p_vectors):
        gesture_similarity = {}
        # Get the query gesture words for each component
        query = mega_data_store[gesture_id]
        for gesture in vectors:
            distance = 1
            for component_id in components:
                for sensor_id in range(1, sensors + 1):
                    query_words = query[component_id][sensor_id - 1]
                    sensor_words = mega_data_store[gesture][component_id][sensor_id - 1]

                    # Calculate and add up edit distances for the sensor wise words between query and gesture DB
                    distance += edit_distance_similarity(query_words, sensor_words)
            gesture_similarity[gesture] = 1 / distance
            
        thread_gestures[gesture_id] = gesture_similarity
    return thread_gestures

# personalize / topic specific page rank : https://www.youtube.com/watch?v=FFNAJUBEfHs
# ref: https://gist.github.com/pj4dev/f636cf54ef9a9d1fd92368f1ef4a3f46
def pagerank(graph, beta, topic, maxiter = 1000):
    matrix = np.zeros((len(graph.nodes), len(graph.nodes)))
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        for neighbour in neighbors:
            matrix[neighbour][node] += beta * 1/len(neighbors)   

    for node in topic:
        neighbours = graph.neighbors(node)
        for neighbour in neighbors:
            matrix[neighbour][node] += (1-beta) * 1/len(topic)

    y = (np.ones((1, len(graph))) * 1.0/len(graph.nodes)).T
    
    count = 0
    i = y.copy()
    while(count <= maxiter):
        tmp = np.matmul(matrix, i)
        if all(tmp == i):
            break
        else:
            i = tmp
        i = tmp
        count += 1
    
    return i

def task1_initial_setup(user_option, vector_model=0, create_vectors=False):
    """
    """
    # need to run task0, 3 from pahse 2
    if create_vectors:
        data = {}
        data['directory'] = input("Enter directory: ")
        data['window_length'] = int(input("Enter window length: "))
        data['shift_length'] = int(input("Enter shift length: "))
        data['resolution'] = int(input("Enter resolution: "))
        
        task0a.main(data)
        task0b.main()
    #############################################  creating gesture-gesture matrix  #####################################
    mapping = {1:'DOT', 2: 'PCA', 3: 'SVD', 4: 'NMF', 5: 'LDA', 6: 'ED', 7: 'DTW'}
    vectors_dir = "intermediate/vectors_dictionary.json"
    parameters_path = "intermediate/data_parameters.json"
    word_vector = "intermediate/word_vector_dict.json"
    word_average_dict_dir = "intermediate/word_avg_dict.json"
    sensor_average_std_dict_dir = "intermediate/sensor_avg_std_dict.json"
    gesture_gesture_similarity_dir = "intermediate/{}_gesture_gesture_similarity_dictionary.pk".format(mapping[user_option])
    k = 4     # top components
    
    if not os.path.exists(gesture_gesture_similarity_dir):
        with open(vectors_dir) as f:
            vectors = json.load(f)
        gesture_gesture_similarity = {}
        if user_option == 1:
            for gesture_id in vectors:
                vector = vectors[gesture_id][vector_model - 1]
                gesture_similarity = {}
                for gesture in vectors:
                    gesture_similarity[gesture] = dot_product_similarity(vector, vectors[gesture][vector_model - 1])
                gesture_gesture_similarity[gesture_id] = gesture_similarity

        elif user_option == 2 or user_option == 3 or user_option == 4 or user_option == 5:
            transformed_data_dir = "intermediate/{}_{}_transformed_data.json".format(mapping[user_option], vector_model)
            
            if not os.path.exists(transformed_data_dir):
                print("Creating transformed vectors.")
                phase2task1.main(user_option-1, vector_model, k)
                
            with open(transformed_data_dir, "r") as f:
                transformed_data = json.load(f)

            for gesture_id in transformed_data:
                query = transformed_data[gesture_id]
                gesture_similarity = {}
                for gesture, gesture_data in transformed_data.items():
                    gesture_similarity[gesture] = calculate_similarity(query, gesture_data)
                gesture_gesture_similarity[gesture_id] = gesture_similarity        

        elif user_option == 6:
            threads = 1
            # Initialize default config
            with open(parameters_path) as f:
                parameters = json.load(f)
            s = parameters['shift_length']
            components = ['X', 'Y', 'Z', 'W']
            sensors = 20

            # Open the word vector from task 0
            with open(word_vector) as f:
                word_store = json.load(f)

            mega_data_store = {}
            for gesture_id in vectors:
                mega_data_store[gesture_id] = get_query_for_edit_distance(components, sensors, gesture_id, word_store, s)

            partitioned_vectors = []
            for i in range(threads):
                j = i * len(vectors) // threads
                partitioned_vectors.append(list(vectors.keys())[j:j + len(vectors) // threads])
            partitioned_vectors.append(list(vectors.keys())[threads * (len(vectors) // threads):])

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = [
                    executor.submit(edit_distance_thread, p_vector, vectors.keys(), components, sensors, mega_data_store, s)
                    for p_vector in partitioned_vectors]

            for f in future:
                gesture_gesture_similarity.update(f.result())

        elif user_option == 7:
            word_average_dict = read_word_average_dict(word_average_dict_dir)
            sensor_avg_std_dict = read_sensor_average_std_dict(sensor_average_std_dict_dir)

            for gesture_id in tqdm(word_average_dict):
                gesture_similarity = {}
                for gesture in word_average_dict:
                    gesture_similarity[gesture] = 1
                    for components_id in word_average_dict[gesture_id]:
                        for sensor_id in word_average_dict[gesture_id][components_id]:
                            weight = abs(sensor_avg_std_dict[gesture][components_id][sensor_id][0] -
                                         sensor_avg_std_dict[gesture_id][components_id][sensor_id][0])
                            gesture_similarity[gesture] += dynamic_time_warping(
                                word_average_dict[gesture_id][components_id][sensor_id],
                                word_average_dict[gesture][components_id][sensor_id], weight)
                    gesture_similarity[gesture] = 1/gesture_similarity[gesture]
                gesture_gesture_similarity[gesture_id] = gesture_similarity

        with open(gesture_gesture_similarity_dir, "wb") as write_file:
            pickle.dump(gesture_gesture_similarity, write_file)
    else:
        with open(gesture_gesture_similarity_dir, "rb") as read_file:
            gesture_gesture_similarity = pickle.load(read_file)
    return gesture_gesture_similarity

def main(gesture_gesture_matrix, k, m, seed_nodes, beta = 0.6, **kwargs):
    """
    input: k, m, (gestures to create gesture-gesture matrix), n seed nodes
    output: top m most dominant gestures in the g-g matrix
    """
    # 0. Select top k edges and make a adjacent matrix representaion of graph
    # 1. Convert matrix to graph representation 
    # 2. personalise run page rank algo with seed nodes taken from user and beta
    parameters_dir = "intermediate/data_parameters.json"

    G = {}
    for node in gesture_gesture_matrix:
        G[node] = [_[0] for _ in sorted(gesture_gesture_matrix[node].items(), key=lambda x: x[1], reverse=True)][:k]
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

    # graph = nx.relabel_nodes(graph, node_inv_map, copy=True)
    # pos=nx.spring_layout(graph)
    # nx.draw(graph, pos, with_labels = True)
    # pr_scaled = min_max_scaler([pr], feature_range=(100, 1000))

    # nx.draw(graph, pos, node_size = pr_scaled)
    # if '_edge_labels' in kwargs and kwargs['_edge_labels']==True:
    #     edge_labels = dict([((u,v,),d['weight']) for u,v,d in graph.edges(data=True)])
    #     nx.draw_networkx_edge_labels(graph,pos,edge_labels=edge_labels)

    # plt.show()

    ppr_score = {node_inv_map[i]: pr[i] for i in range(len(G))}
    top_m_ppr = [_[0] for _ in sorted(ppr_score.items(), key=lambda x: x[1], reverse=True) ][:m]

    if 'viz' in kwargs and kwargs['viz']:
        with open(parameters_dir) as f:
            data_parameters = json.load(f)
        data = data_parameters['directory']
        resolution = data_parameters['resolution']
        for gesture in top_m_ppr:
            (x, y, z, w) = (np.array(load_data(f"{data}/X/{gesture}.csv")),
                        np.array(load_data(f"{data}/Y/{gesture}.csv")), 
                        np.array(load_data(f"{data}/Z/{gesture}.csv")), 
                        np.array(load_data(f"{data}/W/{gesture}.csv")))

            plt.figure(figsize=(18,10))
            visualize(x, plt.subplot(2,2,1), None, resolution, f'{gesture}-X')
            visualize(y, plt.subplot(2,2,2), None, resolution, f'{gesture}-Y')
            visualize(z, plt.subplot(2,2,3), None, resolution, f'{gesture}-Z')
            visualize(w, plt.subplot(2,2,4), None, resolution, f'{gesture}-W')
            
            plt.show(block=False)
        plt.show()
    return top_m_ppr



if __name__=="__main__":

    gesture_gesture_similarity = task1_initial_setup(2, 0, False)
    viz = False

    k, m = 10, 5
    seed_nodes = ["1", "2", "3", "4"]
    dominant_gestures = main(gesture_gesture_similarity, k, m, seed_nodes, viz=viz, _edge_labels = False)
    print(dominant_gestures)