import json

from networkx.generators.classic import complete_graph
from numpy.lib.financial import irr
from lsh import LSH
import numpy as np

"""
Deserialize vector input
"""
def deserialize_vector_file(file_name):
    with open(file_name, "r") as read_file:
        vector = json.load(read_file)
    return vector

"""
Deserialize word position dictionary
"""
def deserialize_word_position_dictionary(file_name):
    with open(file_name, "r") as read_file:
        word_position_dictionary = json.load(read_file)
    return word_position_dictionary

"""
Deserialize component position dictionary
"""
def deserialize_component_position_dictionary(file_name):
    with open(file_name, "r") as read_file:
        component_position_dictionary = json.load(read_file)
    return component_position_dictionary

"""
Deserialize sensor position dictionary
"""
def deserialize_sensor_position_dictionary(file_name):
    with open(file_name, "r") as read_file:
        sensor_position_dictionary = json.load(read_file)
    return sensor_position_dictionary

# Different PPR Code for classification
def pageRank(graph, seed_gestures, beta=0.85, epsilon=0.000001):
    nodes = len(graph)

    # # Keeping record of index and gestures IDs (Since gesture ID values may not be in order)
    M = graph / np.sum(graph, axis=0)

    # Initializing Teleportation matrix and Page Rank Scores with Zeros for all images
    teleportation_matrix = np.zeros(nodes)
    pageRankScores = np.zeros(nodes)

    # Updating Teleportation and Page Rank Score Matrices with 1/num_of_input images for the input images.
    for image_id in seed_gestures:
        teleportation_matrix[image_id] = 1 / len(seed_gestures)
        pageRankScores[image_id] = 1 / len(seed_gestures)

    # Calculating Page Rank Scores
    while True:
        oldPageRankScores = pageRankScores
        pageRankScores = (beta * np.dot(M, pageRankScores)) + ((1 - beta) * teleportation_matrix)
        if np.linalg.norm(pageRankScores - oldPageRankScores) < epsilon:
            break

    # Normalizing & Returning Page Rank Scores
    return pageRankScores / sum(pageRankScores)

def create_adjacency_graph_for_words(vectors, gestures, input_vector_dimension):
    # no edges between gestures
    gestures_gesture_array = np.zeros((len(gestures), len(gestures)))
    # no edges between words
    words_words_array = np.zeros((input_vector_dimension, input_vector_dimension))

    gesture_word_array = []
    for k,v in vectors.items():
        gesture_word_array.append(v[0])
    # gestures word adjacency matrix
    gesture_word_array = np.array(gesture_word_array)

    # transpose gesure word adjacency matrix
    word_gesture_array = gesture_word_array.T

    top_half = np.hstack((words_words_array, word_gesture_array))
    bottom_half = np.hstack((gesture_word_array, gestures_gesture_array))
    adjacency_matrix = np.vstack((top_half, bottom_half))
    return adjacency_matrix

# def create_adjacency_graph_for_components(vectors, gestures, input_vector_dimension, component_position_dictionary):
#     # no edges between gestures
#     gestures_gesture_array = np.zeros((len(gestures), len(gestures)))
#     # no edges between words
#     components_components_array = np.zeros((input_vector_dimension, input_vector_dimension))

#     list_of_all_components = []
#     position_to_component_dictionary = {}
#     for component, positions in component_position_dictionary.items():
#         list_of_all_components.append(component)
#         for position in positions:
#             position_to_component_dictionary[position] = component

#     gesture_component_array = []
#     for k,v in vectors.items():
#         gesture = k
#         tf_vector = v[0]
#         gesture_component_edges = [0]*len(list_of_all_components)
#         for i in range(len(tf_vector)):
#             gesture_component_edges[list_of_all_components.index(position_to_component_dictionary[i])] += tf_vector[i]
#         gesture_component_array.append(gesture_component_edges)

#     # # gestures word adjacency matrix
#     gesture_component_array = np.array(gesture_component_array)
#     print(gesture_component_array)

#     # # transpose gesure word adjacency matrix
#     component_gesture_array = gesture_component_array.T

#     top_half = np.hstack((components_components_array, component_gesture_array))
#     bottom_half = np.hstack((gesture_component_array, gestures_gesture_array))
#     adjacency_matrix = np.vstack((top_half, bottom_half))
#     return adjacency_matrix

# def create_adjacency_graph_for_sensors(vectors, gestures, input_vector_dimension, sensor_position_dictionary):
#     # no edges between gestures
#     gestures_gesture_array = np.zeros((len(gestures), len(gestures)))
#     # no edges between words
#     sensors_sensors_array = np.zeros((input_vector_dimension, input_vector_dimension))

#     list_of_all_sensors = []
#     position_to_sensor_dictionary = {}
#     for sensor, positions in sensor_position_dictionary.items():
#         list_of_all_sensors.append(sensor)
#         for position in positions:
#             position_to_sensor_dictionary[position] = sensor


#     gesture_sensor_array = []
#     for k,v in vectors.items():
#         gesture = k
#         tf_vector = v[0]
#         gesture_sensor_edges = [0]*len(list_of_all_sensors)
#         for i in range(len(tf_vector)):
#             gesture_sensor_edges[list_of_all_sensors.index(position_to_sensor_dictionary[i])] += tf_vector[i]
#         gesture_sensor_array.append(gesture_sensor_edges)

#     # gestures word adjacency matrix
#     gesture_sensor_array = np.array(gesture_sensor_array)
#     print(gesture_sensor_array)

#     # transpose gesure word adjacency matrix
#     sensor_gesture_array = gesture_sensor_array.T

#     top_half = np.hstack((sensors_sensors_array, sensor_gesture_array))
#     bottom_half = np.hstack((gesture_sensor_array, gestures_gesture_array))
#     adjacency_matrix = np.vstack((top_half, bottom_half))
#     return adjacency_matrix

def retrieve_gesture_ids(gesture_subset, gestures, input_vector_dimension):
    gestures_ids = []
    for b in gesture_subset:
        gestures_ids.append(input_vector_dimension + gestures.index(b))
    return gestures_ids

def process_page_rank_output(pgr_output, input_vector_dimension, word_weight_dict):
    result = []
    for i,n in enumerate(pgr_output):
        result.append((i,n))
    result.sort(key=lambda x: abs(x[1]), reverse=True)

    contribution_of_words = 0
    for node in result:
        if node[0] < input_vector_dimension:
            contribution_of_words += node[1]

    for node in result:
        if node[0] < input_vector_dimension:
            if node[0] not in word_weight_dict:
                word_weight_dict[node[0]] = (node[1]/contribution_of_words)*100
            else:
                word_weight_dict[node[0]] = word_weight_dict[node[0]] - ((node[1]/contribution_of_words)*100)
    return word_weight_dict

def perform_relevance_feeback(query_gesture, vectors, vector_model, relevant_gestures, irrelevant_gestures, word_position_dictionary, component_position_dictionary, sensor_position_dictionary):
    gestures = list(vectors.keys())
    # if choose_relative_importance_type == 0:
    input_vector_dimension = len(vectors[gestures[0]][0])
    adjacency_graph = create_adjacency_graph_for_words(vectors, gestures, input_vector_dimension)
    # elif choose_relative_importance_type == 1:
    #     input_vector_dimension = 4
    #     adjacency_graph = create_adjacency_graph_for_components(vectors, gestures, input_vector_dimension, component_position_dictionary)
    # else:
    #     input_vector_dimension = 20
    #     adjacency_graph = create_adjacency_graph_for_sensors(vectors, gestures, input_vector_dimension, sensor_position_dictionary)
    relevant_gesture_ids = retrieve_gesture_ids(relevant_gestures, gestures, input_vector_dimension)
    irrelevant_gestures_ids = retrieve_gesture_ids(irrelevant_gestures, gestures, input_vector_dimension)

    word_weight_dict = dict()
    if len(relevant_gesture_ids) > 0:
        print("\nRunning PPR on graph with relevant gestures as seed...")
        pgr_output = pageRank(adjacency_graph, relevant_gesture_ids, beta = 0.9)
        process_page_rank_output(pgr_output, input_vector_dimension, word_weight_dict)

    if len(irrelevant_gestures_ids) > 0:
        print("Running PPR on graph with irrelevant gestures as seed...")
        pgr_output = pageRank(adjacency_graph, irrelevant_gestures_ids, beta = 0.9)
        process_page_rank_output(pgr_output, input_vector_dimension, word_weight_dict)

    list_of_all_words = []
    position_to_word_dictionary = {}
    for word, position in word_position_dictionary.items():
        list_of_all_words.append(word)
        position_to_word_dictionary[position] = word

    list_of_all_components = []
    position_to_component_dictionary = {}
    for component, positions in component_position_dictionary.items():
        list_of_all_components.append(component)
        for position in positions:
            position_to_component_dictionary[position] = component

    list_of_all_sensors = []
    position_to_sensor_dictionary = {}
    for sensor, positions in sensor_position_dictionary.items():
        list_of_all_sensors.append(sensor)
        for position in positions:
            position_to_sensor_dictionary[position] = sensor

    modified_query_vector = vectors[query_gesture][vector_model]
    word_weight_list = [(0,0)]*input_vector_dimension
    component_weight_list = [(0,0)]*4
    sensor_weight_list = [(0,0)]*20

    # print("Original query vector : " + str(modified_query_vector))

    for k,v in word_weight_dict.items():
        word_index = list_of_all_words.index(position_to_word_dictionary[k])
        component_index = list_of_all_components.index(position_to_component_dictionary[k])
        sensor_index = list_of_all_sensors.index(position_to_sensor_dictionary[k])
        word_weight_list[word_index] = (position_to_word_dictionary[k], word_weight_list[word_index][1] + v)
        component_weight_list[component_index] = (position_to_component_dictionary[k], component_weight_list[component_index][1] + v)
        sensor_weight_list[sensor_index] = (position_to_sensor_dictionary[k], sensor_weight_list[sensor_index][1] + v)
        if v > 0:
            modified_query_vector[k] = 1
        else:
            modified_query_vector[k] = 0

    # print("Modified query vector : " + str(modified_query_vector))

    word_weight_list.sort(key=lambda x: x[1], reverse=True)
    component_weight_list.sort(key=lambda x: x[1], reverse=True)
    sensor_weight_list.sort(key=lambda x: x[1], reverse=True)

    print("\nDominant words:")
    print("\t" + str(word_weight_list[0:10]))
    print("\nDominant components:")
    print("\t" + str(component_weight_list[0:10]))
    print("\nDominant sensors:")
    print("\t" + str(sensor_weight_list[0:10]))

    return modified_query_vector

def main():
    vectors_file_name = "Phase2/intermediate/vectors_dictionary.json"
    word_position_dictionary_file_name = "Phase2/intermediate/word_position_dictionary.json"
    component_position_dictionary_file_name = "Phase2/intermediate/component_position_dictionary.json"
    sensor_position_dictionary_file_name = "Phase2/intermediate/sensor_position_dictionary.json"

    vectors = deserialize_vector_file(vectors_file_name)
    word_position_dictionary = deserialize_word_position_dictionary(word_position_dictionary_file_name)
    component_position_dictionary = deserialize_component_position_dictionary(component_position_dictionary_file_name)
    sensor_position_dictionary = deserialize_sensor_position_dictionary(sensor_position_dictionary_file_name)
    
    query_gesture='1'
    vector_model=0

    gestures = list(vectors.keys())
    input_vector_dimension = len(vectors[gestures[0]][vector_model])

    lsh = LSH(num_layers=4, num_hash_per_layer=8, input_dimension=input_vector_dimension, is_vector_matrix=True, vector_model=vector_model)
    lsh.index(vectors)
    lsh.query(point=vectors[query_gesture][vector_model], t=15, vectors=vectors)

    relevant_gestures = ['1', '30', '9', '11']
    irrelevant_gestures = ['559', '566', '577', '567']

    modified_query_vector = perform_relevance_feeback(query_gesture, vectors, vector_model=vector_model, relevant_gestures=relevant_gestures, irrelevant_gestures=irrelevant_gestures, word_position_dictionary=word_position_dictionary,
     component_position_dictionary=component_position_dictionary, sensor_position_dictionary=sensor_position_dictionary)

    lsh.query(point=modified_query_vector, t=15, vectors=vectors)

if __name__=="__main__":
    main()