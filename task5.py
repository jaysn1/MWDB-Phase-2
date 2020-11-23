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

def create_adjacency_graph(vectors, gestures, input_vector_dimension):
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

def perform_relevance_feeback(vectors, relevant_gestures, irrelevant_gestures):
    gestures = list(vectors.keys())
    input_vector_dimension = len(vectors[gestures[0]][0])
    adjacency_graph = create_adjacency_graph(vectors, gestures, input_vector_dimension)
    relevant_gesture_ids = retrieve_gesture_ids(relevant_gestures, gestures, input_vector_dimension)
    irrelevant_gestures_ids = retrieve_gesture_ids(irrelevant_gestures, gestures, input_vector_dimension)

    word_weight_dict = dict()
    if len(relevant_gesture_ids) > 0:
        print("Running PPR on graph with relevant gestures as seed...")
        pgr_output = pageRank(adjacency_graph, relevant_gesture_ids, beta = 0.9)
        process_page_rank_output(pgr_output, input_vector_dimension, word_weight_dict)

    if len(irrelevant_gestures_ids) > 0:
        print("Running PPR on graph with irrelevant gestures as seed...")
        pgr_output = pageRank(adjacency_graph, irrelevant_gestures_ids, beta = 0.9)
        process_page_rank_output(pgr_output, input_vector_dimension, word_weight_dict)

    modified_query_vector = vectors['1'][0]
    for k,v in word_weight_dict.items():
        if v > 0:
            modified_query_vector[k] = 1
        else:
            modified_query_vector[k] = 0
    return modified_query_vector

def main():
    file_name = "Phase2/intermediate/vectors_dictionary.json"
    vectors = deserialize_vector_file(file_name)
    gestures = list(vectors.keys())
    input_vector_dimension = len(vectors[gestures[0]][0])

    lsh = LSH(num_layers=4, num_hash_per_layer=8, input_dimension=input_vector_dimension, is_similarity_matrix=False)
    lsh.index(vectors)
    lsh.query(point=vectors['1'][0], t=15, vectors=vectors)

    relevant_gestures = ['1', '30', '9', '11']
    irrelevant_gestures = ['559', '566', '577', '567']

    modified_query_vector = perform_relevance_feeback(vectors, relevant_gestures=relevant_gestures, irrelevant_gestures=irrelevant_gestures)

    lsh.query(point=modified_query_vector, t=15, vectors=vectors)

if __name__=="__main__":
    main()