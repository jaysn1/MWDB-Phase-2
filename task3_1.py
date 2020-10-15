# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:56:53 2020

@author: Jaysn
"""
import json
from similarity_calculators.dot_product_similarity import dot_product_similarity
from similarity_calculators.edit_distance_similarity import edit_distance_similarity
from tqdm import tqdm

def get_query_for_edit_distance(components, sensors, gesture_id, word_store, s):
    query = {}
    for component_id in components:
            component_words = []
            for sensor_id in range(1, sensors+1):
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

# def edit_distance_thread(vectors, components, sensors, word_store, s):
#     thread_gestures = []
#     for gesture_id in tqdm(vectors):
#         gesture_similarity = []
#         # Get the query gesture words for each component
#         query = get_query_for_edit_distance(components, sensors, gesture_id, word_store, s)
#         for gesture in vectors:
#             distance = 0
#             for component_id in components:
#                 for sensor_id in range(1, sensors+1):
#                     time = 0
#                     word = (component_id, gesture, sensor_id, time)
#                     sensor_words = []
#                     # Get all the words in the sensor (done this way as partial search doesn't guarantee order)
#                     while word_store.get(str(word)):
#                         sensor_words.append(word_store[str(word)])
#                         time += s
#                         word = (component_id, gesture, sensor_id, time)
#                     query_words = query[component_id][sensor_id-1]
                    
#                     # Calculate and add up edit distances for the sensor wise words between query and gesture DB
#                     distance += edit_distance_similarity(query_words, sensor_words)
#             if distance != 0:
#                 gesture_similarity.append(1/distance)
#             else:
#                 gesture_similarity.append(float("inf"))
#         thread_gestures.append(gesture_similarity)
#     return thread_gestures
    
def main():
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  Phase 2 -Task 2                                                        │
  │                                                                         │
  │                                                                         │
  │    1 - Dot Product                                                      │
  │    2 - PCA                                                              │
  │    3 - SVD                                                              │
  │    4 - NMF                                                              │
  │    5 - LDA                                                              │
  │    6 - Edit Distance                                                    │
  │    7 - DTW Distance                                                     │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘""")
    user_option = int(input("\nEnter method to use to find similarity: "))
    
    vectors_dir="intermediate/vectors_dictionary.json"
    gesture_gesture_similarity_dir="intermediate/gesture_gesture_similarity_dictionary.json"
    parameters_path="intermediate/data_parameters.json"
    word_vector="intermediate/word_vector_dict.json"
    with open(vectors_dir) as f:
        vectors = json.load(f)
    
    if user_option == 1:
        vector_model = int(input("Enter vector model to use? (1: TF, 2: TF-IDF): "))
        gesture_gesture_similarity = {}
        for gesture_id in vectors:
            vector = vectors[gesture_id][vector_model-1]
            gesture_similarity = {}
            for gesture in vectors:
                gesture_similarity[gesture] = dot_product_similarity(vector,vectors[gesture][vector_model-1])
            gesture_gesture_similarity[gesture_id] = gesture_similarity
        
        with open(gesture_gesture_similarity_dir, "w") as write_file:
            json.dump(gesture_gesture_similarity, write_file)
    
    # elif user_option == 6:
    #     threads = 10
    #     # Initialize default config
    #     with open(parameters_path) as f:
    #             parameters = json.load(f)
    #     s = parameters['shift_length']
    #     components = ['X', 'Y', 'Z', 'W']
    #     sensors = 20
        
    #     # Open the word vector from task 0
    #     with open(word_vector) as f:
    #         word_store = json.load(f)
        
    #     partition_length = len(vectors)//threads
        
    #     edit_distance_thread(vectors, components, sensors, word_store, s)
        
    #     gesture_gesture_similarity = []
    #     # Iterating over gestures from vectors to keep the gesture_gesture_similarity matrix consistent
    #     for gesture_id in tqdm(vectors):
    #         gesture_similarity = []
    #         # Get the query gesture words for each component
    #         query = get_query_for_edit_distance(components, sensors, gesture_id, word_store, s)
    #         for gesture in vectors:
    #             distance = 0
    #             for component_id in components:
    #                 for sensor_id in range(1, sensors+1):
    #                     time = 0
    #                     word = (component_id, gesture, sensor_id, time)
    #                     sensor_words = []
    #                     # Get all the words in the sensor (done this way as partial search doesn't guarantee order)
    #                     while word_store.get(str(word)):
    #                         sensor_words.append(word_store[str(word)])
    #                         time += s
    #                         word = (component_id, gesture, sensor_id, time)
    #                     query_words = query[component_id][sensor_id-1]
                        
    #                     # Calculate and add up edit distances for the sensor wise words between query and gesture DB
    #                     distance += edit_distance_similarity(query_words, sensor_words)
    #             if distance != 0:
    #                 gesture_similarity.append(1/distance)
    #             else:
    #                 gesture_similarity.append(float("inf"))
    #         gesture_gesture_similarity.append(gesture_similarity)
    #     return gesture_gesture_similarity
    
a = main()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            