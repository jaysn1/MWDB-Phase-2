# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:56:53 2020

@author: Jaysn
"""
import json, os
from similarity_calculators.dot_product_similarity import dot_product_similarity
from similarity_calculators.distance_for_PCA_SVD_NMF_LDA import calculate_similarity
from similarity_calculators.edit_distance_similarity import edit_distance_similarity
from similarity_calculators.dtw_similarity import dynamic_time_warping

import concurrent.futures
from read_word_average_dict import read_word_average_dict
from read_sensor_average_std_dict import read_sensor_average_std_dict
from task3_util import SVD_gesture_gesture, NMF_gesture_gesture, store_gesture_gesture_score_dict
from tqdm import tqdm

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


def main(user_option, p, latent_semantics_option):
    mapping, mapping_2 = {1:'DOT', 2: 'PCA', 3: 'SVD', 4: 'NMF', 5: 'LDA', 6: 'ED', 7: 'DTW'}, {1: 'SVD', 2: 'NMF'}
    vectors_dir = "intermediate/vectors_dictionary.json"
    parameters_path = "intermediate/data_parameters.json"
    word_vector = "intermediate/word_vector_dict.json"
    word_average_dict_dir = "intermediate/word_avg_dict.json"
    sensor_average_std_dict_dir = "intermediate/sensor_avg_std_dict.json"
    gesture_gesture_similarity_dir = "intermediate/{}_{}_gesture_gesture_similarity_dictionary.json".format(mapping[user_option], mapping_2[latent_semantics_option])
    gesture_gesture_score_json_file_path = "intermediate/{}_gesture_gesture_score.json".format(mapping_2[latent_semantics_option])
    gesture_gesture_score_dir = "intermediate/{}_{}_gesture_gesture_score.txt".format(mapping[user_option], mapping_2[latent_semantics_option])
    transformed_gestures_gestures_dir = "intermediate/{}_transformed_gesture_gesture_data.json".format(mapping_2[latent_semantics_option])

    with open(vectors_dir) as f:
        vectors = json.load(f)

    gesture_gesture_similarity = {}
    if user_option == 1:
        vector_model = int(input("Enter vector model to use? (1: TF, 2: TF-IDF): "))        
        for gesture_id in vectors:
            vector = vectors[gesture_id][vector_model - 1]
            gesture_similarity = {}
            for gesture in vectors:
                gesture_similarity[gesture] = dot_product_similarity(vector, vectors[gesture][vector_model - 1])
            gesture_gesture_similarity[gesture_id] = gesture_similarity

    elif user_option == 2 or user_option == 3 or user_option == 4 or user_option == 5:
        vector_model = int(input("Enter vector model to use? (1: TF, 2: TF-IDF): "))
        transformed_data_dir = "intermediate/{}_{}_transformed_data.json".format(mapping[user_option], vector_model)
        
        if not os.path.exists(transformed_data_dir):
            raise ValueError("Please run task 1 with proper inputs.")
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

        for gesture_id in word_average_dict:
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
                gesture_similarity[gesture] = (gesture_similarity[gesture]) ** -1
            gesture_gesture_similarity[gesture_id] = gesture_similarity

    with open(gesture_gesture_similarity_dir, "w") as write_file:
        json.dump(gesture_gesture_similarity, write_file)
    
    gesture_scores = []
    if latent_semantics_option == 1:
        transformed_gestures_gestures_dict, gesture_scores = SVD_gesture_gesture(p, gesture_gesture_similarity_dir)
    elif latent_semantics_option == 2:
        transformed_gestures_gestures_dict, gesture_scores = NMF_gesture_gesture(p, gesture_gesture_similarity_dir)
    else:
        raise ValueError("Incorrect value of latent semantics.")

    store_gesture_gesture_score_dict(gesture_scores, gesture_gesture_score_dir)

    with open(transformed_gestures_gestures_dir, "w") as f:
        json.dump(transformed_gestures_gestures_dict, f)

    with open(gesture_gesture_score_json_file_path, "w") as f:
            json.dump(gesture_scores, f)

    with open("intermediate/data_parameters.json") as f:
        data_parameters = json.load(f)
    data_parameters['p'] = p
    with open("intermediate/data_parameters.json", "w") as f:
        json.dump(data_parameters, f)


    print("Results generated as \n\t{0}".format(gesture_gesture_score_dir))

if __name__=='__main__':
    print("""
    1 - Dot Product
    2 - PCA
    3 - SVD
    4 - NMF
    5 - LDA
    6 - Edit Distance
    7 - DTW Distance""")
    user_option = int(input("Enter method to use to find gesture-gesture similarity: "))
    print("""
    1 - SVD
    2 - NMF""")
    latent_semantics_option = int(input("Enter method to find top-p latent semantics: "))

    p = int(input("Enter the desired p-value (for top-p principal components): "))

    task3.main(user_option, p, latent_semantics_option)

    main(user_option, p, latent_semantics_option)
