# task2
# Implement a program which, given a gesture file and a vector model, finds and ranks the 10 most similar gestures
# in the gesture database, relying on the following:

# – user option #1: dot product of gesture vectors,
# – user options #2, 3, 4, 5: top-k latent sensor semantics (PCA, SVD, NMF, LDA),
# – user option #6: edit distance on symbolic quantized window descriptors (propose an edit cost function among
# symbols),
# – user option #7: DTW distance on average quantized amplitudes

# Results are presented in the form of hgesture, scorei pairs sorted in non-increasing order of similarity scores.
from similarity_calculators.dot_product_similarity import dot_product_similarity
from similarity_calculators.distance_for_PCA_SVD_NMF_LDA import calculate_similarity
from similarity_calculators.edit_distance_similarity import edit_distance_similarity
import json

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

def main():
    k = 10
    vectors_dir="intermediate/vectors_dictionary.json"
    word_vector="intermediate/word_vector_dict.json"
    
    with open(vectors_dir) as f:
        vectors = json.load(f)

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
    
    gesture_id = input("Enter query gesture: ")
    assert gesture_id in vectors
    
    if user_option == 1:
        vector_model = int(input("Enter vector model to use? (1: TF, 2: TF-IDF): "))
        vector = vectors[gesture_id][vector_model-1]
        similarity = {}
        for gesture in vectors:
            similarity[gesture] = dot_product_similarity(vector,vectors[gesture][vector_model-1])
        top_k_similar = [_[0] for _ in sorted(similarity.items(), key=lambda x: x[1],reverse=True)[:k]]
        print(top_k_similar)
    
    elif user_option == 2 or user_option == 3 or user_option == 4 or user_option == 5:
        with open("intermediate/transformed_data.json", "r") as f:
            transformed_data = json.load(f)
        query = transformed_data[gesture_id]
        similarity = {}
        for gesture_id, gesture_data in transformed_data.items():
            similarity[gesture_id] = calculate_similarity(query, gesture_data)
        top_k_similar = [_[0] for _ in sorted(similarity.items(), key=lambda x: x[1],reverse=True)[:k]]
        print(top_k_similar)

    elif user_option == 6:
        # Initialize default config
        parameters_path="intermediate/data_parameters.json"
        with open(parameters_path) as f:
                parameters = json.load(f)
        s = parameters['shift_length']
        components = ['X', 'Y', 'Z', 'W']
        sensors = 20
        
        # Open the word vector from task 0
        with open(word_vector) as f:
            word_store = json.load(f)
        
        # Get the query gesture words for each component
        query = get_query_for_edit_distance(components, sensors, gesture_id, word_store, s)
        
        # Iterate over all gestures in dir, components, sensors to calculate edit distance (1*20*4*60)
        similarity = {}
        
        for gesture in vectors:
            distance = 0
            for component_id in components:
                for sensor_id in range(1, sensors+1):
                    time = 0
                    word = (component_id, gesture, sensor_id, time)
                    sensor_words = []
                    # Get all the words in the sensor (done this way as partial search doesn't guarantee order)
                    while word_store.get(str(word)):
                        sensor_words.append(word_store[str(word)])
                        time += s
                        word = (component_id, gesture, sensor_id, time)
                    query_words = query[component_id][sensor_id-1]
                    
                    # Calculate and add up edit distances for the sensor wise words between query and gesture DB
                    distance += edit_distance_similarity(query_words, sensor_words)
            if distance != 0:
                similarity[gesture] = 1/distance
            else:
                similarity[gesture] = float("inf")
        # Sort the similarities to give the top 10 similar gestures.
        top_k_similar = [_[0] for _ in sorted(similarity.items(), key=lambda x: x[1],reverse=True)[:k]]
        print(top_k_similar)

main()