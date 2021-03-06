from similarity_calculators.dot_product_similarity import dot_product_similarity
from similarity_calculators.distance_for_PCA_SVD_NMF_LDA import calculate_similarity
from similarity_calculators.edit_distance_similarity import edit_distance_similarity
from similarity_calculators.dtw_similarity import dynamic_time_warping, multi_dimension_dynamic_time_warping, derivative_dynamic_time_wraping
from read_word_average_dict import read_word_average_dict
from read_sensor_average_std_dict import read_sensor_average_std_dict
import json, os

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

def main(user_option, gesture_id):
    k = 10
    mapping = {1: 'DOT', 2: 'PCA', 3: 'SVD', 4: 'NMF', 5: 'LDA', 6: 'ED', 7: 'DTW'}
    vectors_dir="intermediate/vectors_dictionary.json"
    word_vector="intermediate/word_vector_dict.json"
    parameters_path="intermediate/data_parameters.json"
    word_average_dict_dir="intermediate/word_avg_dict.json"
    sensor_average_std_dict_dir = "intermediate/sensor_avg_std_dict.json"
    result_dir = "output/{}_{}_top10.txt".format(mapping[user_option], gesture_id)

    with open(vectors_dir) as f:
        vectors = json.load(f)
    
    if user_option == 1:
        vector_model = int(input("Enter vector model to use? (1: TF, 2: TF-IDF): "))
        vector = vectors[gesture_id][vector_model-1]
        similarity = {}
        for gesture in vectors:
            similarity[gesture] = dot_product_similarity(vector,vectors[gesture][vector_model-1])
    
    elif user_option == 2 or user_option == 3 or user_option == 4 or user_option == 5:
        vector_model = int(input("\n\t1: TF\n\t2: TF-IDF \nEnter vector model to use: "))

        transformed_data_dir = "intermediate/{}_{}_transformed_data.json".format(mapping[user_option], vector_model)
        if not os.path.exists(transformed_data_dir):
            raise ValueError("Please run task 1 with proper inputs.")

        with open(transformed_data_dir, "r") as f:
            transformed_data = json.load(f)
        query = transformed_data[gesture_id]
        similarity = {}
        for gesture_id, gesture_data in transformed_data.items():
            similarity[gesture_id] = calculate_similarity(query, gesture_data)
        
    elif user_option == 6:
        # Initialize default config
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
            distance = 1
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
            similarity[gesture] = 1/distance

    elif user_option == 7:
        word_average_dict = read_word_average_dict(word_average_dict_dir)
        sensor_avg_std_dict = read_sensor_average_std_dict(sensor_average_std_dict_dir)

        assert gesture_id in word_average_dict
        similarity = {}
        for gesture in word_average_dict:
            similarity[gesture] = 1
            for components_id in word_average_dict[gesture_id]:
                for sensor_id in word_average_dict[gesture_id][components_id]:
                    weight = abs(sensor_avg_std_dict[gesture][components_id][sensor_id][0] - sensor_avg_std_dict[gesture_id][components_id][sensor_id][0])
                    similarity[gesture] += dynamic_time_warping(word_average_dict[gesture_id][components_id][sensor_id],word_average_dict[gesture][components_id][sensor_id], weight)
            
            similarity[gesture] = 1/similarity[gesture]


    top_k_similar = [("(%s, %2.6f)"%(_)) for _ in sorted(similarity.items(), key=lambda x: [x[1], int(x[0])],reverse=True)[:k]]
    with open(result_dir, "w") as f:
        f.write(", ".join(top_k_similar))
    print("Results are stored in file: {}\n".format(result_dir))

    return ", ".join(top_k_similar)
        
if __name__ == "__main__": 
    main()