import os
import json
import math
import fnmatch
import task0a

# Deserialize word average dictionary
def deserialize_word_avg_dict(file_name):
    with open(file_name, "r") as read_file:
        word_avg_dict = json.load(read_file)
    return word_avg_dict

# Deserialize data parameters
def deserialize_data_parameters(file_name):
    with open(file_name, "r") as read_file:
        data = json.load(read_file)
    return data

# Create a dictionary containing the position of each word
# in the vector representations
def generate_word_position_dictionary(list_of_all_words):
    word_position_dictionary = {}
    k=0
    for word in list_of_all_words:
        word_position_dictionary[word] = k
        k += 1
    return word_position_dictionary

# Generate set of all words in the database
def get_set_of_words_across_the_database(word_avg_dict):
    set_of_all_words = set()
    for key, value in word_avg_dict.items():
        key_tuple = key[1:-1].split(",")
        component_id = key_tuple[0].strip(' ').strip('\'')
        gesture_id = key_tuple[1].strip(' ').strip('\'')
        sensor_id = key_tuple[2].strip(' ').strip('\'')
        word = (component_id, sensor_id, value[1])
        set_of_all_words.add(word)
    return set_of_all_words

# Generate TF vector
def generate_tf_vector_dictionary(word_avg_dict, word_position_dictionary):
    tf_vector_dictionary = {}
    for key, value in word_avg_dict.items():
        key_tuple = key[1:-1].split(",")
        component_id = key_tuple[0].strip(' ').strip('\'')
        gesture_id = key_tuple[1].strip(' ').strip('\'')
        sensor_id = key_tuple[2].strip(' ').strip('\'')
        word = (component_id, sensor_id, value[1])
        if gesture_id not in tf_vector_dictionary:
            tf_vector_dictionary[gesture_id] = {}
        if component_id not in tf_vector_dictionary[gesture_id]:
            tf_vector_dictionary[gesture_id][component_id] = {}
        if sensor_id not in tf_vector_dictionary[gesture_id][component_id]:
            tf_vector_dictionary[gesture_id][component_id][sensor_id] = [0] * len(word_position_dictionary.keys())
        tf_vector_dictionary[gesture_id][component_id][sensor_id][word_position_dictionary[word]] += 1

    for gesture_id in tf_vector_dictionary.keys():
        for component_id in tf_vector_dictionary[gesture_id].keys():
            for sensor_id in tf_vector_dictionary[gesture_id][component_id].keys():
                tf_vector = tf_vector_dictionary[gesture_id][component_id][sensor_id]
                sum = 0
                for x in tf_vector:
                    sum += x
                tf_vector[:] = [x / sum for x in tf_vector]
                tf_vector_dictionary[gesture_id][component_id][sensor_id] = tuple(tf_vector)
    return tf_vector_dictionary

# Generate IDF vector
def generate_idf_vector(word_avg_dict, word_position_dictionary):
    word_document_occurrence_dictionary = {}
    set_of_gestures = set()
    idf_vector = [0] * len(word_position_dictionary.keys())
    for key, value in word_avg_dict.items():
        key_tuple = key[1:-1].split(",")
        component_id = key_tuple[0].strip(' ').strip('\'')
        gesture_id = key_tuple[1].strip(' ').strip('\'')
        sensor_id = key_tuple[2].strip(' ').strip('\'')
        word = (component_id, sensor_id, value[1])

        if word not in word_document_occurrence_dictionary:
            word_document_occurrence_dictionary[word] = set()
        word_document_occurrence_dictionary[word].add(gesture_id)
        set_of_gestures.add(gesture_id)
    
    for word, gestures_in_which_word_occurs in word_document_occurrence_dictionary.items():
        idf_vector[word_position_dictionary[word]] = math.log(len(set_of_gestures)/len(gestures_in_which_word_occurs))
    
    idf_vector = tuple(idf_vector)
    return idf_vector

# Combine all vector representations into one dictionary
def combine_all_vectors(tf_vector_dictionary, idf_vector):
    idf_vector_tuple = idf_vector
    vectors = {}
    for gesture_id in tf_vector_dictionary.keys():
        for component_id in tf_vector_dictionary[gesture_id].keys():
            for sensor_id in tf_vector_dictionary[gesture_id][component_id].keys():
                tf_vector_tuple = tf_vector_dictionary[gesture_id][component_id][sensor_id]
                tfidf_vector_tuple = tuple([a*b for a,b in zip(tf_vector_tuple, idf_vector_tuple)])
                vector_list = (tf_vector_tuple, tfidf_vector_tuple)
                if gesture_id not in vectors:
                    vectors[gesture_id] = {}
                if component_id not in vectors[gesture_id]:
                    vectors[gesture_id][component_id] = {}
                if sensor_id not in vectors[gesture_id][component_id]:
                    vectors[gesture_id][component_id][sensor_id] = {}
                vectors[gesture_id][component_id][sensor_id] = vector_list
    return vectors

# Delete vectors.txt file if present
def delete_vector_text_file(directory):
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*_vectors_*.txt'):
            os.remove(os.path.join(root,filename))

# Write vectors.txt file
def write_vector_text_file(directory, vectors_dict):
    delete_vector_text_file(directory)
    for gesture_id in vectors_dict.keys():
        tf_vector_file_path = directory + "/tf_vectors_" + gesture_id + ".txt"
        tfidf_vector_file_path = directory + "/tfidf_vectors_" + gesture_id + ".txt"
        f_tf = open(tf_vector_file_path, "w")
        f_tfidf = open(tfidf_vector_file_path, "w")
        for component_id in vectors_dict[gesture_id].keys():
            f_tf.write("Component ID: " + str(component_id) + "->\n")
            f_tfidf.write("Component ID: " + str(component_id) + "->\n")
            for sensor_id in vectors_dict[gesture_id][component_id].keys():
                f_tf.write("\tSensor ID: " + str(sensor_id) + "->\n")
                f_tf.write("\t\t" + str(vectors_dict[gesture_id][component_id][sensor_id][0]) + "\n")
                f_tfidf.write("\tSensor ID: " + str(sensor_id) + "->\n")
                f_tfidf.write("\t\t" + str(vectors_dict[gesture_id][component_id][sensor_id][1]) + "\n")
        f_tf.close()
        f_tfidf.close()

# Serialize vector dictiionary
def serialize_vectors_dictionary(vectors_dictionary):
    with open("intermediate/vectors_dictionary.json", "w") as write_file:
        json.dump(vectors_dictionary, write_file)

def generate_vectors():
    print("Deserializing objects from previous tasks...")
    file_name = "intermediate/word_avg_dict.json"
    word_avg_dict = deserialize_word_avg_dict(file_name)
    data_parameters_file_name = "intermediate/data_parameters.json"
    data = deserialize_data_parameters(data_parameters_file_name)

    print("Generating TF and TF-IDF vectors...")
    set_of_all_words = get_set_of_words_across_the_database(word_avg_dict)
    # Convert set of all words to list
    list_of_all_words = list(set_of_all_words)
    word_position_dictionary = generate_word_position_dictionary(list_of_all_words)
    tf_vector_dictionary = generate_tf_vector_dictionary(word_avg_dict, word_position_dictionary)
    idf_vector = generate_idf_vector(word_avg_dict, word_position_dictionary)

    # Combine all the three representations of all gestures into one dictionary
    print("Writing vectors.txt...")
    vectors_dictionary = combine_all_vectors(tf_vector_dictionary, idf_vector)
    # Write vectors.txt
    write_vector_text_file(data['directory'], vectors_dictionary)

    print("Serializing objects needed for future tasks...")
    serialize_vectors_dictionary(vectors_dictionary)

    print("Task-2 complete!")

def main():
    # while(True):
    #     print("\n******************** Task-2 **********************")
    #     print(" Enter 1 to generate TF, TF-IDF and TF-IDF2 vectors.")
    #     print(" Enter 2 to exit")
    #     option = input("Enter option: ")
    #     if option == '1':
            task0a.create_output_directories()
            generate_vectors()
        # else:
        #     break

if __name__ == "__main__":
    main()