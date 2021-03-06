import os
import json
import math
import fnmatch
# import task0a

# Deserialize word average dictionary
def deserialize_word_avg_dict(file_name):
    with open(file_name, "r") as read_file:
        word_avg_dict = json.load(read_file)
    return word_avg_dict

# Deserialize word vector dictionary
def deserialize_word_vector_dict(file_name):
    with open(file_name, "r") as read_file:
        word_vector_dict = json.load(read_file)
    for key, value in word_vector_dict.items():
        word_vector_dict[key] = tuple(value)
    return word_vector_dict

# Deserialize data parameters
def deserialize_data_parameters(file_name):
    with open(file_name, "r") as read_file:
        data = json.load(read_file)
    return data

# Create a dictionary containing the position of each word
# in the vector representations
def generate_word_position_dictionary(list_of_all_words):
    word_position_dictionary = {}
    component_position_dictionary = {}
    sensor_position_dictionary = {}
    k=0
    for word in list_of_all_words:
        component_id = word[0]
        sensor_id = word[1]
        word_position_dictionary[str(word)] = k
        if component_id not in component_position_dictionary:
            component_position_dictionary[component_id] = [k]
        else:
            component_position_dictionary[component_id].append(k)
        if sensor_id not in sensor_position_dictionary:
            sensor_position_dictionary[sensor_id] = [k]
        else:
            sensor_position_dictionary[sensor_id].append(k)
        k += 1
    return word_position_dictionary, component_position_dictionary, sensor_position_dictionary

# Generate set of all words in the database
def get_set_of_words_across_the_database(word_vector_dict):
    set_of_all_words = set()
    for key, value in word_vector_dict.items():
        key_tuple = key[1:-1].split(",")
        component_id = key_tuple[0].strip(' ').strip('\'')
        gesture_id = key_tuple[1].strip(' ').strip('\'')
        sensor_id = key_tuple[2].strip(' ').strip('\'')
        word = (component_id, sensor_id, value)
        set_of_all_words.add(word)
    return set_of_all_words

# Generate TF vector
def generate_tf_vector_dictionary(word_vector_dict, word_position_dictionary):
    tf_vector_dictionary = {}
    for key, value in word_vector_dict.items():
        key_tuple = key[1:-1].split(",")
        component_id = key_tuple[0].strip(' ').strip('\'')
        gesture_id = key_tuple[1].strip(' ').strip('\'')
        sensor_id = key_tuple[2].strip(' ').strip('\'')
        word = (component_id, sensor_id, value)
        if gesture_id not in tf_vector_dictionary:
            tf_vector_dictionary[gesture_id] = [0] * len(word_position_dictionary.keys())
        tf_vector_dictionary[gesture_id][word_position_dictionary[str(word)]] += 1

    for gesture_id in tf_vector_dictionary.keys():
        tf_vector = tf_vector_dictionary[gesture_id]
        tf_vector[:] = [x / len(tf_vector) for x in tf_vector]
        tf_vector_dictionary[gesture_id] = tuple(tf_vector)
    return tf_vector_dictionary

# Generate IDF vector
def generate_idf_vector(word_vector_dict, word_position_dictionary):
    word_document_occurrence_dictionary = {}
    set_of_gestures = set()
    idf_vector = [0] * len(word_position_dictionary.keys())
    for key, value in word_vector_dict.items():
        key_tuple = key[1:-1].split(",")
        component_id = key_tuple[0].strip(' ').strip('\'')
        gesture_id = key_tuple[1].strip(' ').strip('\'')
        sensor_id = key_tuple[2].strip(' ').strip('\'')
        word = (component_id, sensor_id, value)

        if word not in word_document_occurrence_dictionary:
            word_document_occurrence_dictionary[word] = set()
        word_document_occurrence_dictionary[word].add(gesture_id)
        set_of_gestures.add(gesture_id)
    
    for word, gestures_in_which_word_occurs in word_document_occurrence_dictionary.items():
        idf_vector[word_position_dictionary[str(word)]] = math.log(len(set_of_gestures)/len(gestures_in_which_word_occurs))
    
    idf_vector = tuple(idf_vector)
    return idf_vector

# Combine all vector representations into one dictionary
def combine_all_vectors(tf_vector_dictionary, idf_vector):
    idf_vector_tuple = idf_vector
    vectors = {}
    for gesture_id in tf_vector_dictionary.keys():
        tf_vector_tuple = tf_vector_dictionary[gesture_id]
        tfidf_vector_tuple = tuple([a*b for a,b in zip(tf_vector_tuple, idf_vector_tuple)])
        vector_tuple = (tf_vector_tuple, tfidf_vector_tuple)
        vectors[gesture_id] = vector_tuple
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
        f_tf.write(str(vectors_dict[gesture_id][0]) + "\n")
        f_tfidf.write(str(vectors_dict[gesture_id][1]) + "\n")
        f_tf.close()
        f_tfidf.close()

# Serialize vector dictiionary
def serialize_vectors_dictionary(vectors_dictionary):
    with open("intermediate/vectors_dictionary.json", "w") as write_file:
        json.dump(vectors_dictionary, write_file)

def serialize_word_position_dictionary(word_position_dictionary):
    with open("intermediate/word_position_dictionary.json", "w") as write_file:
        json.dump(word_position_dictionary, write_file)

def serialize_component_position_dictionary(component_position_dictionary):
    with open("intermediate/component_position_dictionary.json", "w") as write_file:
        json.dump(component_position_dictionary, write_file)

def serialize_sensor_position_dictionary(sensor_position_dictionary):
    with open("intermediate/sensor_position_dictionary.json", "w") as write_file:
        json.dump(sensor_position_dictionary, write_file)

def generate_vectors():
    print("Deserializing objects from previous tasks...")
    # file_name = "intermediate/word_avg_dict.json"
    # word_avg_dict = deserialize_word_avg_dict(file_name)
    file_name = "intermediate/word_vector_dict.json"
    word_vector_dict = deserialize_word_vector_dict(file_name)
    data_parameters_file_name = "intermediate/data_parameters.json"
    data = deserialize_data_parameters(data_parameters_file_name)

    print("Generating TF and TF-IDF vectors...")
    set_of_all_words = get_set_of_words_across_the_database(word_vector_dict)
    # Convert set of all words to list
    list_of_all_words = list(set_of_all_words)
    word_position_dictionary, component_position_dictionary, sensor_position_dictionary = generate_word_position_dictionary(list_of_all_words)
    tf_vector_dictionary = generate_tf_vector_dictionary(word_vector_dict, word_position_dictionary)
    idf_vector = generate_idf_vector(word_vector_dict, word_position_dictionary)

    # Combine all the three representations of all gestures into one dictionary
    print("Writing vectors.txt...")
    vectors_dictionary = combine_all_vectors(tf_vector_dictionary, idf_vector)
    # Write vectors.txt
    write_vector_text_file(data['directory'], vectors_dictionary)

    print("Serializing objects needed for future tasks...")
    serialize_vectors_dictionary(vectors_dictionary)
    serialize_word_position_dictionary(word_position_dictionary)
    serialize_component_position_dictionary(component_position_dictionary)
    serialize_sensor_position_dictionary(sensor_position_dictionary)

def main():
    # task0a.create_output_directories()
    generate_vectors()

if __name__ == "__main__":
    main()