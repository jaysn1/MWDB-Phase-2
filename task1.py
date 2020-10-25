from sklearn.decomposition import NMF, PCA, TruncatedSVD as SVD, LatentDirichletAllocation as LDA
from pandas import DataFrame
import numpy as np
import json
import sys
import os
import fnmatch

def read_vectors(vector_model, vectors_dir = "intermediate/vectors_dictionary.json"):
    with open(vectors_dir) as f:
        vectors = json.load(f)
    gestures = []
    gesture_ids = []
    for key in vectors.keys():
        gestures.append(vectors[key][vector_model-1])
        gesture_ids.append(key)
    gestures = np.array(gestures)    
    return gestures, gesture_ids

def read_words(word_dir = "intermediate/word_position_dictionary.json"):
    with open(word_dir) as f:
        words_position_dictionary = json.load(f)
    return words_position_dictionary

def calculate_pca(vector_model, k):
    """
    This function returns a dictionary of top K components from all gestures.

    params: vector_model: 1,2 suggesting TF, TF-IDF respectively
            k: tol-k latent semantics
    return: dictionary with key as gesture ID and transformed vector as value
    """
    gestures, gesture_ids = read_vectors(vector_model)

    gestures = np.array(gestures)
    pca = PCA(k)
    pca.fit(gestures)
    eigen_vectors = pca.components_
    transformed_data = pca.transform(gestures) 

    word_scores = []
    word_position_dictionary = read_words()
    words = sorted(word_position_dictionary.keys(), key=lambda x: word_position_dictionary[x])
    for eigen_vector in eigen_vectors:
        word_score = sorted(zip(abs(eigen_vector), words), key=lambda x: -x[0])
        word_scores.append(word_score)

    transformed_gestures_dict = {}
    for i in range(len(gesture_ids)):
        transformed_gestures_dict[gesture_ids[i]] = list(transformed_data[i])

    return transformed_gestures_dict, word_scores

def calculate_svd(vector_model, k):
    """
    This function returns a dictionary of top K components from all gestures.

    params: vector_model: 1,2 suggesting TF, TF-IDF respectively
            k: tol-k latent semantics
    return: dictionary with key as gesture ID and transformed vector as value
    
    # http://infolab.stanford.edu/~ullman/mmds/ch11.pdf
    """
    gestures, gesture_ids = read_vectors(vector_model)

    gestures = np.array(gestures)
    svd = SVD(n_components=k, algorithm = "randomized", n_iter=7, random_state=42)
    # algorithm: To computes a truncated randomized SVD
    # n_iter: Number of iterations for randomized SVD solver
    # random_states: Used during randomized svd. To produce reproducible results
    try:
        svd.fit(gestures)
    except ValueError as e:
        raise ValueError(e)
    eigen_vectors = svd.components_
    transformed_data = svd.transform(gestures)
    
    word_scores = []
    word_position_dictionary = read_words()
    words = sorted(word_position_dictionary.keys(), key=lambda x: word_position_dictionary[x])
    for eigen_vector in eigen_vectors:
        word_score = sorted(zip(abs(eigen_vector), words), key=lambda x: -x[0])
        word_scores.append(word_score)

    transformed_gestures_dict = {}
    for i in range(len(gesture_ids)):
        transformed_gestures_dict[gesture_ids[i]] = list(transformed_data[i])

    return transformed_gestures_dict, word_scores

def calculate_nmf(vector_model, k):
    """
    This function returns a dictionary of top K components from all gestures.

    params: vector_model: 1,2 suggesting TF, TF-IDF respectively
            k: tol-k latent semantics
    return: dictionary with key as gesture ID and transformed vector as value
    """
    gestures, gesture_ids = read_vectors(vector_model)

    gestures = np.array(gestures)
    nmf = NMF(n_components=k, init='random', random_state=0)
    # init: non-negative random matrices initialization
    # random_state=0 : for reproducible results
    nmf.fit(gestures)
    eigen_vectors = nmf.components_
    transformed_data = nmf.transform(gestures) 

    word_scores = []
    word_position_dictionary = read_words()
    words = sorted(word_position_dictionary.keys(), key=lambda x: word_position_dictionary[x])
    for eigen_vector in eigen_vectors:
        word_score = sorted(zip(abs(eigen_vector), words), key=lambda x: -x[0])
        word_scores.append(word_score)

    transformed_gestures_dict = {}
    for i in range(len(gesture_ids)):
        transformed_gestures_dict[gesture_ids[i]] = list(transformed_data[i])

    return transformed_gestures_dict, word_scores

def calculate_lda(vector_model, k):
    """
    This function returns a dictionary of top K components from all gestures.

    params: vector_model: 1,2 suggesting TF, TF-IDF respectively
            k: tol-k latent semantics
    return(2): dictionary with key as gesture ID AND transformed vector as value
    """
    gestures, gesture_ids = read_vectors(vector_model)

    gestures = np.array(gestures)
    lda = LDA(n_components = k)
    lda.fit(gestures)
    eigen_vectors = lda.components_
    transformed_data = lda.transform(gestures) 

    word_scores = []
    word_position_dictionary = read_words()
    words = sorted(word_position_dictionary.keys(), key=lambda x: word_position_dictionary[x])
    for eigen_vector in eigen_vectors:
        word_score = sorted(zip(abs(eigen_vector), words), key=lambda x: -x[0])
        word_scores.append(word_score)

    transformed_gestures_dict = {}
    for i in range(len(gesture_ids)):
        transformed_gestures_dict[gesture_ids[i]] = list(transformed_data[i])

    return transformed_gestures_dict, word_scores

# Deserialize data parameters
def deserialize_data_parameters(file_name):
    with open(file_name, "r") as read_file:
        data = json.load(read_file)
    return data

# Delete word_score.txt file if present
def delete_word_score_text_file(directory):
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*word_score*.txt'):
            os.remove(os.path.join(root,filename))

# Write word_score.txt file
def write_word_score_text_file(directory, file_name, word_score_matrix):
    delete_word_score_text_file(directory)
    with open(os.path.join(directory, file_name), "w") as f:
        for row in range(len(word_score_matrix)):
            for col in range(len(word_score_matrix[row])):
                score, word = word_score_matrix[row][col]
                f.write("{} : {} \t".format(word, score))
            f.write("\n")

def main():
    vectors_dir = "intermediate/vectors_dictionary.json"
    transformed_data_dir = "intermediate/transformed_data.json"
    word_score_file_name = "word_score.txt"
    data_parameters_file_name = "intermediate/data_parameters.json"


    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Phase 2 -Task 1                                                        │
    │                                                                         │
    │    1 - PCA                                                              │
    │    2 - SVD                                                              │
    │    3 - NMF                                                              │
    │    4 - LDA                                                              │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘""")
    user_option = int(input("\nEnter method to use to find latent latent semantics: "))
    print("\n1. TF")
    print("2. TF-IDF")
    vector_model = int(input("Enter a vector model: "))

    k = int(input("Enter k for top-k latent features: "))
    
    try:
        if user_option == 1:
            transformed_gestures_dict, word_score_matrix = calculate_pca(vector_model, k)
        elif user_option == 2:
            transformed_gestures_dict, word_score_matrix = calculate_svd(vector_model, k)
        elif user_option == 3:
            transformed_gestures_dict, word_score_matrix = calculate_nmf(vector_model, k)
        elif user_option == 4:
            transformed_gestures_dict, word_score_matrix = calculate_lda(vector_model, k)
        else:
            print("Incorrect value!")
            sys.exit(0)            
    except ValueError as e:
        print(e)
        sys.exit()

    data = deserialize_data_parameters(data_parameters_file_name)
    write_word_score_text_file(data['directory'], word_score_file_name, word_score_matrix)
    with open(transformed_data_dir, "w") as f:
        json.dump(transformed_gestures_dict, f)
    
    print("\nResults for this task are stored in: ", word_score_file_name)


if __name__ == '__main__':
    main()
