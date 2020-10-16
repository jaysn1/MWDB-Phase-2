from sklearn.decomposition import NMF, PCA, TruncatedSVD as SVD, LatentDirichletAllocation as LDA
from pandas import DataFrame
from functools import wraps
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import sys


# copied change this
def latent_semantic_to_string(table, save=True):
    table = table.T
    to_return = ""

    if save:
        table.to_csv('latent_semantics.csv')
    for i, col_id in enumerate(table):
        column = table[col_id]
        column = column.sort_values(ascending=False)
        to_return += '-' * 50
        to_return += "\nLATENT SEMANTIC " + str(i)
        to_return += '-' * 50
        to_return += column.__repr__()
    return to_return


def deserialize_vectors_dict(file_name):
    global vectors_dict
    with open(file_name, "r") as read_file:
        vectors_dict = json.load(read_file)


def pca(table, k):
    model = PCA(n_components=k)
    out = model.fit_transform(table)
    return out, model.explained_variance_ratio_


def svd(table, k):
    model = SVD(n_components=k)
    out = model.fit_transform(table)
    return out, model.components_


def nmf(table, k):
    model = NMF(n_components=k)
    out = model.fit_transform(table)
    return out, model.components_


def lda(table, k):
    model = LDA(n_components=k)
    out = model.fit_transform(table)
    return out, model.components_


def select_method(table):
    global method_name, k
    print("Select method:")
    print("1. PCA")
    print("2. SVD")
    print("3. NMF")
    print("4. LDA")
    try:
        option = input("Enter option: ")
        method = int(option)
        k = int(input("Enter k: "))
        if method == 1:
            method_name = "PCA"
            return pca(table, k)
        elif method == 2:
            method_name = "SVD"
            return svd(table, k)
        elif method == 3:
            method_name = "NMF"
            return nmf(table, k)
        elif method == 4:
            method_name = "LDA"
            return lda(table, k)
        else:
            exit(400)
    except ValueError:
        sys.exit(500)

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
    eigen_values = pca.explained_variance_
    transformed_data = pca.transform(gestures)
    
    print(transformed_data)

    word_scores = []
    word_position_dictionary = read_words()
    words = sorted(word_position_dictionary.keys(), key=lambda x: word_position_dictionary[x])
    for eigen_vector in eigen_vectors:
        word_score = sorted(zip(eigen_vector, words), key=lambda x: -x[0])
        word_scores.append(word_score)
    
    # transformed_data = 
    transformed_gestures_dict = {}
    for i in range(len(gesture_ids)):
        transformed_gestures_dict[gesture_ids[i]] = list(transformed_data[i])

    return transformed_gestures_dict, word_scores



def main():
    vectors_dir="intermediate/vectors_dictionary.json"
    
    
    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Phase 2 -Task 1                                                        │
    │                                                                         │
    │    1 - PCA                                                              │
    │    2 - CVD                                                              │
    │    3 - NMF                                                              │
    │    4 - LDA                                                              │
    └─────────────────────────────────────────────────────────────────────────┘""")
    user_option = int(input("\nEnter method to use to find latent latent semantics: "))
    print("\n", "1. TF")
    print("2. TF-IDF")
    vector_model = int(input("Enter a vector model: "))

    k = int(input("Enter k for top-k latent features: "))
    

    if user_option==1:
        transformed_gestures_dict, word_score_matrix = calculate_pca(vector_model, k)
    
        
    with open("intermediate/transformed_data.json", "w") as write_file:
        json.dump(transformed_gestures_dict, write_file)
            
 


if __name__ == '__main__':
    main()
