from sklearn.decomposition import NMF, PCA, TruncatedSVD as SVD, LatentDirichletAllocation as LDA
from pandas import DataFrame
from functools import wraps
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import sys

vectors_dict = {}
k = 0
method_name = "NaN"

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


def main():
    global vectors_dict, method_name, k
    print("Select a vector model:")
    print("1. TF")
    print("2. TF-IDF")
    try:
        option = input("Enter option: ")
        model = int(option)
        vector_file_name = "intermediate/vectors_dictionary.json"
        deserialize_vectors_dict(vector_file_name)
        if model == 1 or model == 2:
            table = []
            gesture_ids = []
            for key in sorted(vectors_dict.keys()):
                table.append(vectors_dict[key][model - 1])
                gesture_ids.append(key)
                
            table = np.array(table)
            
            transformed_gestures, latent_semantics = select_method(table)
            transformed_gestures = DataFrame(transformed_gestures)
            transformed_gestures_dict = {}
            
            for i in range(len(gesture_ids)):
                transformed_gestures_dict[gesture_ids[i]] = list(transformed_gestures.iloc[i])
            with open("intermediate/transformed_data.json", "w") as write_file:
                json.dump(transformed_gestures_dict, write_file)
                
        else:
            exit(400)
    except ValueError:
        exit(500)


if __name__ == '__main__':
    main()
