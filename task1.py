from sklearn.decomposition import NMF, PCA, TruncatedSVD as SVD, LatentDirichletAllocation as LDA
from pandas import DataFrame
from functools import wraps
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json

vectors_dict = {}


def deserialize_vectors_dict(file_name):
    global vectors_dict
    with open(file_name, "r") as read_file:
        vectors_dict = json.load(read_file)


def pca(table, k):
    matrix = PCA(n_components=k)
    out = matrix.fit_transform(table)
    return out, matrix.components_


def svd(table, k):
    matrix = SVD(n_components=k)
    out = matrix.fit_transform(table)
    return out, matrix.components_


def nmf(table, k):
    matrix = NMF(n_components=k)
    out = matrix.fit_transform(table)
    return out, matrix.components_


def lda(table, k):
    matrix = LDA(n_components=k)
    out = matrix.fit_transform(table)
    return out, matrix.components_


def select_method(table):
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
            return pca(table, k)
        elif method == 2:
            return svd(table, k)
        elif method == 3:
            return nmf(table, k)
        elif method == 4:
            return lda(table, k)
        else:
            exit(400)
    except ValueError:
        exit(500)


def main():
    global vectors_dict
    gesture_files = input("Enter desired gesture files, separated by comma (ex.: 1,2,3,...,60 etc.):\n")
    gesture_files = gesture_files.split(',')
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
            print(gesture_files)
            for gesture_number in gesture_files:
                table.append(vectors_dict[gesture_number][model - 1])
            return select_method(table)
        else:
            exit(400)
    except ValueError:
        exit(500)


if __name__ == '__main__':
    main()
