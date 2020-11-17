import pandas as pd
from random import randint
import numpy as np

class LSH:
    def generate_shingle_matrix(self, k, vectors):
        df = pd.DataFrame(columns=[x for x in range(0, len(vectors))])
        print(df.head())
        count = 0
        for vector in vectors:
            for i in range(0, len(vector)-k+1):
                shingle = str(tuple(vector[i:i+k]))
                if(shingle in df.index) == False:
                    df.loc[shingle] = [0]*len(vectors)
                df.at[shingle, count] = 1
            count += 1
        return df

    def generate_hash_functions(self, num_rows, num_hash_functions):
        hash_functions = []
        for i in range(0, num_hash_functions):
            def hash_function(row):
                return (randint(1,5*num_rows)*row + randint(1,5*num_rows))%num_rows
            hash_functions.append(hash_function)
        return hash_functions
    
    def generate_signature_matrix(self, shingle_matrix):
        num_rows, num_columns = shingle_matrix.shape[0], shingle_matrix.shape[1]
        num_hash_functions = 300
        df = pd.DataFrame(index=[x for x in range(0, num_hash_functions)], columns=[x for x in range(0, num_columns)])
        hash_functions = self.generate_hash_functions(num_rows, num_hash_functions)
        for i in range(num_rows):
            for j in range(num_columns):
                if shingle_matrix.iat[i, j] == 1:
                    for k in range(num_hash_functions):
                        if pd.isna(df.iat[k, j]):
                            df.iat[k, j] = hash_functions[k](i)
                        else:
                            df.iat[k, j] = min(df.iat[k, j], hash_functions[k](i))
        return df

    def generate_hash_buckets(self, signature_matrix, num_rows_in_band):
        num_rows = signature_matrix.shape[0]
        num_hash_buckets = num_rows//num_rows_in_band
        hash_buckets = [{} for i in range(num_hash_buckets)]
        for i in range(0, num_rows-num_rows_in_band+1, num_rows_in_band):
            band_matrix = signature_matrix.loc[i:i+num_rows_in_band, :]
            for column in band_matrix.columns:
                hash_of_band = hash(tuple(band_matrix[column].values))
                hash_bucket = hash_buckets[i//num_rows_in_band]
                if hash_of_band not in hash_bucket:
                    hash_bucket[hash_of_band] = [column]
                else:
                    hash_bucket[hash_of_band].append(column)
        similar_docs = []
        for hash_bucket in hash_buckets:
                for k,v in hash_bucket.items():
                    if(len(v)>1):
                        similar_docs.append(hash_bucket.values())
        return hash_buckets, similar_docs

def main():
    lsh = LSH()
    str0 = "Construct an optimal solution from computed values."
    str1 = "It is easiest to calculate the cosinus of the angle between the vectors instead of the angle by the formula"
    str2 = "The vector space model (or term vector model) is an algebraic model for representing text documents"
    str3 = "is an algebraic model used to represent text documents, as well as any objects in general"
    str4 = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
    str5 = "The vector space model (or term vector model) is an algebraic model for representing text documents"
    list_of_str = [str0, str1, str2, str3, str4, str5]
    vectors = []
    for s in list_of_str:
        vectors.append([v for i,v in enumerate(s)])
    print(vectors)
    shingle_matrix = lsh.generate_shingle_matrix(4, vectors)
    print(shingle_matrix)
    signature_matrix = lsh.generate_signature_matrix(shingle_matrix)
    print(signature_matrix)
    hash_buckets, similar_docs = lsh.generate_hash_buckets(signature_matrix, 2)
    print(similar_docs)


if __name__=="__main__":
    main()

