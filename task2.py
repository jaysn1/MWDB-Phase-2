# task2
# Implement a program which, given a gesture file and a vector model, finds and ranks the 10 most similar gestures
# in the gesture database, relying on the following:

# – user option #1: dot product of gesture vectors,
# – user options #2, 3, 4, 5: top-k latent sensor semantics (PCA, SVD, NMF, LDA),
# – user option #6: edit distance on symbolic quantized window descriptors (propose an edit cost function among
# symbols),
# – user option #7: DTW distance on average quantized amplitudes

# Results are presented in the form of hgesture, scorei pairs sorted in non-increasing order of similarity scores.
from dot_product_similarity import dot_product_similarity
from read_vectors import read_vectors
import json

def main():
    k = 10
    vectors_dir="intermediate/vectors_dictionary.json"

    print("************* Task-2 **************")
    with open(vectors_dir) as f:
        vectors = json.load(f)

    gesture_id = input("Enter gesture: ")
    assert gesture_id in vectors

    vector_model = int(input("Enter Gesture vector to use? (1: TF, 2: TF-IDF): "))
    user_option = int(input(" #1: dot product\n #2:PCA \n #3: SVD \n \
#4:NMF \n #5:LDA \n #6: edit distance \n #7: DTW distance \nEnter method to use to find similarity: "))
    if user_option == 1:
        vector = vectors[gesture_id][vector_model]
        similarity = {}
        for gesture in vectors:
            similarity[gesture] = dot_product_similarity(vector,vectors[gesture][vector_model])
        top_k_similar = [_[0] for _ in sorted(similarity.items(), key=lambda x: x[1],reverse=True)[:k]]
        print(top_k_similar)

main()