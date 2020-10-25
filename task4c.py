import json
import pandas as pd
from k_means import K_Means

# Deserialize word average dictionary
def deserialize_gesture_gesture_similarity_dict(file_name):
    with open(file_name, "r") as read_file:
        gesture_gesture_similarity = json.load(read_file)
    return gesture_gesture_similarity

def convert_similarity_dict_to_df(gesture_gesture_similarity_dict):
    # get all gesture ids
    columns = list(gesture_gesture_similarity_dict.keys())
    # index
    columns.insert(0, "gesture_id")

    # create a position dictionary for each gestureid
    gesture_position_dict = dict()
    for i in range(1, len(columns)):
        gesture_position_dict[columns[i]] = i
    
    # parent data frame
    gesture_gesture_similarity_df = pd.DataFrame(columns=columns)

    # convert the gesture gesture similary dictionary into a dataframe
    for gesture_id, similarity_dict in gesture_gesture_similarity_dict.items():
        similarity_list = [None]*len(columns)
        for key, value in similarity_dict.items():
            similarity_list[gesture_position_dict[key]] = value
        similarity_list[0] = gesture_id
        temp_df = pd.DataFrame([similarity_list], columns=columns)
        gesture_gesture_similarity_df = pd.concat([gesture_gesture_similarity_df, temp_df])
    
    gesture_gesture_similarity_df = gesture_gesture_similarity_df.set_index('gesture_id')
    return gesture_gesture_similarity_df

def main():
    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Phase 2 -Task 4c                                                       │
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
    
    user_option = int(input("\nEnter which type of gesture gesture similarity to use: "))
    p = int(input("\nEnter the number of clusters: "))
    print("Performing k-means clustering on the gesture-gesture similarity matrix...")

    gesture_gesture_similarity_file_path = "intermediate/gesture_gesture_similarity_dictionary.json"
    word_position_dictionary_file_path = "intermediate/word_position_dictionary.json"

    # deserialize gesture similarity dictionary
    gesture_gesture_similarity_dict = deserialize_gesture_gesture_similarity_dict(gesture_gesture_similarity_file_path)
    # convert gesture similarity dictionary into a dataframe for k-means
    gesture_gesture_similarity_df = convert_similarity_dict_to_df(gesture_gesture_similarity_dict)

    # perform k-means
    k_means = K_Means(p, max_iter=500)
    clusters = k_means.fit(gesture_gesture_similarity_df)

    # output cluster membership of all gestures
    # count = 0
    for cluster_number, gestures in clusters.items():
        print("Under cluster {}:".format(cluster_number + 1))
        print("\t {}".format(gestures))
        # count += len(gestures)

if __name__ == "__main__":
    main()