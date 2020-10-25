import json
import pandas as pd
from spectral_clustering import SpectralClustering

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

def main(user_option):
    mapping = {1:'DOT', 2: 'PCA', 3: 'SVD', 4: 'NMF', 5: 'LDA', 6: 'ED', 7: 'DTW'}
    gesture_gesture_similarity_file_path = "intermediate/{}_NMF_gesture_gesture_similarity_dictionary.json".format(mapping[user_option])
    data_parameters_dir = "intermediate/data_parameters.json"

    # deserialize gesture similarity dictionary
    try:
        gesture_gesture_similarity_dict = deserialize_gesture_gesture_similarity_dict(gesture_gesture_similarity_file_path)
    except FileNotFoundError as e:
        raise ValueError("Run Task 3 with proper input.")
    # convert gesture similarity dictionary into a dataframe for k-means
    gesture_gesture_similarity_df = convert_similarity_dict_to_df(gesture_gesture_similarity_dict)

    # reading value of p from data_parameters.json
    with open(data_parameters_dir) as f:
        data_parameters = json.load(f)
    if 'p' not in data_parameters:
        raise ValueError("Run task 3.")
    p = data_parameters['p']
    print("Using p={} from task 3.".format(p))

    # perform spectral clustering
    spectral_clustering = SpectralClustering(p, max_iter=500)
    clusters = spectral_clustering.fit(gesture_gesture_similarity_df)

    # output cluster membership of all gestures
    # count = 0
    for cluster_number, gestures in clusters.items():
        print("Under cluster {}:".format(cluster_number + 1))
        print("\t {}".format(gestures))
        # count += len(gestures)

if __name__ == "__main__":
    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Phase 2 -Task 4d                                                       │
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
    
    print("Performing spectral clustering on the gesture-gesture similarity matrix...")
    
    
    main(user_option)