import json

# Deserialize gesture latent topic
def deserialize_gesture_latent_topic(file_name):
    with open(file_name, "r") as read_file:
        gesture_latent_topic = json.load(read_file)
    return gesture_latent_topic

def main():
    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Phase 2 - Task 4                                                       │
    │                                                                         │
    │                                                                         │
    │    1) SVD                                                               │
    │    2) NMF                                                               │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘""")
    user_option = int(input("\nEnter method to used to partition gestures into p groups: "))
    gesture_gesture_score_json_file_path = "intermediate/gesture_gesture_score.json"
    gesture_latent_topic = deserialize_gesture_latent_topic(gesture_gesture_score_json_file_path)
    set_of_all_gestures = set()
    gesture_percentage_dict = {}
    # calculate percentage contribution of each gesture in latent topic
    for i in range(len(gesture_latent_topic)):
        latent_topic = gesture_latent_topic[i]
        sum_of_all_scores = 0
        for gesture_score in latent_topic:
            sum_of_all_scores += gesture_score[0]
        for gesture_score in latent_topic:
            if i not in gesture_percentage_dict:
                gesture_percentage_dict[i] = {}
            set_of_all_gestures.add(gesture_score[1])
            gesture_percentage_dict[i][gesture_score[1]] = gesture_score[0]/sum_of_all_scores

    list_of_all_gestures = list(set_of_all_gestures)
    clusters = {}
    # assign gesture to the latent topic in which it has largest % contribution
    for gesture_id in list_of_all_gestures:
        largest_contribution = 0
        cluster_index = 0
        for i in range(len(gesture_latent_topic)):
            if gesture_percentage_dict[i][gesture_id] > largest_contribution:
                largest_contribution = gesture_percentage_dict[i][gesture_id]
                cluster_index = i
        if cluster_index not in clusters:
            clusters[cluster_index] = []
        clusters[cluster_index].append(gesture_id)

    # output cluster membership of all gestures
    # count = 0
    for cluster_number, gestures in clusters.items():
        print("Under cluster {}:".format(cluster_number + 1))
        print("\t {}".format(gestures))
        # count += len(gestures)

if __name__ == '__main__':
    main()