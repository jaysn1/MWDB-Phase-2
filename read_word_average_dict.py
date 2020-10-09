# given the location of word_average_dict, this function reads the file and converts it to our required format.
# -> gesture: [ssendor_id: [1,2,3,4,5,6,7] (time series of average word dict), Sendor_id: [1,2,3,4,5,6,7] (time series of average word dict)]
import json


def read_word_average_dict(word_average_dict_dir="intermediate/word_avg_dict.json"):
    with open(word_average_dict_dir) as f:
        word_average_dict = json.load(f)
    average_dict = {}

    for key in word_average_dict.keys():
        key_tuple = key[1:-1].split(",")
        component_id = key_tuple[0].strip(' ').strip('\'')
        gesture_id = key_tuple[1].strip(' ').strip('\'')
        sensor_id = int(key_tuple[2].strip(' ').strip('\''))
        time = int(key_tuple[3].strip(' ').strip('\''))
        average = word_average_dict[key][0]

        if gesture_id not in average_dict:
            average_dict[gesture_id] = {}
        if component_id not in average_dict[gesture_id]:
            average_dict[gesture_id][component_id] = {}
        if sensor_id not in average_dict[gesture_id][component_id]:
            average_dict[gesture_id][component_id][sensor_id] = []
        average_dict[gesture_id][component_id][sensor_id].append((average, time))
    
    for gesture_id in average_dict:
        for component_id in average_dict[gesture_id]:
            for sensor_id in average_dict[gesture_id][component_id]:
                average_dict[gesture_id][component_id][sensor_id] = [_[0] for _ in sorted(average_dict[gesture_id][component_id][sensor_id], key=lambda x: x[1])]
    return average_dict

if __name__=="__main__":
    print(read_word_average_dict().keys())
    print(read_word_average_dict()['1'].keys())
    print(read_word_average_dict()['1']['X'].keys())
    print(read_word_average_dict()['1']['X'][1])