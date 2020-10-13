# given the location of sensor_average_dict, this function reads the file and converts it to our required format.
# -> gesture: [ssendor_id: [1,2,3,4,5,6,7] (time series of average word dict), Sendor_id: average, std (time series of average word dict)]
import json


def read_sensor_average_std_dict(sensor_average_dict_dir="intermediate/sensor_avg_std_dict.json"):
    with open(sensor_average_dict_dir) as f:
        sensor_average_dict = json.load(f)
    average_dict = {}

    for key in sensor_average_dict.keys():
        key_tuple = key[1:-1].split(",")
        gesture_id = key_tuple[0].strip(' ').strip('\'')
        component_id = key_tuple[1].strip(' ').strip('\'')
        sensor_id = int(key_tuple[2].strip(' ').strip('\''))
        average = sensor_average_dict[key][0]
        std = sensor_average_dict[key][1]

        if gesture_id not in average_dict:
            average_dict[gesture_id] = {}
        if component_id not in average_dict[gesture_id]:
            average_dict[gesture_id][component_id] = {}
        if sensor_id not in average_dict[gesture_id][component_id]:
            average_dict[gesture_id][component_id][sensor_id] = []
        average_dict[gesture_id][component_id][sensor_id] = (average, std)
    
    return average_dict

if __name__=="__main__":
    d = read_sensor_average_std_dict()
    print(d.keys())
    print(d['1'].keys())
    print(d['1']['X'].keys())
    print(d['1']['X']['1'])