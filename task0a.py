import pandas as pd
import numpy as np
import os
import json
from scipy.integrate import quad
import scipy.stats
import numpy as np
import statistics as st
import fnmatch

# Read and load parameters from user
def load_params():
    data = {}
    # data['directory'] = input("Enter directory: ")
    # data['window_length'] = int(input("Enter window length: "))
    # data['shift_length'] = int(input("Enter shift length: "))
    # data['resolution'] = int(input("Enter resolution: "))
    data['directory'] = 'data'
    data['window_length'] = 3
    data['shift_length'] = 2
    data['resolution'] = 3
    return data

# Load gesture files from the directory
def load_all_the_component_gestures(directory):
    complete_df = pd.DataFrame()
    for link in os.listdir(directory):
        if os.path.isdir(os.path.join(directory,link)):
            component_id = link
            for filename in os.listdir(os.path.join(directory,link)):
                if filename.endswith(".csv"):
                    df = pd.read_csv(os.path.join(directory, link, filename), header=None)
                    sensor_id_series = list(range(1, len(df)+1))
                    gesture_id_series = [filename[:-4]] * len(df)
                    component_id_series = [component_id] * len(df)
                    df['sensor_id'] = sensor_id_series
                    df['gesture_id'] = gesture_id_series
                    df['component_id'] = component_id_series
                    complete_df = pd.concat([complete_df, df])
    return complete_df

# Create output directories if it does not exist
def create_output_directories():
    outdir = './intermediate'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

# Evaluate normal distribution
def normal_distribution_function(x, mean=0, std=0.25):
    value = scipy.stats.norm.pdf(x, mean, std)
    return value

# Get Gaussian band intervals
def get_intervals(r):
    # Normal Distribution
    total_intervals = 2*r
    x_min = -1
    x_max = 1

    interval_spacing = 2/float(2*r)
    x = np.linspace(x_min, x_max, 100)
    
    intervals = []
    count = 0
    start = -1
    while count < 2*r:
        res, err = quad(normal_distribution_function, -1, start)
        intervals.append(res)
        count += 1
        start += interval_spacing
    
    result = {}
    count = 1
    intervals.append(1)
    for i in range(0, len(intervals)):
        intervals[i] = -1 + 2*intervals[i]
    for i in range(0, len(intervals)-1):
        if i == len(intervals)-1:
            result[count] = [intervals[i], 1]
            continue
        result[count] = [intervals[i], intervals[i+1]-0.000000000000000001]
        count += 1
    return result

# Normalize each row
def normalize(row):
    columns_to_drop = ['sensor_id', 'gesture_id', 'component_id']
    row_with_only_sensor_values = row.drop(labels=columns_to_drop)
    max_value = max(row_with_only_sensor_values)
    min_value = min(row_with_only_sensor_values)
    list_without_na_values = []
    for idx in row.index:
        if idx in columns_to_drop or pd.isnull(row[idx]):
            continue
        list_without_na_values.append(row[idx])
        if max_value == min_value:
            row[idx] = 0
        else:
            row[idx] = (row[idx] - min_value)/(max_value - min_value)
            row[idx] = row[idx]*2 + -1

    row['avg_amplitude'] = st.mean(list_without_na_values)
    row['std_deviation'] = st.pstdev(list_without_na_values)
    return row

# Quantize each row
def quantize(row, interval_dict):
    columns_to_drop = ['sensor_id', 'gesture_id', 'component_id', 'avg_amplitude', 'std_deviation']
    for idx in row.index:
        if idx in columns_to_drop or pd.isnull(row[idx]):
            continue
        for key, value in interval_dict.items():
            if row[idx] >= value[0] and row[idx] <= value[1]:
                row[idx] = int(key)
                break
    return row

# Generate word vectors by using the sliding window technique
def generate_word_vectors(row, word_vector_dict, window_length, shift_length):
    columns_to_drop = ['sensor_id', 'gesture_id', 'component_id', 'avg_amplitude', 'std_deviation']
    sensor_id = row.sensor_id
    gesture_id = row.gesture_id
    component_id = row.component_id

    row = row.drop(labels=columns_to_drop)
    i=0
    while i < (len(row.index)-window_length):
        if pd.isnull(row[i]):
            break

        temp_key = str((component_id, gesture_id, sensor_id, i))
        k = i
        temp_list = []

        while k < (i + window_length):
            if pd.isnull(row[k]):
                break
            temp_list.append(int(row[k]))
            k += 1

        if len(temp_list) < window_length:
            break;
        # final symbol would look like (sensor_id, window_element_1, window_element_2, ...)
        temp_list.insert(0, sensor_id)
        word_vector_dict[temp_key] = tuple(temp_list)
        i += shift_length

    return row

# Generate word vectors by using the sliding window technique
def generate_window_representation_dict(row, word_avg_dict, sensor_avg_std_dict, interval_dict, window_length, shift_length):
    columns_to_drop = ['sensor_id', 'gesture_id', 'component_id', 'avg_amplitude', 'std_deviation']
    sensor_id = row.sensor_id
    gesture_id = row.gesture_id
    component_id = row.component_id
    key = str((str(gesture_id), component_id, str(sensor_id)))
    sensor_avg_std_dict[key] = (row.avg_amplitude, row.std_deviation)
    row = row.drop(labels=columns_to_drop)
    i=0
    while i < (len(row.index)-window_length):
        if pd.isnull(row[i]):
            break

        temp_key = str((component_id, gesture_id, sensor_id, i))
        k = i
        temp_list = []

        while k < (i + window_length):
            if pd.isnull(row[k]):
                break
            temp_list.append(row[k])
            k += 1

        if len(temp_list) < window_length:
            break;

        temp_avg = st.mean(temp_list)
        
        for key, value in interval_dict.items():
            if temp_avg >= value[0] and temp_avg <= value[1]:
                word_avg_dict[temp_key] = (temp_avg, int(key))
                break
        i += shift_length

    return row

# Delete existing word files
def delete_all_word_files(directory):
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.wrd'):
            os.remove(os.path.join(root,filename))

# Write gesture word files
def write_word_files(directory, word_vector_dict, word_avg_dict, row_avg_std_dict):
    delete_all_word_files(directory)
    prev_gesture_id = None
    prev_component_id = None
    prev_sensor_id = None
    file_path = None
    for key, value in word_vector_dict.items():
        key_tuple = key[1:-1].split(",")
        component_id = key_tuple[0].strip(' ').strip('\'')
        gesture_id = key_tuple[1].strip(' ').strip('\'')
        sensor_id = key_tuple[2].strip(' ').strip('\'')
        time = key_tuple[3].strip(' ').strip('\'')
        if(prev_gesture_id == None or prev_gesture_id != gesture_id):
            file_path = os.path.join(directory, gesture_id) + ".wrd"
            prev_component_id = None
            prev_sensor_id = None
        word_file = open(file_path, "a")  # append mode
        if(prev_component_id == None or prev_component_id != component_id):
            word_file.write("Component ID: " + str(component_id) + "\n")
            prev_sensor_id = None
        if(prev_sensor_id == None or prev_sensor_id != sensor_id):
            word_file.write("\tSensor ID: " + str(sensor_id) + " Avg Amplitude: " + str(row_avg_std_dict[str((gesture_id, component_id, sensor_id))][0]) + \
                " & Standard Deviation of Amplitude: " + str(row_avg_std_dict[str((gesture_id, component_id, sensor_id))][1]) + "\n")
        new_key = (component_id, sensor_id, time) 
        word_file.write("\t\t" + str(new_key) + " -> " + str(value) + ", " + str(word_avg_dict[key][0]) + "\n")
        word_file.close()
        prev_gesture_id = gesture_id
        prev_component_id = component_id
        prev_sensor_id = sensor_id

# Serialize word average dictionary for future tasks
def serialize_word_avg_dict(word_avg_dict):
    with open("intermediate/word_avg_dict.json", "w") as write_file:
        json.dump(word_avg_dict, write_file)

# Serialize word vector dictionary for future tasks
def serialize_word_vector_dict(word_vector_dict):
    with open("intermediate/word_vector_dict.json", "w") as write_file:
        json.dump(word_vector_dict, write_file)

# Serialize sensor average dictionary for future tasks
def serialize_sensor_avg_std_dict(sensor_avg_std_dict):
    with open("intermediate/sensor_avg_std_dict.json", "w") as write_file:
        json.dump(sensor_avg_std_dict, write_file)

# Serialize data paramters for future tasks
def serialize_data_parameters(data):
    with open("intermediate/data_parameters.json", "w") as write_file:
        json.dump(data, write_file)

# Generates gesture word dictionary
def generate_word_dictionary(data):
    print("Loading gesture files...")
    original_df = load_all_the_component_gestures(data['directory'])
    original_df.to_csv('intermediate/original.csv', index=False)

    print("Normalizing gesture files...")
    normalized_df = original_df.apply(lambda x: normalize(x), axis=1)
    normalized_df.to_csv('intermediate/normalized.csv', index=False)

    # normalized_df = pd.read_csv('intermediate/normalized.csv')
    
    print("Quantizing values...")
    interval_dict = get_intervals(data['resolution'])
    quantized_df = normalized_df.apply(lambda x: quantize(x, interval_dict), axis=1)
    quantized_df.to_csv('intermediate/quantized.csv', index=False)

    # quantized_df = pd.read_csv('intermediate/quantized.csv')

    print("Generating word files...")
    word_avg_dict = {}
    word_vector_dict = {}
    sensor_avg_std_dict = {}
    normalized_df.apply(lambda x: generate_window_representation_dict(x, word_avg_dict, sensor_avg_std_dict, interval_dict, data['window_length'], data['shift_length']), axis = 1)
    quantized_df.apply(lambda x: generate_word_vectors(x, word_vector_dict, data['window_length'], data['shift_length']), axis = 1)

    print("Writing word files...")
    write_word_files(data['directory'], word_vector_dict, word_avg_dict, sensor_avg_std_dict)

    print("Serializing objects needed for future tasks...")
    serialize_word_vector_dict(word_vector_dict)
    serialize_word_avg_dict(word_avg_dict)
    serialize_data_parameters(data)
    serialize_sensor_avg_std_dict(sensor_avg_std_dict)

    print("Taskoa complete!")

def set_pandas_display_options() -> None:
    """Set pandas display options."""
    # Ref: https://stackoverflow.com/a/52432757/
    display = pd.options.display

    display.max_columns = 1000
    display.max_rows = 1000
    display.max_colwidth = 199
    display.width = None
    # display.precision = 2  # set as needed

set_pandas_display_options()

def main():
    # Menu
    # while(True):
    #     print("\n******************** Task-1 **********************")
    #     print(" Enter 1 to generate word dictionary")
    #     print(" Enter 2 to exit")
    #     option = input("Enter option: ")
    #     if option == '1':
            data = load_params()
            create_output_directories()
            generate_word_dictionary(data)
        # else:
        #     break

if __name__ == "__main__":
    main()