import inquirer
from lsh import LSH
from task5 import perform_relevance_feeback
import json

"""
Deserialize vector input
"""
def deserialize_vector_file(file_name):
    with open(file_name, "r") as read_file:
        vector = json.load(read_file)
    return vector

"""
Deserialize word position dictionary
"""
def deserialize_word_position_dictionary(file_name):
    with open(file_name, "r") as read_file:
        word_position_dictionary = json.load(read_file)
    return word_position_dictionary

"""
Deserialize component position dictionary
"""
def deserialize_component_position_dictionary(file_name):
    with open(file_name, "r") as read_file:
        component_position_dictionary = json.load(read_file)
    return component_position_dictionary

"""
Deserialize sensor position dictionary
"""
def deserialize_sensor_position_dictionary(file_name):
    with open(file_name, "r") as read_file:
        sensor_position_dictionary = json.load(read_file)
    return sensor_position_dictionary

def task5(ges_no, num_layers, num_hash_per_layer, num_res, vectors, 
    word_position_dictionary, component_position_dictionary, sensor_position_dictionary):

  vector_model = 0
  input_vector_dimension = len(vectors[ges_no][vector_model])

  lsh = LSH(int(num_layers), int(num_hash_per_layer), input_dimension=input_vector_dimension, is_vector_matrix=True, vector_model=vector_model)
  lsh.index(vectors=vectors)
  res = lsh.query(point=vectors[ges_no][vector_model], t=int(num_res), vectors=vectors)

  list_ges = []
  for val in res:
    list_ges.append(val[0])

  # Choosing all the relevent gestures from the result list
  rel_ges = [
    inquirer.Checkbox('Relevent',
                      message="Choose the list of result gestures that are relevent: ",
                      choices=list_ges,
                      ),
  ]
  relevent = inquirer.prompt(rel_ges)
  print("List of relevent gestures: " + str(relevent['Relevent']))

  # Removing all relevent from the result gesture list
  list_ges = [x for x in list_ges if x not in relevent['Relevent']]

  # Code to choose the irrelevent gesture from the result list
  irrel_ges = [
    inquirer.Checkbox('Irrelevent',
                      message="Choose the list of result gestures that are irrelevent: ",
                      choices=list_ges,
                      ),
  ]
  irrelevent = inquirer.prompt(irrel_ges)
  print("List of irrelevent gestures: " + str(irrelevent['Irrelevent']))

  relevant_gestures = relevent['Relevent']
  irrelevant_gestures = irrelevent['Irrelevent']

  modified_query_vector = perform_relevance_feeback(query_gesture=ges_no, vectors=vectors, vector_model=vector_model, relevant_gestures=relevant_gestures, irrelevant_gestures=irrelevant_gestures, word_position_dictionary=word_position_dictionary,
     component_position_dictionary=component_position_dictionary, sensor_position_dictionary=sensor_position_dictionary)

  lsh.query(point=modified_query_vector, t=int(num_res), vectors=vectors)

def task4(ges_no, num_layers, num_hash_per_layer, num_res, vectors, 
    word_position_dictionary, component_position_dictionary, sensor_position_dictionary):
    print("Task 4 implementation")

def main():
  print("Enter the query parameters as input:")
  ges_no = input("Enter the query gesture: ")
  num_layers = input("Enter the number of layers: ")
  num_hash_per_layer = input("Enter the number of hashes per layer: ")
  num_res = input("Enter the number of results to show: ")
  task_choice = input("1)Task-4\n2)Task-5\n Enter the choice: ")

  vectors_file_name = "Phase2/intermediate/vectors_dictionary.json"
  word_position_dictionary_file_name = "Phase2/intermediate/word_position_dictionary.json"
  component_position_dictionary_file_name = "Phase2/intermediate/component_position_dictionary.json"
  sensor_position_dictionary_file_name = "Phase2/intermediate/sensor_position_dictionary.json"

  vectors = deserialize_vector_file(vectors_file_name)
  word_position_dictionary = deserialize_word_position_dictionary(word_position_dictionary_file_name)
  component_position_dictionary = deserialize_component_position_dictionary(component_position_dictionary_file_name)
  sensor_position_dictionary = deserialize_sensor_position_dictionary(sensor_position_dictionary_file_name)

  if task_choice == 4:
    task4(ges_no, num_layers, num_hash_per_layer, num_res, vectors, 
    word_position_dictionary, component_position_dictionary, sensor_position_dictionary)
  else:
    task5(ges_no, num_layers, num_hash_per_layer, num_res, vectors, 
    word_position_dictionary, component_position_dictionary, sensor_position_dictionary)

if __name__=="__main__":
    main()