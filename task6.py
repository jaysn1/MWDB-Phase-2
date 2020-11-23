import inquirer
from lsh import deserialize_vector_file, LSH

print("Enter the query parameters as input:")
ges_no = input("Enter the number of the query gesture: ")
num_layers = input("Enter the number of layers: ")
num_hash_per_layer = input("Enter the number of hashes per layer: ")
num_res = input("Enter the number of results to show: ")

file_name = "Phase2/intermediate/LDA_gesture_gesture_similarity_dictionary.json"
vectors = deserialize_vector_file(file_name)

test = {}
for k,v in vectors.items():
    x = list(v.values())
    test[k] = x

gestures = list(vectors.keys())
input_vector_dimension = len(test[gestures[0]])
lsh = LSH(int(num_layers), int(num_hash_per_layer), input_dimension=input_vector_dimension)
lsh.index(test)
res = lsh.query(test[ges_no], int(num_res), vectors=test)

print(res)
print(type(res))

list_ges = []
for val in res:
  list_ges.append(val[0])

print(list_ges)

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