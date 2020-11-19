import numpy as np
import json

class LSH:
    """
    Layer table
    """
    class LayerTable:
        def __init__(self):
            self.table = dict()

        def get_val(self, key):
            return self.table[key]

        def append_val(self, key, val):
            self.table.setdefault(key, []).append(val)

        def get_list(self, key):
            return self.table.get(key, [])

        def print_table(self):
            print(self.table)

    """
    Constructor for LSH
    """
    def __init__(self, num_layers, num_hash_per_layer, input_dimension):
        self.num_layers = num_layers
        self.num_hash_per_layer = num_hash_per_layer
        self.input_dimension = input_dimension
        self.uniform_hyperplanes = [np.random.randn(self.num_hash_per_layer, self.input_dimension) for i in range(self.num_layers)]
        self.layertables = [self.LayerTable() for i in range(self.num_layers)]
    
    """
    Hash data point
    """
    def _hash(self, layer_num, point):
        point = np.array(point)
        projections = np.dot(self.uniform_hyperplanes[layer_num], point)
        return "".join(['1' if i > 0 else '0' for i in projections])

    """
    Index data point
    """
    def index(self, vectors):
        for k,v in vectors.items():
            for i in range(self.num_layers):
                self.layertables[i].append_val(self._hash(i, v[0]), k)
        return
    
    """
    Query data point
    """
    def query(self, point, t, vectors):
        potential_candidates = set()
        num_buckets_searched = 0
        for i in range(self.num_layers):
            binary_hash = self._hash(i, point)
            potential_candidates.update(self.layertables[i].get_list(binary_hash))
            num_buckets_searched += 1
        results = []
        for candidate in potential_candidates:
            results.append((candidate, self.calculate_l2_distance(vectors[candidate][0], point)))
        results.sort(key = lambda x: x[1])
        if t < len(results):
            results = results[0:t]
        print("Number of buckets searched: " + str(num_buckets_searched))
        print("Number of candidates considered: " + str(len(potential_candidates)))
        print("Results: " + str(results))
    
    """
    Euclidean distance
    """
    def calculate_l2_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1)-np.array(point2))

    
"""
Deserialize vector input
"""
def deserialize_vector_file(file_name):
    with open(file_name, "r") as read_file:
        vector = json.load(read_file)
    return vector

def main():
    file_name = "Phase2/intermediate/vectors_dictionary.json"
    vectors = deserialize_vector_file(file_name)

    gestures = list(vectors.keys())
    input_vector_dimension = len(vectors[gestures[0]][0])
    lsh = LSH(num_layers=4, num_hash_per_layer=8, input_dimension=input_vector_dimension)
    lsh.index(vectors)
    lsh.query(point=vectors['63'][0], t=15, vectors=vectors)

if __name__=="__main__":
    main()
