import json

def read_vectors(vectors_dir="intermediate/vectors_dictionary.json", word_position_dir="intermediate/word_position_dictionary.json"):
    """
    Reads 
    """
    # reading word position dictionary
    with open(word_position_dir) as f:
        word_position_dictionary = json.load(f)
    # reading vector file
    with open(vectors_dir) as f:
        vectors_dictionary = json.load(f)
    TF, TFIDF = {}, {}
    for gesture_id in vectors_dictionary:
        tf, tfidf = vectors_dictionary[gesture_id]
        tf = {word: tf[word_position_dictionary[word]] for word in word_position_dictionary}
        tfidf = {word: tfidf[word_position_dictionary[word]] for word in word_position_dictionary}
        TF[gesture_id] = tf
        TFIDF[gesture_id] = tfidf
    return TF, TFIDF

if __name__=='__main__':
    read_vectors()