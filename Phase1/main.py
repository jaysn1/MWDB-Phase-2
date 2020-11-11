import matplotlib.pyplot as plt
import seaborn as sns		
import json, os
from tqdm import tqdm
from helper import * 
from tfidf import *

def task1(config):
	"""
	This function performs task1 for given configrations that include
	(directory with gestures, resolution, window length, shift length)
	and generates " .wrd " files for all gestures into a directory specified 
	in the configration.
	"""
	directory = os.path.join(config['base'], config['data_directory'])
	store_patterns_location = os.path.join(config['base'], config['store_patterns_location'])
	if not os.path.exists(store_patterns_location):
		os.mkdir(store_patterns_location)
	
	# get path of all files that are to be processed for TASK 1
	files = load_files(directory)
	for file in tqdm(files, "Extracting patterns: ", bar_format=bar_format):
		# load the file
		data = load_data(file)
		# normalize the data row-wise for each file seperate
		normalized_data = min_max_scaler(data, feature_range=(-1,1))

		# quantize each row the data based on resolution
		resolution = config['resolution']
		gausian_band_length = get_gausian_band_length(resolution, mean=0, std=0.25, x_range=(-1,1))
		quantized_data = quantize(normalized_data, gausian_band_length)
		
		window = config['window length']
		shift = config['shift length']
		
		patterns = extract_patterns(quantized_data, file, window, shift)
		
		file_name = os.path.basename(file)
		file_name_without_extension = os.path.splitext(file_name)[0]
		file_name_without_extension = os.path.join(store_patterns_location, file_name_without_extension)
		with open(file_name_without_extension+".wrd", "w") as f:
			f.write(str(patterns))
	print("Task 1 complete. Results are saved to {}. \n".format(config['store_patterns_location']))

def task2(config):
	"""
	This function performs task2 for given configrations that include
	(directory with gestures, resolution, window length, shift length)
	and generates a " vector.txt " files combining all " .wrd " files for 
	all gestures into a directory specified in the configration.
	"""
	directory = os.path.join(config['base'], config['store_patterns_location'])
	store_vector_location = os.path.join(config['base'], config['vectors_location'])
	if not os.path.exists(store_vector_location):
		os.mkdir(store_vector_location)

	files, all_words, gesture_words = load_files(directory), set(), []
	data = collections.defaultdict(list)
	for file in tqdm(files, "Reading Files: ", bar_format=bar_format):
		with open(file) as f:
			individual = eval(f.read())
		documents = collections.defaultdict(list)
		w = set()
		for row in individual:
			i, f = row[0][1], row[0][0]
			word = row[1]
			documents[i-1].append(word)
			w.add(word)
		for k,v in documents.items():
			data[f].append(v)
	document_order = [k for k in data.keys()]
	DATA, TF = {}, {}
	## calculating TF-IDF2
	ravel = lambda l: [item for sublist in l for item in sublist]  # this function will be used to convert whole gesture with different sensort to one document
	IDF2 = {}
	for key in tqdm(data.keys(), "Calculating TF-IDF2: ", bar_format=bar_format):
		IDF2[key] = (idf(data[key])[0])
		d = ravel(data[key])
		TF[key] = (tf([d])[0])
		DATA[key] = (d)

	convert_dict_to_ordered_list = lambda dic: [dic[k] for k in document_order]
	convert_ordered_list_to_dict = lambda lis: {k:v for k,v in zip(document_order, lis)}
	IDF, words = idf(convert_dict_to_ordered_list(DATA))
	TFIDF = convert_ordered_list_to_dict(tf_idf(convert_dict_to_ordered_list(DATA), convert_dict_to_ordered_list(TF), IDF, words)[1])
	TFIDF2 = {}
	for key in document_order:
		d = DATA[key]
		TFIDF2[key] = (tf_idf([d], [TF[key]], IDF2[key], words=None)[1][0])
	######### we have TF, TF-IDF, TF-IDF2. now just combining them. Storing IDF to query later ##############
	vectors = {}
	for key in document_order:
		temp = {}
		for k in TF[key].keys():
			temp[k] = (TF[key][k], TFIDF[key][k], TFIDF2[key][k])
		vectors[key] = temp

	file_name = os.path.join(store_vector_location, "vectors.txt")
	with open(file_name, "w") as f:
		f.writelines(str(vectors))
	print("Task 2 complete. Vectors and Data are saved to {}. \n".format(config['vectors_location']))

def task3(config, plot_file_number, plot):
	"""
	This function performs task3 on the given data. Loads the " vectors.txt " which has 
	TF, TF-IDF, TF-IDF2 for all the gestures and combine it with corresponding " .wrd " 
	files to plot the value specified by the user.
	"""
	plot_d = {0: 'TF', 1:'TF-IDF', 2:'TF-IDF2'}
	file_name = "vectors.txt"
	store_vector_location = os.path.join(config['base'], config['vectors_location'])
	with open(os.path.join(store_vector_location, file_name)) as f:
		vectors = eval(f.read())
	# vector = vectors[file_number-1]
	store_vector_location = os.path.join(config['base'], config['store_patterns_location'])
	file_name = str(plot_file_number) + ".wrd"
	with open(os.path.join(store_vector_location, file_name)) as f:
		wrd_file = eval(f.read())
	VALUE = []
	temp = collections.defaultdict(list)
	for row in wrd_file:
		temp[row[0][1]].append(vectors[str(plot_file_number)+".csv"][row[1]][plot])
	for k in sorted(temp.keys()):
		VALUE.append(temp[k])
	fig, ax = plt.subplots()
	ax.set_title("{} for {}.csv".format(plot_d[plot], plot_file_number))
	sns.heatmap(VALUE, cmap=sns.color_palette("Greys"), annot=False,linewidths=.5, ax=ax)
	plt.tight_layout()
	plt.show()

def task4(config):
	"""
	The function performs task4 to calculate similar gestures in the databse. This will take
	gesture label from user as input and returns top K most similar gestures.
	"""
	file_name = "vectors.txt"
	store_vector_location = os.path.join(config['base'], config['vectors_location'])
	with open(os.path.join(store_vector_location, file_name)) as f:
		vectors = eval(f.read())

	files, all_files  = [], []
	for i, f in enumerate(vectors.keys()):
		all_files.append(f)
		files.append("| %10s"%(f))
		if (i+1)%6==0:
			files.append(" |\n")
	# showing database to user to select a gesture
	print("".join(files))
	query_file_name = input("Enter file name to query: ")
	if "{}.csv".format(query_file_name) not in sorted(all_files):
		print("File is not in the directory.")
	else:
		query_file_name = "{}.csv".format(query_file_name)
		index = int(input("What values to use for comparision (TF:0, TF-IDF:1, TF-IDF2:2): "))
		vector1 = {k:v[index] for k,v in vectors[query_file_name].items()}
		similarity_values = {}
		for key in tqdm(vectors.keys(), desc="Finding similar gestures: ", bar_format=bar_format):
			vector2 = {k:v[index] for k,v in vectors[key].items()}
			similarity_values[key] = similarity(vector1, vector2)

		related_gestures = [_[0].split(".")[0] for _ in sorted(similarity_values.items(), key=lambda x: -x[1])][:10]
		return related_gestures

def main():
	# defining some globals
	global bar_format, K
	bar_format='{desc:<25}{percentage:3.0f}%|{bar:15}{r_bar}'
	K = 10

	if os.path.exists("config.json"):
		with open("config.json") as f:
			config = json.load(f)
		while (True):
			print("%10s %s"%("", "CSE 515: Phase 1"))
			task_number = int(input("\n1: Task1\n2: Task2 \n3: Task3\n4: Task4\n5: Exit\n\nEnter number of task to run: : "))
			if task_number==1:
				task1(config)
			elif task_number==2:
				task2(config)
			elif task_number==3:
				files, all_files = [], []
				store_vector_location = os.path.join(config['base'], config['store_patterns_location'])
				for i,f in enumerate(os.listdir(store_vector_location)):
					all_files.append(f)
					files.append("| %10s"%(f))
					if (i+1)%6==0:
						files.append(" |\n")
				print("".join(files))
				file_number = input("Enter file index (12 for vectors_12.txt) to visualize: ")
				if "{}.wrd".format(file_number) not in all_files:
					print("File is not in the directory.")
				else:
					plot = int(input("What to plot? (TF: 0, TFIDF: 1, TFIDF2: 2):  "))
					assert plot in [0,1,2]
					task3(config, file_number, plot)
			elif task_number==4:
				related_gestures = task4(config)
				print("Top {} similar gestures: ".format(K), end="")
				for i in range(K):
					print(related_gestures[i], end=", ")
			elif task_number==5:
				break
			else:
				print("Enter valid input.")
			print("\n\n")
	else:
		print("Add a config.json file.")


if __name__=='__main__':
	main()