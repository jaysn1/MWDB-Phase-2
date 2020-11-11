import os
from scipy.integrate import quad
import scipy.stats
import math

def load_files(directory):
	"""
	given path of directory, returns list of absolute paths for all files inside that directory 
	"""
	assert isinstance(directory, str)

	if os.path.exists(directory):
		files = []
		for f in os.listdir(directory):
			if os.path.isfile(os.path.join(directory, f)):
				files.append(os.path.join(directory, f))
		return files
	else:
		raise FileNotFoundError("{} doesnot exists.".format(directory))

def load_data(file_path):
	"""
	given the location of file, load the data from the file
	returns: list of list
	"""
	assert isinstance(file_path, str)

	with open(file_path) as f:
		data = []
		for line in f.readlines():
			line = line.split(",")
			data.append(list(map(lambda x: float(x), line)))
	return data

def min_max_scaler(data, feature_range=(-1,1)):
	"""
	given a dataframe and feature range to normalize the data, returns the modified dataframe with row-wise normalized data
	"""
	for i in range(len(data)):
		x = data[i]
		ma, mi = max(x), min(x)
		if ma==mi:
			x = [0]*len(x)
		else:
			x = list(map(lambda x: feature_range[0] + (feature_range[1]-feature_range[0])*(x-mi)/(ma-mi), x))
		data[i] = x
	return data

def get_gausian_band_length(resolution, mean=0, std=0.25, x_range=(-1,1)):
	"""
	given resolution, mean, standard deviation and range, return a list with length of gausian bands
	"""
	assert isinstance(resolution, int)

	normal_distribution_function = lambda x: scipy.stats.norm.pdf(x,mean,std) # A probability density function that gives
	# def normal_distribution_function(x):
	# 	return (1/(std * math.sqrt(2*math.pi)))*math.exp(-1*((x-mean)**2)/(2*std**2))

	length, x1, x2 = [], -1, 1
	denominator, err = quad(normal_distribution_function, x1, x2)  # denominator is common for all
	
	for i in range(1, 2*resolution+1):
		x1 = (i-resolution-1)/resolution
		x2 = (i-resolution)/resolution
		res, err = quad(normal_distribution_function, x1, x2)
		length.append(2*res/denominator)
	return length

def quantize(data, gausian_band_length):
	"""
	given the data matrix and gausian bands, convert it to digital format
	"""
	assert isinstance(data, list)
	assert isinstance(data[0], list)
	assert isinstance(gausian_band_length, list)

	X = [-1]
	for x in gausian_band_length:
		X.append(X[-1]+x)
	quantized_data = []
	X[0], Y = -1.1, []
	for i in range(len(data)):
		row = data[i]
		temp = []
		for value in row:
			for ind, x in enumerate(X):
				if value <= x:	# lower bound will be considerd
					temp.append(ind)
					break
		assert len(row) == len(temp)
		quantized_data.append(temp)
	return quantized_data

def extract_patterns(data, f, window, shift):
	"""
	given the dataframe, file name, window length, shift length, create a new file in the given directory with patterns in it
	"""
	assert isinstance(data, list)
	assert isinstance(data[0], list)
	assert isinstance(window, int)
	assert isinstance(shift, int)

	file_name = os.path.basename(f)
	file_name_without_extension = os.path.splitext(file_name)[0]	
	patterns = []
		
	for s in range(len(data)):
		row, temp = data[s], []
		for indx in range(0, len(row)-window+1, shift):
			win = row[indx:indx+window]
			idx = (file_name, s+1, indx)
			temp.append((idx, tuple(win)))
		assert (len(row)-window)//shift + 1 == len(temp) # making sure the length of generated vectors is valid
		patterns.extend(temp)
	return patterns