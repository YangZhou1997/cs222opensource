## Parameters: 
* com_ratio = 0.9: the encoded sketch size / the matrix size. 
* base = 8: the size of one counter/weight in the matrix
* float_step = 0.008381030138801126: float step for the float number

## Functions of each lib
1. sketch_coding_maxsketch.py: using one max sketch to code the whole matrix (adding (1 << (base - 1)) makes matrix positive)
	* Usage: `M1 = sketch_transform(M, base, com_ratio, float_step)`
	* Suitable for 8bit and 2bit
	
1. sketch_coding_maxsketch_double.py: using two max sketches and an idencator matrix to code the positive and negative matrix weights seperately, the indencator matrix cannot be compressed. 
	* Usage: `M1 = sketch_transform(M, base, com_ratio, float_step)`
	* Suitable for 8bit

1. sketch_coding_maxsketch_pruning.py: using one max sketch and an idencator matrix to code the zero and non-zero matrix weights seperately, the indencator matrix cannot be compressed.
	* Usage `M1 = sketch_transform(M, base, com_ratio, float_step)`
	* Suitable for 8bit
	
1. sketch_coding_bf.py: using one Bloom filter to coding the binary weight matrix. 
	* Usage `M1 = sketch_transform(M, com_ratio)`
	* Suitable for 1bit

## Experiment
* Since Min and Max sketch behaves similar, we only experiment on Max. 

* Three model: 1bit, 8bits, 8bits w/ pruning. 
	* 1bit: sketch_coding_bf
	* 8bits: sketch_coding_maxsketch, sketch_coding_maxsketch_double, sketch_coding_maxsketch_pruning
	* 8bits w/ pruning: sketch_coding_maxsketch, sketch_coding_maxsketch_double, sketch_coding_maxsketch_pruning

## About the com_ratio
* Try small value even like 0.2. (I am not sure whether 0.2 works or not)  
