#! python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import ctypes
from myhash import myhash1, myhash2, myhash3

def sketch_update(sketch, sketch_len, index_i, index_j, update_value):
	hash_value = myhash1(index_i, index_j) % sketch_len
	sketch[hash_value] = min(sketch[hash_value], update_value)
	hash_value = myhash2(index_i, index_j) % sketch_len
	sketch[hash_value] = min(sketch[hash_value], update_value)
	hash_value = myhash3(index_i, index_j) % sketch_len
	sketch[hash_value] = min(sketch[hash_value], update_value)

# M is the original matrix waiting for encoding into the hash array -- the return value. 
# base is the number of bits used in M
# com_ratio is the compression ratio.
def sketch_encoding(M, base, com_ratio, float_step):

# now M is float 32 used in training
# 
# @En-Yu, please do float-int transformation for M here
# 
# now M locates in [-128, 127], and is int
	M_int = (M / float_step).astype(np.int32)
	
	i_len = M_int.shape[0]
	j_len = M_int.shape[1]

	sketch_len = int((i_len * j_len * (com_ratio - 2.0/base)) / 2 + 1)

	sketch1 = np.zeros((sketch_len,), dtype=np.int32)
	sketch1[:] = ((1 << base) - 1)
	sketch2 = np.zeros((sketch_len,), dtype=np.int32)
	sketch2[:] = ((1 << base) - 1)
	indicator = np.zeros((i_len,j_len), dtype=np.int32)

	for i in range(0, i_len):
		for j in range(0, j_len):
			if(M_int[i][j] == 0):
				indicator[i][j] = 0
			elif(M_int[i][j] > 0):
				indicator[i][j] = 1
				sketch_update(sketch1, sketch_len, i, j, M_int[i][j])
			else:
				indicator[i][j] = -1
				sketch_update(sketch2, sketch_len, i, j, -M_int[i][j])

	return [sketch1, sketch2, indicator]

def sketch_getvalue(sketch, sketch_len, index_i, index_j):
	hash_value1 = myhash1(index_i, index_j) % sketch_len
	hash_value2 = myhash2(index_i, index_j) % sketch_len
	hash_value3 = myhash3(index_i, index_j) % sketch_len
	return max(sketch[hash_value1], sketch[hash_value2], sketch[hash_value3])


# sketch1/2/indicator -> np.int32, M -> np.float32
def sketch_decoding(sketch1, sketch2, indicator, base, com_ratio, float_step):
	i_len = indicator.shape[0]
	j_len = indicator.shape[1]

	M = np.zeros((i_len, j_len), dtype=np.float32)
	sketch_len = int((i_len * j_len * (com_ratio - 2.0/base)) / 2 + 1)


	for i in range(0, i_len):
		for j in range(0, j_len):
			if(indicator[i][j] == 0):
				M[i][j] = 0
			elif(indicator[i][j] == 1):
				M[i][j] = sketch_getvalue(sketch1, sketch_len, i, j) * float_step
			else:
				M[i][j] = -sketch_getvalue(sketch2, sketch_len, i, j) * float_step

# now M is natural float 32 
# 
# @En-Yu, please do float-float transformation for M here
# 
# now M is float 32 used in training

	return M


def sketch_transform(M, base, com_ratio, float_step):
	sketch = sketch_encoding(M, base, com_ratio, float_step)
	M1 = sketch_decoding(sketch[0], sketch[1], sketch[2], base, com_ratio, float_step)
	return M1


def test():
	com_ratio = 0.9
	shape1 = 64
	shape2 =64
	base = 8
	float_step = 0.008381030138801126
	half_base_mask = 1<<(base-1)


	# M = np.random.uniform(-half_base_mask * float_step, (half_base_mask - 1) * float_step, size=(shape1, shape1))

	M = np.load( "to_yang.npy" )
	print(M)

	M1 = sketch_transform(M, base, com_ratio, float_step)
	print(M1)

if __name__ == '__main__':
	test()
