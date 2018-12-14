#! python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import ctypes
from myhash import myhash1, myhash2, myhash3

def sketch_update(sketch, sketch_len, index_i, index_j, update_value):
	hash_value = myhash1(index_i, index_j) % sketch_len
	sketch[hash_value] = max(sketch[hash_value], update_value)
	hash_value = myhash2(index_i, index_j) % sketch_len
	sketch[hash_value] = max(sketch[hash_value], update_value)
	hash_value = myhash3(index_i, index_j) % sketch_len
	sketch[hash_value] = max(sketch[hash_value], update_value)


# M is the original matrix waiting for encoding into the hash array -- the return value. 
# base is the number of bits used in M
# com_ratio is the compression ratio.
def sketch_encoding(M, base, com_ratio, float_step):

# now M is float 32 used in training
# 
# @En-Yu, please do float-int transformation for M here
# 
# now M locates in [-128, 127], and is int
	M_int = (M / float_step).astype(np.int32) + (1 << (base - 1))
# now M located in [0, 255]

	i_len = M_int.shape[0]
	j_len = M_int.shape[1]

	sketch_len = int(i_len * j_len * com_ratio + 1)

	sketch = np.zeros((sketch_len,), dtype=np.int32)

	for i in range(0, i_len):
		for j in range(0, j_len):
			sketch_update(sketch, sketch_len, i, j, M_int[i][j])

	return sketch

def sketch_getvalue(sketch, sketch_len, index_i, index_j):
	hash_value1 = myhash1(index_i, index_j) % sketch_len
	hash_value2 = myhash2(index_i, index_j) % sketch_len
	hash_value3 = myhash3(index_i, index_j) % sketch_len
	return min(sketch[hash_value1], sketch[hash_value2], sketch[hash_value3])


# sketch/2/indicator -> np.int32, M -> np.float32
def sketch_decoding(sketch, i_len, j_len, base, com_ratio, float_step):

	M = np.zeros((i_len, j_len), dtype=np.float32)
	sketch_len = int(i_len * j_len * com_ratio + 1)

	for i in range(0, i_len):
		for j in range(0, j_len):
			M[i][j] = (sketch_getvalue(sketch, sketch_len, i, j) - (1 << (base - 1))) * float_step

	return M


def sketch_transform(M, base, com_ratio, float_step):
	sketch = sketch_encoding(M, base, com_ratio, float_step)
	M1 = sketch_decoding(sketch, M.shape[0], M.shape[1], base, com_ratio, float_step)
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